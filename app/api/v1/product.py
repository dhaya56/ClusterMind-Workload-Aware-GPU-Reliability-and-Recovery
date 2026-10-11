from __future__ import annotations

from datetime import timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from app.agents.workflow import (
    WorkflowConflictError,
    WorkflowError,
    WorkflowNotFoundError,
)
from app.db import DatabaseRuntime
from app.db.repositories import (
    WorkflowLifecycleRepository,
    WorkflowRepositoryConflictError,
    WorkflowRepositoryError,
)
from app.services.analysis_dispatcher import (
    AnalysisDispatchError,
    AnalysisDispatcher,
)
from app.schemas.agents import WorkflowResult, utc_now
from app.schemas.api import (
    AnalysisDispatchResponse,
    AnalyzeIncidentRequest,
    IncidentDetailResponse,
    IncidentListItem,
    IncidentListResponse,
    OverviewResponse,
    PlatformComponent,
    PlatformStatusResponse,
    WorkflowDecisionRequest,
    WorkflowListResponse,
    WorkflowStatusResponse,
)
from app.services.incidents import IncidentRepositoryError
from app.services.ollama import OllamaService, OllamaServiceError
from app.services.workflow_runtime import WorkflowRuntime


router = APIRouter(tags=["operations"])


def get_runtime(request: Request) -> WorkflowRuntime:
    runtime = getattr(request.app.state, "workflow_runtime", None)
    if not isinstance(runtime, WorkflowRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow runtime is not available",
        )
    return runtime


RuntimeDependency = Annotated[WorkflowRuntime, Depends(get_runtime)]


def _ollama_health_snapshot() -> dict[str, object]:
    with OllamaService() as service:
        return service.health()


def _repository_error(exc: IncidentRepositoryError) -> HTTPException:
    message = str(exc)
    code = (
        status.HTTP_404_NOT_FOUND
        if message.startswith("Unknown incident_id")
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return HTTPException(status_code=code, detail=message)


def _result_is_available(status_value: str) -> bool:
    return status_value not in {"created", "queued", "running"}


def _json_workflow_status(result: WorkflowResult) -> WorkflowStatusResponse:
    return WorkflowStatusResponse(
        workflow_id=result.workflow_id,
        incident_id=result.incident_id,
        status=result.status,
        current_stage=str(result.metadata.get("route") or result.status),
        dispatch_status="not_dispatched",
        retry_count=0,
        version=1,
        updated_at=result.updated_at,
        result_available=True,
        result=result,
    )


async def _workflow_status_response(
    state,
    *,
    runtime: WorkflowRuntime,
) -> WorkflowStatusResponse:
    result_available = _result_is_available(state.status)
    result = None
    if result_available:
        try:
            result = await run_in_threadpool(
                runtime.workflow.store.get,
                state.workflow_id,
            )
        except (WorkflowNotFoundError, WorkflowError):
            result_available = False

    return WorkflowStatusResponse(
        workflow_id=state.workflow_id,
        incident_id=state.incident_id,
        status=state.status,
        current_stage=state.current_stage,
        dispatch_status=state.dispatch_status,
        task_id=state.task_id,
        worker_id=state.worker_id,
        retry_count=state.retry_count,
        version=state.version,
        queued_at=state.queued_at,
        dispatched_at=state.dispatched_at,
        claimed_at=state.claimed_at,
        last_heartbeat_at=state.last_heartbeat_at,
        updated_at=state.updated_at,
        result_available=result_available,
        result=result,
    )


@router.get("/overview", response_model=OverviewResponse)
async def overview(runtime: RuntimeDependency) -> OverviewResponse:
    try:
        base = await run_in_threadpool(runtime.repository.overview)
        workflows = await run_in_threadpool(runtime.workflow.store.list, limit=1000)
    except (IncidentRepositoryError, WorkflowError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Overview data is temporarily unavailable",
        ) from exc

    workflow_status_counts: dict[str, int] = {}
    for item in workflows:
        workflow_status_counts[item.status] = (
            workflow_status_counts.get(item.status, 0) + 1
        )
    return OverviewResponse(
        **base,
        workflow_count=len(workflows),
        workflow_status_counts=dict(sorted(workflow_status_counts.items())),
        pending_approval_count=sum(
            item.status == "awaiting_approval" for item in workflows
        ),
    )


@router.get("/incidents", response_model=IncidentListResponse)
async def list_incidents(
    runtime: RuntimeDependency,
    severity: str | None = Query(default=None, max_length=20),
    incident_type: str | None = Query(default=None, max_length=100),
    node_id: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> IncidentListResponse:
    try:
        items = await run_in_threadpool(
            runtime.repository.list_incidents,
            severity=severity,
            incident_type=incident_type,
            node_id=node_id,
            limit=limit,
            offset=offset,
        )
    except IncidentRepositoryError as exc:
        raise _repository_error(exc) from exc
    return IncidentListResponse(
        items=[IncidentListItem.model_validate(item.model_dump()) for item in items],
        count=len(items),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/incidents/{incident_id}",
    response_model=IncidentDetailResponse,
)
async def incident_detail(
    incident_id: str,
    runtime: RuntimeDependency,
) -> IncidentDetailResponse:
    try:
        incident = await run_in_threadpool(
            runtime.repository.get_incident,
            incident_id,
        )
        runbook = runtime.repository.runbooks.get(incident.incident_type)
    except IncidentRepositoryError as exc:
        raise _repository_error(exc) from exc
    return IncidentDetailResponse(
        **incident.model_dump(),
        runbook_id=runbook.runbook_id,
        runbook_title=runbook.title,
    )


@router.post(
    "/incidents/{incident_id}/analyze",
    response_model=AnalysisDispatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_incident(
    incident_id: str,
    request_body: AnalyzeIncidentRequest,
    runtime: RuntimeDependency,
    request: Request,
) -> AnalysisDispatchResponse:
    if request_body.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Forced asynchronous reanalysis is not enabled",
        )

    database_runtime = getattr(request.app.state, "database_runtime", None)
    if not isinstance(database_runtime, DatabaseRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database runtime is not available",
        )

    try:
        await run_in_threadpool(runtime.repository.get_incident, incident_id)
        dispatcher = AnalysisDispatcher(
            WorkflowLifecycleRepository(database_runtime)
        )
        dispatched = await run_in_threadpool(
            dispatcher.dispatch,
            incident_id,
            use_llm=request_body.use_llm,
        )
        return AnalysisDispatchResponse(
            workflow_id=dispatched.workflow_id,
            task_id=dispatched.task_id,
            status=dispatched.status,
            dispatch_status=dispatched.dispatch_status,
            status_url=(
                f"/api/v1/workflows/{dispatched.workflow_id}"
            ),
        )
    except IncidentRepositoryError as exc:
        raise _repository_error(exc) from exc
    except WorkflowRepositoryConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (AnalysisDispatchError, WorkflowRepositoryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Incident analysis could not be queued safely",
        ) from exc


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    request: Request,
    runtime: RuntimeDependency,
    limit: int = Query(default=100, ge=1, le=1000),
) -> WorkflowListResponse:
    database_runtime = getattr(request.app.state, "database_runtime", None)
    if isinstance(database_runtime, DatabaseRuntime) and (
        runtime.workflow_store_detail == "postgresql"
    ):
        lifecycle = WorkflowLifecycleRepository(database_runtime)
        try:
            states = await run_in_threadpool(
                lifecycle.list_statuses,
                limit=limit,
            )
            items = [
                await _workflow_status_response(
                    state,
                    runtime=runtime,
                )
                for state in states
            ]
        except WorkflowRepositoryError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow records are temporarily unavailable",
            ) from exc
        return WorkflowListResponse(
            items=items,
            count=len(items),
            limit=limit,
        )

    try:
        results = await run_in_threadpool(
            runtime.workflow.store.list,
            limit=limit,
        )
    except WorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow records are temporarily unavailable",
        ) from exc
    items = [
        _json_workflow_status(result)
        for result in results
    ]
    return WorkflowListResponse(items=items, count=len(items), limit=limit)


@router.get(
    "/workflows/{workflow_id}",
    response_model=WorkflowStatusResponse,
)
async def workflow_detail(
    workflow_id: str,
    request: Request,
    runtime: RuntimeDependency,
) -> WorkflowStatusResponse:
    database_runtime = getattr(request.app.state, "database_runtime", None)
    if isinstance(database_runtime, DatabaseRuntime) and (
        runtime.workflow_store_detail == "postgresql"
    ):
        lifecycle = WorkflowLifecycleRepository(database_runtime)
        try:
            state = await run_in_threadpool(
                lifecycle.get_status,
                workflow_id,
            )
            return await _workflow_status_response(
                state,
                runtime=runtime,
            )
        except WorkflowRepositoryNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except WorkflowRepositoryError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow record is temporarily unavailable",
            ) from exc

    try:
        result = await run_in_threadpool(
            runtime.workflow.store.get,
            workflow_id,
        )
        return _json_workflow_status(result)
    except WorkflowNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except WorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow record is temporarily unavailable",
        ) from exc


@router.post(
    "/workflows/{workflow_id}/decision",
    response_model=WorkflowResult,
)
async def decide_workflow(
    workflow_id: str,
    request_body: WorkflowDecisionRequest,
    runtime: RuntimeDependency,
) -> WorkflowResult:
    if request_body.decision != "approved" and not request_body.reason:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A reason is required for rejection or more evidence",
        )
    try:
        return await run_in_threadpool(
            runtime.decide,
            workflow_id,
            decision=request_body.approval_decision(),
            actor=request_body.actor,
            reason=request_body.reason,
        )
    except WorkflowNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except WorkflowConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except WorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow decision could not be persisted",
        ) from exc


@router.get("/platform/status", response_model=PlatformStatusResponse)
async def platform_status(runtime: RuntimeDependency) -> PlatformStatusResponse:
    components: list[PlatformComponent] = []
    try:
        overview_data = await run_in_threadpool(runtime.repository.overview)
        components.append(
            PlatformComponent(
                name="incident_repository",
                status="healthy",
                detail=f"{overview_data['incident_count']} incidents available",
            )
        )
    except IncidentRepositoryError:
        components.append(
            PlatformComponent(
                name="incident_repository",
                status="unavailable",
                detail="Incident evidence could not be loaded",
            )
        )

    try:
        health = await run_in_threadpool(_ollama_health_snapshot)
        components.append(
            PlatformComponent(
                name="ollama",
                status=("healthy" if health["model_available"] else "degraded"),
                detail=(
                    f"Configured model {health['configured_model']} is available"
                    if health["model_available"]
                    else f"Configured model {health['configured_model']} is missing"
                ),
            )
        )
    except OllamaServiceError:
        components.append(
            PlatformComponent(
                name="ollama",
                status="unavailable",
                detail="Ollama health check failed",
            )
        )

    components.append(
        PlatformComponent(
            name="workflow_store",
            status="healthy",
            detail=runtime.workflow_store_detail,
        )
    )
    overall = (
        "unavailable"
        if any(item.status == "unavailable" for item in components)
        else "degraded"
        if any(item.status == "degraded" for item in components)
        else "healthy"
    )
    return PlatformStatusResponse(
        status=overall,
        timestamp=utc_now().astimezone(timezone.utc),
        components=components,
        capabilities={
            "langgraph_orchestration": True,
            "human_approval_required": True,
            "recovery_execution_enabled": False,
            "ollama_root_cause_reasoning": True,
            "deterministic_recovery_planning": True,
        },
    )
