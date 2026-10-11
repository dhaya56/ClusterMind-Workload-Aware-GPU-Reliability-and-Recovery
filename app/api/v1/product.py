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
from app.schemas.agents import WorkflowResult, utc_now
from app.schemas.api import (
    AnalyzeIncidentRequest,
    IncidentDetailResponse,
    IncidentListItem,
    IncidentListResponse,
    OverviewResponse,
    PlatformComponent,
    PlatformStatusResponse,
    WorkflowDecisionRequest,
    WorkflowListResponse,
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
    response_model=WorkflowResult,
    status_code=status.HTTP_200_OK,
)
async def analyze_incident(
    incident_id: str,
    request_body: AnalyzeIncidentRequest,
    runtime: RuntimeDependency,
) -> WorkflowResult:
    try:
        return await run_in_threadpool(
            runtime.analyze,
            incident_id,
            use_llm=request_body.use_llm,
            force=request_body.force,
        )
    except IncidentRepositoryError as exc:
        raise _repository_error(exc) from exc
    except WorkflowConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except WorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Incident analysis failed safely",
        ) from exc


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    runtime: RuntimeDependency,
    limit: int = Query(default=100, ge=1, le=1000),
) -> WorkflowListResponse:
    try:
        items = await run_in_threadpool(runtime.workflow.store.list, limit=limit)
    except WorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow records are temporarily unavailable",
        ) from exc
    return WorkflowListResponse(items=list(items), count=len(items), limit=limit)


@router.get("/workflows/{workflow_id}", response_model=WorkflowResult)
async def workflow_detail(
    workflow_id: str,
    runtime: RuntimeDependency,
) -> WorkflowResult:
    try:
        return await run_in_threadpool(runtime.workflow.store.get, workflow_id)
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
