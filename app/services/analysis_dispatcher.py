from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from kombu.exceptions import KombuError

from app.agents.workflow import ClusterMindWorkflow
from app.db.repositories import WorkflowLifecycleRepository
from app.tasks.celery_app import ANALYSIS_QUEUE, celery_app


class AnalysisDispatchError(RuntimeError):
    """Raised when a durable analysis request cannot be dispatched safely."""


@dataclass(frozen=True, slots=True)
class AnalysisDispatch:
    workflow_id: str
    task_id: str
    status: str
    dispatch_status: str


class AnalysisDispatcher:
    def __init__(self, lifecycle: WorkflowLifecycleRepository) -> None:
        self.lifecycle = lifecycle


    def dispatch_pending(self, workflow_id: str) -> AnalysisDispatch:
        state = self.lifecycle.get_status(workflow_id)
        if state.dispatch_status in {"dispatched", "claimed"} and state.task_id:
            return AnalysisDispatch(
                workflow_id=state.workflow_id,
                task_id=state.task_id,
                status=state.status,
                dispatch_status=state.dispatch_status,
            )
        if state.dispatch_status != "pending":
            raise AnalysisDispatchError(
                f"Workflow cannot be redispatched from {state.dispatch_status}"
            )

        task_id = str(uuid4())
        marked = self.lifecycle.mark_dispatched(workflow_id, task_id=task_id)
        try:
            celery_app.send_task(
                "clustermind.analyze_workflow",
                kwargs={
                    "workflow_id": state.workflow_id,
                    "incident_id": state.incident_id,
                    "use_llm": bool(marked.requested_use_llm),
                },
                task_id=task_id,
                queue=ANALYSIS_QUEUE,
                routing_key=ANALYSIS_QUEUE,
            )
        except (KombuError, OSError, RuntimeError) as exc:
            self.lifecycle.mark_dispatch_failed(
                workflow_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise AnalysisDispatchError(
                f"Unable to redispatch workflow {workflow_id}"
            ) from exc
        return AnalysisDispatch(
            workflow_id=workflow_id,
            task_id=task_id,
            status=marked.status,
            dispatch_status=marked.dispatch_status,
        )

    def dispatch(
        self,
        incident_id: str,
        *,
        use_llm: bool,
    ) -> AnalysisDispatch:
        normalized = incident_id.strip()
        if not normalized:
            raise ValueError("incident_id cannot be blank")
        workflow_id = ClusterMindWorkflow.workflow_id_for(normalized)
        queued = self.lifecycle.create_queued(
            workflow_id=workflow_id,
            incident_id=normalized,
            use_llm=use_llm,
        )
        if queued.dispatch_status in {"dispatched", "claimed"} and queued.task_id:
            return AnalysisDispatch(
                workflow_id=workflow_id,
                task_id=queued.task_id,
                status=queued.status,
                dispatch_status=queued.dispatch_status,
            )
        if queued.dispatch_status != "pending":
            raise AnalysisDispatchError(
                f"Workflow cannot be dispatched from {queued.dispatch_status}"
            )

        task_id = str(uuid4())
        marked = self.lifecycle.mark_dispatched(workflow_id, task_id=task_id)
        try:
            celery_app.send_task(
                "clustermind.analyze_workflow",
                kwargs={
                    "workflow_id": workflow_id,
                    "incident_id": normalized,
                    "use_llm": use_llm,
                },
                task_id=task_id,
                queue=ANALYSIS_QUEUE,
                routing_key=ANALYSIS_QUEUE,
            )
        except (KombuError, OSError, RuntimeError) as exc:
            self.lifecycle.mark_dispatch_failed(
                workflow_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise AnalysisDispatchError(
                f"Unable to dispatch workflow {workflow_id}"
            ) from exc

        return AnalysisDispatch(
            workflow_id=workflow_id,
            task_id=task_id,
            status=marked.status,
            dispatch_status=marked.dispatch_status,
        )
