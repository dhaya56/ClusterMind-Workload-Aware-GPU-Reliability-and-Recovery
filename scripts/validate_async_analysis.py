from __future__ import annotations

from sqlalchemy import delete

from app.agents.workflow import ClusterMindWorkflow
from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.models import (
    AgentExecutionRecord,
    ApprovalDecisionRecord,
    AuditEvent,
    OutboxEvent,
    WorkflowRun,
)
from app.db.repositories import PostgresWorkflowRepository, WorkflowLifecycleRepository
from app.services.analysis_dispatcher import AnalysisDispatcher

INCIDENT_ID = "INC-2026-0032"


def clear(runtime: DatabaseRuntime, workflow_id: str) -> None:
    with runtime.transaction() as session:
        row = session.query(WorkflowRun).filter_by(workflow_id=workflow_id).one_or_none()
        if row is None:
            return
        session.execute(delete(OutboxEvent).where(OutboxEvent.aggregate_id == workflow_id))
        session.execute(delete(ApprovalDecisionRecord).where(ApprovalDecisionRecord.workflow_run_id == row.id))
        session.execute(delete(AuditEvent).where(AuditEvent.workflow_run_id == row.id))
        session.execute(delete(AgentExecutionRecord).where(AgentExecutionRecord.workflow_run_id == row.id))
        session.delete(row)


def main() -> None:
    settings = get_settings()
    runtime = DatabaseRuntime(settings)
    lifecycle = WorkflowLifecycleRepository(runtime)
    workflows = PostgresWorkflowRepository(runtime)
    dispatcher = AnalysisDispatcher(lifecycle)
    workflow_id = ClusterMindWorkflow.workflow_id_for(INCIDENT_ID)
    try:
        clear(runtime, workflow_id)
        dispatched = dispatcher.dispatch(INCIDENT_ID, use_llm=False)
        result = dispatcher.lifecycle.runtime.engine
        async_result = __import__("app.tasks.celery_app", fromlist=["celery_app"]).celery_app.AsyncResult(dispatched.task_id)
        value = async_result.get(timeout=180)
        persisted = workflows.get(workflow_id)
        with runtime.read_session() as session:
            row = session.query(WorkflowRun).filter_by(workflow_id=workflow_id).one()
            dispatch_status = row.dispatch_status
            task_id = row.task_id
        assert value["workflow_id"] == workflow_id
        assert value["duplicate_delivery"] is False
        assert persisted.status in {"awaiting_approval", "suppressed", "more_evidence_requested"}
        assert dispatch_status == "completed"
        assert task_id == dispatched.task_id
        print("SUCCESS: asynchronous analysis executed through Celery")
        print("Workflow:", workflow_id)
        print("Task:", task_id)
        print("Workflow status:", persisted.status)
        print("Dispatch status:", dispatch_status)
    finally:
        clear(runtime, workflow_id)
        runtime.dispose()
        print("SUCCESS: asynchronous validation workflow cleaned up")


if __name__ == "__main__":
    main()
