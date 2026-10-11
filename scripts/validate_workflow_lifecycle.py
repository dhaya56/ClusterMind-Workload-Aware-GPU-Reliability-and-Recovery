from __future__ import annotations

from datetime import timedelta

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
from app.db.repositories import WorkflowLifecycleRepository
from app.schemas.agents import utc_now


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
    incident_id = "INC-LIFECYCLE-VALIDATION"
    workflow_id = ClusterMindWorkflow.workflow_id_for(incident_id)
    task_id = "task-lifecycle-validation"
    worker_id = "worker-lifecycle-validation"
    runtime = DatabaseRuntime(get_settings())
    repository = WorkflowLifecycleRepository(runtime)
    try:
        clear(runtime, workflow_id)
        queued = repository.create_queued(
            workflow_id=workflow_id,
            incident_id=incident_id,
            use_llm=False,
        )
        repeated = repository.create_queued(
            workflow_id=workflow_id,
            incident_id=incident_id,
            use_llm=False,
        )
        assert repeated == queued
        dispatched = repository.mark_dispatched(workflow_id, task_id=task_id)
        claimed = repository.claim(
            workflow_id,
            task_id=task_id,
            worker_id=worker_id,
        )
        heartbeat = repository.heartbeat(
            workflow_id,
            task_id=task_id,
            worker_id=worker_id,
            current_stage="gpu_telemetry",
        )
        assert queued.dispatch_status == "pending"
        assert dispatched.dispatch_status == "dispatched"
        assert claimed.dispatch_status == "claimed"
        assert heartbeat.current_stage == "gpu_telemetry"

        with runtime.transaction() as session:
            row = session.query(WorkflowRun).filter_by(workflow_id=workflow_id).one()
            now = utc_now()
            row.queued_at = now - timedelta(minutes=40)
            row.dispatched_at = now - timedelta(minutes=35)
            row.claimed_at = now - timedelta(minutes=30)
            row.last_heartbeat_at = now - timedelta(minutes=20)

        recovered = repository.recover_stale_claims(
            stale_before=utc_now() - timedelta(minutes=5)
        )
        assert len(recovered) == 1
        assert recovered[0].dispatch_status == "pending"
        assert recovered[0].retry_count == 1
        assert recovered[0].task_id is None

        second_task = "task-lifecycle-validation-retry"
        repository.mark_dispatched(workflow_id, task_id=second_task)
        repository.claim(
            workflow_id,
            task_id=second_task,
            worker_id=worker_id,
        )
        completed = repository.mark_completed(
            workflow_id,
            task_id=second_task,
            worker_id=worker_id,
        )
        assert completed.dispatch_status == "completed"

        print("SUCCESS: workflow task lifecycle validated")
        print("Initial task:", task_id)
        print("Retry task:", second_task)
        print("Retry count:", completed.retry_count)
        print("Final dispatch status:", completed.dispatch_status)
        print("Final version:", completed.version)
    finally:
        clear(runtime, workflow_id)
        runtime.dispose()


if __name__ == "__main__":
    main()
