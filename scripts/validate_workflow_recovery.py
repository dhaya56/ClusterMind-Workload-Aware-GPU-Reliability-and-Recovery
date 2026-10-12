from __future__ import annotations

from datetime import timedelta

from sqlalchemy import delete

from app.agents.workflow import ClusterMindWorkflow
from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.models import AuditEvent, OutboxEvent, WorkflowRun
from app.db.repositories import WorkflowLifecycleRepository
from app.schemas.agents import utc_now

INCIDENT_ID = "INC-RECOVERY-VALIDATION"


def clear(runtime: DatabaseRuntime, workflow_id: str) -> None:
    with runtime.transaction() as session:
        row = session.query(WorkflowRun).filter_by(workflow_id=workflow_id).one_or_none()
        if row is None:
            return
        session.execute(delete(OutboxEvent).where(OutboxEvent.aggregate_id == workflow_id))
        session.execute(delete(AuditEvent).where(AuditEvent.workflow_run_id == row.id))
        session.delete(row)


def main() -> None:
    settings = get_settings()
    runtime = DatabaseRuntime(settings)
    lifecycle = WorkflowLifecycleRepository(runtime)
    workflow_id = ClusterMindWorkflow.workflow_id_for(INCIDENT_ID)
    try:
        clear(runtime, workflow_id)
        lifecycle.create_queued(
            workflow_id=workflow_id,
            incident_id=INCIDENT_ID,
            use_llm=False,
        )
        lifecycle.mark_dispatched(workflow_id, task_id="recovery-old-task")
        lifecycle.claim(
            workflow_id,
            task_id="recovery-old-task",
            worker_id="recovery-old-worker",
        )

        with runtime.transaction() as session:
            row = session.query(WorkflowRun).filter_by(workflow_id=workflow_id).one()
            now = utc_now()
            row.queued_at = now - timedelta(minutes=30)
            row.dispatched_at = now - timedelta(minutes=25)
            row.claimed_at = now - timedelta(minutes=20)
            row.last_heartbeat_at = now - timedelta(minutes=15)

        recovered = lifecycle.recover_stale_claims(
            stale_before=utc_now() - timedelta(minutes=5),
            limit=10,
        )
        pending = lifecycle.list_pending(limit=10)
        selected = [item for item in pending if item.workflow_id == workflow_id]

        assert len(recovered) == 1
        assert len(selected) == 1
        assert selected[0].dispatch_status == "pending"
        assert selected[0].task_id is None
        assert selected[0].retry_count == 1

        print("SUCCESS: stale workflow recovery validated")
        print("Recovered:", len(recovered))
        print("Pending for redispatch:", len(selected))
        print("Retry count:", selected[0].retry_count)
    finally:
        clear(runtime, workflow_id)
        runtime.dispose()
        print("SUCCESS: recovery validation workflow cleaned up")


if __name__ == "__main__":
    main()
