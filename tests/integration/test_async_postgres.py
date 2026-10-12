from __future__ import annotations

import os
from datetime import timedelta

import pytest
from sqlalchemy import delete

from app.agents.workflow import ClusterMindWorkflow
from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.models import AuditEvent, OutboxEvent, WorkflowRun
from app.db.repositories import WorkflowLifecycleRepository
from app.schemas.agents import utc_now

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)

INCIDENT_ID = "INC-PYTEST-ASYNC"


def _clear(runtime: DatabaseRuntime, workflow_id: str) -> None:
    with runtime.transaction() as session:
        row = session.query(WorkflowRun).filter_by(workflow_id=workflow_id).one_or_none()
        if row is None:
            return
        session.execute(delete(OutboxEvent).where(OutboxEvent.aggregate_id == workflow_id))
        session.execute(delete(AuditEvent).where(AuditEvent.workflow_run_id == row.id))
        session.delete(row)


def test_stale_claim_becomes_pending_with_retry() -> None:
    runtime = DatabaseRuntime(get_settings())
    lifecycle = WorkflowLifecycleRepository(runtime)
    workflow_id = ClusterMindWorkflow.workflow_id_for(INCIDENT_ID)

    try:
        _clear(runtime, workflow_id)
        lifecycle.create_queued(
            workflow_id=workflow_id,
            incident_id=INCIDENT_ID,
            use_llm=False,
        )
        lifecycle.mark_dispatched(workflow_id, task_id="pytest-old-task")
        lifecycle.claim(
            workflow_id,
            task_id="pytest-old-task",
            worker_id="pytest-old-worker",
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
        state = lifecycle.get_status(workflow_id)

        assert len(recovered) == 1
        assert state.dispatch_status == "pending"
        assert state.task_id is None
        assert state.worker_id is None
        assert state.retry_count == 1
    finally:
        _clear(runtime, workflow_id)
        runtime.dispose()
