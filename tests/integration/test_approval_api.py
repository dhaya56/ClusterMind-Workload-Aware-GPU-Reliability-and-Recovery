from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.agents.workflow import ClusterMindWorkflow, WorkflowStore
from app.core.config import Settings
from app.db import DatabaseRuntime
from app.db.models import (
    AgentExecutionRecord,
    ApprovalDecisionRecord,
    AuditEvent,
    OutboxEvent,
    WorkflowRun,
)
from app.db.repositories import PostgresWorkflowRepository
from app.main import create_app
from app.services.incidents import get_incident_repository

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)

INCIDENT_ID = "INC-2026-0034"


def _clear(runtime: DatabaseRuntime, workflow_id: str) -> None:
    with runtime.transaction() as session:
        row = session.scalar(
            select(WorkflowRun).where(WorkflowRun.workflow_id == workflow_id)
        )
        if row is None:
            return
        session.execute(
            delete(OutboxEvent).where(OutboxEvent.aggregate_id == workflow_id)
        )
        session.execute(
            delete(ApprovalDecisionRecord).where(
                ApprovalDecisionRecord.workflow_run_id == row.id
            )
        )
        session.execute(
            delete(AuditEvent).where(AuditEvent.workflow_run_id == row.id)
        )
        session.execute(
            delete(AgentExecutionRecord).where(
                AgentExecutionRecord.workflow_run_id == row.id
            )
        )
        session.delete(row)


def _seed_workflow(settings: Settings, runtime: DatabaseRuntime) -> str:
    with TemporaryDirectory(prefix="clustermind-approval-api-") as directory:
        workflow = ClusterMindWorkflow(
            repository=get_incident_repository(settings),
            store=WorkflowStore(Path(directory)),
        )
        try:
            result = workflow.analyze(
                INCIDENT_ID,
                use_llm=False,
                force=True,
            )
        finally:
            workflow.close()

    assert result.status == "awaiting_approval"
    PostgresWorkflowRepository(runtime).save(result)
    return result.workflow_id


def test_concurrent_api_decisions_are_consistent() -> None:
    settings = Settings(
        workflow_repository_backend="postgres",
        otel_enabled=False,
    )
    runtime = DatabaseRuntime(settings)
    workflow_id = ClusterMindWorkflow.workflow_id_for(INCIDENT_ID)

    try:
        _clear(runtime, workflow_id)
        workflow_id = _seed_workflow(settings, runtime)
        app = create_app(settings)

        with TestClient(app) as client:
            missing_reason = client.post(
                f"/api/v1/workflows/{workflow_id}/decision",
                json={
                    "decision": "rejected",
                    "actor": "pytest-operator",
                },
            )
            assert missing_reason.status_code == 422

            def submit(decision: str):
                payload = {
                    "decision": decision,
                    "actor": f"pytest-{decision}",
                }
                if decision == "rejected":
                    payload["reason"] = "Concurrent rejection validation"
                return client.post(
                    f"/api/v1/workflows/{workflow_id}/decision",
                    json=payload,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = [
                    executor.submit(submit, "approved").result(),
                    executor.submit(submit, "rejected").result(),
                ]

            status_codes = sorted(response.status_code for response in responses)
            assert status_codes == [200, 409], [
                (response.status_code, response.text) for response in responses
            ]

            successful = next(response for response in responses if response.status_code == 200)
            final_status = successful.json()["status"]
            duplicate = client.post(
                f"/api/v1/workflows/{workflow_id}/decision",
                json={
                    "decision": final_status,
                    "actor": "pytest-duplicate",
                    "reason": (
                        "Duplicate rejection validation"
                        if final_status == "rejected"
                        else None
                    ),
                },
            )
            assert duplicate.status_code == 200
            assert duplicate.json()["status"] == final_status

        with runtime.read_session() as session:
            row = session.scalar(
                select(WorkflowRun).where(
                    WorkflowRun.workflow_id == workflow_id
                )
            )
            assert row is not None
            approval_count = session.scalar(
                select(func.count(ApprovalDecisionRecord.id)).where(
                    ApprovalDecisionRecord.workflow_run_id == row.id
                )
            )
            audit_count = session.scalar(
                select(func.count(AuditEvent.id)).where(
                    AuditEvent.workflow_run_id == row.id
                )
            )
            outbox_count = session.scalar(
                select(func.count(OutboxEvent.id)).where(
                    OutboxEvent.aggregate_id == workflow_id
                )
            )

        assert approval_count == 1
        assert audit_count == 2
        assert outbox_count == 2
    finally:
        _clear(runtime, workflow_id)
        runtime.dispose()
