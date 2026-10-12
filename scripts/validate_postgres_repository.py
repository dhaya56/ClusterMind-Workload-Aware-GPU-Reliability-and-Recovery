from __future__ import annotations

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.models import AgentExecutionRecord, AuditEvent, OutboxEvent, WorkflowRun
from app.db.repositories import PostgresWorkflowRepository


def main() -> None:
    runtime = DatabaseRuntime(get_settings())
    repository = PostgresWorkflowRepository(runtime)
    try:
        workflows = repository.list(limit=100)
        if len(workflows) != 1:
            raise AssertionError(f"Expected one workflow, found {len(workflows)}")
        workflow = workflows[0]
        loaded = repository.get(workflow.workflow_id)
        if loaded != workflow:
            raise AssertionError("Repository get/list results differ")

        with runtime.read_session() as session:
            counts = {
                "workflow_runs": session.scalar(select(func.count()).select_from(WorkflowRun)),
                "agent_executions": session.scalar(select(func.count()).select_from(AgentExecutionRecord)),
                "audit_events": session.scalar(select(func.count()).select_from(AuditEvent)),
                "outbox_events": session.scalar(select(func.count()).select_from(OutboxEvent)),
            }
        if counts != {
            "workflow_runs": 1,
            "agent_executions": 4,
            "audit_events": 1,
            "outbox_events": 1,
        }:
            raise AssertionError(counts)
        print("SUCCESS: PostgreSQL workflow repository validated")
        print("Workflow:", workflow.workflow_id)
        print("Status:", workflow.status)
        print("Execution:", workflow.execution_status.value)
        print("Counts:", counts)
    finally:
        runtime.dispose()


if __name__ == "__main__":
    main()
