from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from celery.exceptions import SoftTimeLimitExceeded

from app.agents.workflow import ClusterMindWorkflow, WorkflowStore
from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.repositories import (
    PostgresWorkflowRepository,
    WorkflowLifecycleRepository,
    WorkflowRepositoryConflictError,
)
from app.services.incidents import get_incident_repository
from app.tasks.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="clustermind.analyze_workflow",
    autoretry_for=(),
    max_retries=0,
    ignore_result=False,
)
def analyze_workflow(
    self,
    *,
    workflow_id: str,
    incident_id: str,
    use_llm: bool,
) -> dict[str, object]:
    task_id = str(self.request.id)
    worker_id = str(self.request.hostname or "unknown-worker")[:150]
    settings = get_settings()
    runtime = DatabaseRuntime(settings)
    lifecycle = WorkflowLifecycleRepository(runtime)
    repository = PostgresWorkflowRepository(runtime)

    try:
        try:
            lifecycle.claim(
                workflow_id,
                task_id=task_id,
                worker_id=worker_id,
            )
        except WorkflowRepositoryConflictError:
            existing = repository.get(workflow_id)
            return {
                "workflow_id": workflow_id,
                "status": existing.status,
                "duplicate_delivery": True,
            }

        lifecycle.heartbeat(
            workflow_id,
            task_id=task_id,
            worker_id=worker_id,
            current_stage="loading_evidence",
        )

        with TemporaryDirectory(prefix="clustermind-worker-") as directory:
            workflow = ClusterMindWorkflow(
                repository=get_incident_repository(settings),
                store=WorkflowStore(Path(directory)),
            )
            try:
                result = workflow.analyze(
                    incident_id,
                    use_llm=use_llm,
                    force=True,
                )
            finally:
                workflow.close()

        repository.save(result)
        lifecycle.mark_completed(
            workflow_id,
            task_id=task_id,
            worker_id=worker_id,
        )
        return {
            "workflow_id": workflow_id,
            "status": result.status,
            "duplicate_delivery": False,
        }
    except SoftTimeLimitExceeded:
        lifecycle.mark_failed(
            workflow_id,
            task_id=task_id,
            worker_id=worker_id,
            error="Analysis task exceeded the soft time limit",
        )
        raise
    except Exception as exc:
        try:
            lifecycle.mark_failed(
                workflow_id,
                task_id=task_id,
                worker_id=worker_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise
    finally:
        runtime.dispose()
