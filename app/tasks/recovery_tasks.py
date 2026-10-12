from __future__ import annotations

from datetime import timedelta

from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.db.repositories import WorkflowLifecycleRepository
from app.schemas.agents import utc_now
from app.services.analysis_dispatcher import AnalysisDispatcher, AnalysisDispatchError
from app.tasks.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="clustermind.recover_workflows",
    autoretry_for=(),
    max_retries=0,
    ignore_result=False,
)
def recover_workflows(self, *, limit: int = 20) -> dict[str, int]:
    """Recover stale claims and redispatch eligible pending workflows."""

    settings = get_settings()
    runtime = DatabaseRuntime(settings)
    lifecycle = WorkflowLifecycleRepository(runtime)
    dispatcher = AnalysisDispatcher(lifecycle)
    recovered_count = 0
    dispatched_count = 0
    failed_count = 0

    try:
        stale_before = utc_now() - timedelta(
            seconds=settings.celery_workflow_stale_seconds
        )
        recovered = lifecycle.recover_stale_claims(
            stale_before=stale_before,
            limit=limit,
        )
        recovered_count = len(recovered)

        pending = lifecycle.list_pending(limit=limit)
        for state in pending:
            try:
                dispatcher.dispatch_pending(state.workflow_id)
                dispatched_count += 1
            except AnalysisDispatchError:
                failed_count += 1

        return {
            "recovered": recovered_count,
            "dispatched": dispatched_count,
            "failed": failed_count,
        }
    finally:
        runtime.dispose()
