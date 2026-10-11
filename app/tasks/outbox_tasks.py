from __future__ import annotations

from app.core.config import get_settings
from app.db import DatabaseRuntime
from app.services.outbox import NtfyPublisher, OutboxPublisher, OutboxRepository
from app.tasks.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="clustermind.publish_outbox",
    autoretry_for=(),
    max_retries=0,
    ignore_result=False,
)
def publish_outbox(self, *, limit: int = 20) -> dict[str, int]:
    settings = get_settings()
    runtime = DatabaseRuntime(settings)
    worker_id = str(self.request.hostname or "unknown-worker")[:150]
    try:
        result = OutboxPublisher(
            OutboxRepository(runtime),
            NtfyPublisher(settings),
        ).publish_batch(worker_id=worker_id, limit=limit)
        return {
            "claimed": result.claimed,
            "published": result.published,
            "failed": result.failed,
        }
    finally:
        runtime.dispose()
