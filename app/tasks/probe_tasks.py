from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.tasks.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="clustermind.probe",
    autoretry_for=(),
    max_retries=0,
    ignore_result=False,
)
def probe(self) -> dict[str, Any]:
    """Small JSON-only task used to validate broker, worker, and backend."""

    return {
        "status": "ok",
        "task_id": self.request.id,
        "worker": self.request.hostname,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
