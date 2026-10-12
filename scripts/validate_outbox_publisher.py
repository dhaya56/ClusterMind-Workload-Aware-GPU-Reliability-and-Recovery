from __future__ import annotations

from sqlalchemy import delete, select

from app.core.config import Settings
from app.db import DatabaseRuntime
from app.db.models import OutboxEvent
from app.services.outbox import NtfyPublisher, OutboxPublisher, OutboxRepository

AGGREGATE_ID = "WF-OUTBOX-VALIDATION"


def main() -> None:
    settings = Settings(
        ntfy_enabled=True,
        ntfy_base_url="http://localhost:18080",
        ntfy_topic="clustermind-validation",
    )
    runtime = DatabaseRuntime(settings)
    try:
        with runtime.transaction() as session:
            session.execute(
                delete(OutboxEvent).where(
                    OutboxEvent.aggregate_id == AGGREGATE_ID
                )
            )
            session.add(
                OutboxEvent(
                    aggregate_type="workflow",
                    aggregate_id=AGGREGATE_ID,
                    event_type="workflow.validation",
                    payload={"status": "validation"},
                )
            )

        result = OutboxPublisher(
            OutboxRepository(runtime),
            NtfyPublisher(settings),
        ).publish_batch(
            worker_id="outbox-validation",
            limit=1,
            aggregate_id=AGGREGATE_ID,
        )

        with runtime.read_session() as session:
            row = session.scalar(
                select(OutboxEvent).where(
                    OutboxEvent.aggregate_id == AGGREGATE_ID
                )
            )
            assert row is not None
            assert row.published_at is not None
            assert row.attempt_count == 0
            assert row.last_error is None

        assert result.claimed == 1
        assert result.published == 1
        assert result.failed == 0
        print("SUCCESS: transactional outbox publication validated")
        print("Claimed:", result.claimed)
        print("Published:", result.published)
        print("Failed:", result.failed)
    finally:
        with runtime.transaction() as session:
            session.execute(
                delete(OutboxEvent).where(
                    OutboxEvent.aggregate_id == AGGREGATE_ID
                )
            )
        runtime.dispose()
        print("SUCCESS: validation outbox event cleaned up")


if __name__ == "__main__":
    main()
