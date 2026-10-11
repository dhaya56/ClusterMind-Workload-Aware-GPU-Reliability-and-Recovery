from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.db.models import OutboxEvent
from app.db.session import DatabaseRuntime
from app.schemas.agents import utc_now


class OutboxPublisherError(RuntimeError):
    """Raised when outbox claiming or publication cannot complete safely."""


@dataclass(frozen=True, slots=True)
class OutboxMessage:
    id: UUID
    aggregate_id: str
    event_type: str
    payload: dict[str, Any]
    attempt_count: int


@dataclass(frozen=True, slots=True)
class PublishBatchResult:
    claimed: int
    published: int
    failed: int


class OutboxRepository:
    def __init__(self, runtime: DatabaseRuntime) -> None:
        self.runtime = runtime

    def claim_batch(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        lock_timeout_seconds: int = 120,
        aggregate_id: str | None = None,
    ) -> tuple[OutboxMessage, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        now = utc_now()
        expired = now - timedelta(seconds=lock_timeout_seconds)
        try:
            with self.runtime.transaction() as session:
                statement = select(OutboxEvent).where(
                    OutboxEvent.published_at.is_(None),
                    OutboxEvent.available_at <= now,
                    or_(
                        OutboxEvent.locked_at.is_(None),
                        OutboxEvent.locked_at < expired,
                    ),
                )
                if aggregate_id is not None:
                    statement = statement.where(
                        OutboxEvent.aggregate_id == aggregate_id
                    )
                rows = session.scalars(
                    statement
                    .order_by(OutboxEvent.created_at, OutboxEvent.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                ).all()
                messages: list[OutboxMessage] = []
                for row in rows:
                    row.locked_at = now
                    row.locked_by = worker_id[:150]
                    messages.append(
                        OutboxMessage(
                            id=row.id,
                            aggregate_id=row.aggregate_id,
                            event_type=row.event_type,
                            payload=dict(row.payload),
                            attempt_count=row.attempt_count,
                        )
                    )
                return tuple(messages)
        except SQLAlchemyError as exc:
            raise OutboxPublisherError("Unable to claim outbox events") from exc

    def mark_published(self, message_id: UUID, *, worker_id: str) -> None:
        now = utc_now()
        try:
            with self.runtime.transaction() as session:
                row = session.get(OutboxEvent, message_id, with_for_update=True)
                if row is None:
                    raise OutboxPublisherError("Outbox event does not exist")
                if row.published_at is not None:
                    return
                if row.locked_by != worker_id[:150]:
                    raise OutboxPublisherError("Outbox lock ownership mismatch")
                row.published_at = now
                row.locked_at = None
                row.locked_by = None
                row.last_error = None
        except SQLAlchemyError as exc:
            raise OutboxPublisherError("Unable to mark outbox event published") from exc

    def mark_failed(
        self,
        message_id: UUID,
        *,
        worker_id: str,
        error: str,
    ) -> None:
        now = utc_now()
        summary = error.strip()[:1000] or "Notification publication failed"
        try:
            with self.runtime.transaction() as session:
                row = session.get(OutboxEvent, message_id, with_for_update=True)
                if row is None:
                    raise OutboxPublisherError("Outbox event does not exist")
                if row.published_at is not None:
                    return
                if row.locked_by != worker_id[:150]:
                    raise OutboxPublisherError("Outbox lock ownership mismatch")
                row.attempt_count += 1
                delay_seconds = min(3600, 30 * (2 ** min(row.attempt_count - 1, 7)))
                row.available_at = now + timedelta(seconds=delay_seconds)
                row.last_error = summary
                row.locked_at = None
                row.locked_by = None
        except SQLAlchemyError as exc:
            raise OutboxPublisherError("Unable to record outbox failure") from exc


class NtfyPublisher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def publish(self, message: OutboxMessage) -> None:
        if not self.settings.ntfy_enabled:
            raise OutboxPublisherError("ntfy notifications are disabled")
        url = f"{self.settings.ntfy_base_url}/{self.settings.ntfy_topic}"
        title = f"ClusterMind: {message.event_type}"
        body = (
            f"Workflow {message.aggregate_id}\n"
            f"Status: {message.payload.get('status', 'unknown')}\n"
            f"Event: {message.event_type}"
        )
        try:
            response = httpx.post(
                url,
                content=body.encode("utf-8"),
                headers={
                    "Title": title,
                    "Tags": "robot,computer",
                    "Priority": "default",
                    "Content-Type": "text/plain; charset=utf-8",
                },
                timeout=10.0,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OutboxPublisherError(
                f"ntfy publish failed: {type(exc).__name__}"
            ) from exc


class OutboxPublisher:
    def __init__(self, repository: OutboxRepository, publisher: NtfyPublisher) -> None:
        self.repository = repository
        self.publisher = publisher

    def publish_batch(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        aggregate_id: str | None = None,
    ) -> PublishBatchResult:
        if not self.publisher.settings.ntfy_enabled:
            return PublishBatchResult(claimed=0, published=0, failed=0)
        messages = self.repository.claim_batch(
            worker_id=worker_id,
            limit=limit,
            aggregate_id=aggregate_id,
        )
        published = 0
        failed = 0
        for message in messages:
            try:
                self.publisher.publish(message)
                self.repository.mark_published(message.id, worker_id=worker_id)
                published += 1
            except Exception as exc:
                self.repository.mark_failed(
                    message.id,
                    worker_id=worker_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                failed += 1
        return PublishBatchResult(
            claimed=len(messages),
            published=published,
            failed=failed,
        )
