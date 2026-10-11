from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.models import AuditEvent, OutboxEvent, WorkflowRun
from app.db.session import DatabaseRuntime
from app.schemas.agents import utc_now

from app.db.repositories.workflow import (
    WorkflowRepositoryConflictError,
    WorkflowRepositoryError,
    WorkflowRepositoryNotFoundError,
)


@dataclass(frozen=True, slots=True)
class WorkflowLifecycleState:
    workflow_id: str
    incident_id: str
    status: str
    current_stage: str
    dispatch_status: str
    task_id: str | None
    worker_id: str | None
    retry_count: int
    version: int
    queued_at: datetime | None
    dispatched_at: datetime | None
    claimed_at: datetime | None
    last_heartbeat_at: datetime | None
    dispatch_error: str | None


def _snapshot(row: WorkflowRun) -> WorkflowLifecycleState:
    return WorkflowLifecycleState(
        workflow_id=row.workflow_id,
        incident_id=row.incident_id,
        status=row.status,
        current_stage=row.current_stage,
        dispatch_status=row.dispatch_status,
        task_id=row.task_id,
        worker_id=row.worker_id,
        retry_count=row.retry_count,
        version=row.version,
        queued_at=row.queued_at,
        dispatched_at=row.dispatched_at,
        claimed_at=row.claimed_at,
        last_heartbeat_at=row.last_heartbeat_at,
        dispatch_error=row.dispatch_error,
    )


class WorkflowLifecycleRepository:
    """Transactional lifecycle operations used by API dispatchers and workers."""

    def __init__(self, runtime: DatabaseRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _locked(session, workflow_id: str) -> WorkflowRun:
        row = session.scalar(
            select(WorkflowRun)
            .where(WorkflowRun.workflow_id == workflow_id)
            .with_for_update()
        )
        if row is None:
            raise WorkflowRepositoryNotFoundError(
                f"Workflow does not exist: {workflow_id}"
            )
        return row

    @staticmethod
    def _append_event(
        session,
        row: WorkflowRun,
        *,
        event_type: str,
        previous_status: str | None,
        payload: dict[str, object],
        actor_type: str,
        actor_id: str | None,
    ) -> None:
        sequence = session.scalar(
            select(AuditEvent.sequence_number)
            .where(AuditEvent.workflow_run_id == row.id)
            .order_by(AuditEvent.sequence_number.desc())
            .limit(1)
        )
        session.add(
            AuditEvent(
                workflow_run_id=row.id,
                sequence_number=int(sequence or 0) + 1,
                event_type=event_type,
                previous_status=previous_status,
                new_status=row.status,
                actor_type=actor_type,
                actor_id=actor_id,
                trace_id=row.trace_id,
                event_payload=payload,
                occurred_at=row.updated_at,
            )
        )
        session.add(
            OutboxEvent(
                aggregate_type="workflow",
                aggregate_id=row.workflow_id,
                event_type=event_type,
                payload={
                    "workflow_id": row.workflow_id,
                    "incident_id": row.incident_id,
                    "status": row.status,
                    "dispatch_status": row.dispatch_status,
                    "version": row.version,
                    **payload,
                },
                available_at=row.updated_at,
            )
        )

    def create_queued(
        self,
        *,
        workflow_id: str,
        incident_id: str,
        use_llm: bool,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        try:
            with self.runtime.transaction() as session:
                existing = session.scalar(
                    select(WorkflowRun)
                    .where(WorkflowRun.workflow_id == workflow_id)
                    .with_for_update()
                )
                if existing is not None:
                    if (
                        existing.incident_id == incident_id
                        and existing.status in {"queued", "running"}
                    ):
                        return _snapshot(existing)
                    raise WorkflowRepositoryConflictError(
                        "Workflow already exists in a non-queueable state"
                    )

                row = WorkflowRun(
                    workflow_id=workflow_id,
                    incident_id=incident_id,
                    status="queued",
                    current_stage="created",
                    execution_status="not_started",
                    requested_use_llm=use_llm,
                    started_at=now,
                    updated_at=now,
                    dispatch_status="pending",
                    queued_at=now,
                    retry_count=0,
                    errors=[],
                    metadata_payload={
                        "orchestrator": "celery",
                        "use_llm": use_llm,
                        "workflow_schema_version": "2.0",
                    },
                )
                session.add(row)
                session.flush()
                self._append_event(
                    session,
                    row,
                    event_type="workflow.queued",
                    previous_status=None,
                    payload={"use_llm": use_llm},
                    actor_type="system",
                    actor_id="analysis-dispatcher",
                )
                return _snapshot(row)
        except WorkflowRepositoryConflictError:
            raise
        except IntegrityError as exc:
            raise WorkflowRepositoryConflictError(
                f"Unable to queue workflow for incident {incident_id}"
            ) from exc
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to create queued workflow") from exc

    def mark_dispatched(
        self,
        workflow_id: str,
        *,
        task_id: str,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        try:
            with self.runtime.transaction() as session:
                row = self._locked(session, workflow_id)
                if row.dispatch_status == "dispatched" and row.task_id == task_id:
                    return _snapshot(row)
                if row.dispatch_status != "pending":
                    raise WorkflowRepositoryConflictError(
                        f"Workflow is not pending dispatch: {row.dispatch_status}"
                    )
                row.task_id = task_id
                row.dispatch_status = "dispatched"
                row.dispatched_at = now
                row.dispatch_error = None
                row.updated_at = now
                row.version += 1
                self._append_event(
                    session,
                    row,
                    event_type="workflow.dispatched",
                    previous_status=row.status,
                    payload={"task_id": task_id},
                    actor_type="system",
                    actor_id="analysis-dispatcher",
                )
                return _snapshot(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except IntegrityError as exc:
            raise WorkflowRepositoryConflictError(
                "Celery task_id is already assigned"
            ) from exc
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to mark workflow dispatched") from exc

    def mark_dispatch_failed(
        self,
        workflow_id: str,
        *,
        error: str,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        summary = error.strip()[:1000] or "Task dispatch failed"
        try:
            with self.runtime.transaction() as session:
                row = self._locked(session, workflow_id)
                if row.dispatch_status == "failed" and row.dispatch_error == summary:
                    return _snapshot(row)
                if row.dispatch_status not in {"pending", "dispatched"}:
                    raise WorkflowRepositoryConflictError(
                        f"Dispatch failure is invalid from {row.dispatch_status}"
                    )
                previous = row.status
                row.status = "failed"
                row.current_stage = "failed"
                row.dispatch_status = "failed"
                row.dispatch_error = summary
                row.failure_code = "dispatch_failed"
                row.failure_summary = summary
                row.errors = [summary]
                row.completed_at = now
                row.updated_at = now
                row.version += 1
                self._append_event(
                    session,
                    row,
                    event_type="workflow.dispatch_failed",
                    previous_status=previous,
                    payload={"error": summary},
                    actor_type="system",
                    actor_id="analysis-dispatcher",
                )
                return _snapshot(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to persist dispatch failure") from exc

    def claim(
        self,
        workflow_id: str,
        *,
        task_id: str,
        worker_id: str,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        try:
            with self.runtime.transaction() as session:
                row = self._locked(session, workflow_id)
                if (
                    row.dispatch_status == "claimed"
                    and row.task_id == task_id
                    and row.worker_id == worker_id
                ):
                    return _snapshot(row)
                if row.dispatch_status != "dispatched" or row.task_id != task_id:
                    raise WorkflowRepositoryConflictError(
                        "Workflow task cannot be claimed"
                    )
                previous = row.status
                row.status = "running"
                row.current_stage = "loading_evidence"
                row.dispatch_status = "claimed"
                row.worker_id = worker_id.strip()[:150]
                row.claimed_at = now
                row.last_heartbeat_at = now
                row.updated_at = now
                row.version += 1
                self._append_event(
                    session,
                    row,
                    event_type="workflow.claimed",
                    previous_status=previous,
                    payload={"task_id": task_id, "worker_id": row.worker_id},
                    actor_type="worker",
                    actor_id=row.worker_id,
                )
                return _snapshot(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to claim workflow task") from exc

    def heartbeat(
        self,
        workflow_id: str,
        *,
        task_id: str,
        worker_id: str,
        current_stage: str | None = None,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        try:
            with self.runtime.transaction() as session:
                row = self._locked(session, workflow_id)
                if (
                    row.dispatch_status != "claimed"
                    or row.task_id != task_id
                    or row.worker_id != worker_id
                ):
                    raise WorkflowRepositoryConflictError(
                        "Heartbeat ownership does not match the active claim"
                    )
                if current_stage is not None:
                    row.current_stage = current_stage
                row.last_heartbeat_at = now
                row.updated_at = now
                return _snapshot(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to update workflow heartbeat") from exc

    def mark_completed(
        self,
        workflow_id: str,
        *,
        task_id: str,
        worker_id: str,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        try:
            with self.runtime.transaction() as session:
                row = self._locked(session, workflow_id)
                if row.dispatch_status == "completed":
                    return _snapshot(row)
                if (
                    row.dispatch_status != "claimed"
                    or row.task_id != task_id
                    or row.worker_id != worker_id
                ):
                    raise WorkflowRepositoryConflictError(
                        "Completion ownership does not match the active claim"
                    )
                row.dispatch_status = "completed"
                row.last_heartbeat_at = now
                row.updated_at = now
                row.version += 1
                self._append_event(
                    session,
                    row,
                    event_type="workflow.task_completed",
                    previous_status=row.status,
                    payload={"task_id": task_id},
                    actor_type="worker",
                    actor_id=worker_id,
                )
                return _snapshot(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to complete workflow task") from exc

    def mark_failed(
        self,
        workflow_id: str,
        *,
        task_id: str,
        worker_id: str,
        error: str,
    ) -> WorkflowLifecycleState:
        now = utc_now()
        summary = error.strip()[:1000] or "Workflow task failed"
        try:
            with self.runtime.transaction() as session:
                row = self._locked(session, workflow_id)
                if row.dispatch_status == "failed" and row.dispatch_error == summary:
                    return _snapshot(row)
                if row.task_id != task_id:
                    raise WorkflowRepositoryConflictError(
                        "Failure task_id does not match the workflow"
                    )
                previous = row.status
                row.status = "failed"
                row.current_stage = "failed"
                row.dispatch_status = "failed"
                row.dispatch_error = summary
                row.failure_code = "analysis_task_failed"
                row.failure_summary = summary
                row.errors = [summary]
                row.completed_at = now
                row.last_heartbeat_at = now
                row.updated_at = now
                row.version += 1
                self._append_event(
                    session,
                    row,
                    event_type="workflow.task_failed",
                    previous_status=previous,
                    payload={"task_id": task_id, "error": summary},
                    actor_type="worker",
                    actor_id=worker_id,
                )
                return _snapshot(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to fail workflow task") from exc

    def recover_stale_claims(
        self,
        *,
        stale_before: datetime,
        limit: int = 100,
    ) -> tuple[WorkflowLifecycleState, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        now = utc_now()
        recovered: list[WorkflowLifecycleState] = []
        try:
            with self.runtime.transaction() as session:
                rows = session.scalars(
                    select(WorkflowRun)
                    .where(
                        WorkflowRun.dispatch_status == "claimed",
                        WorkflowRun.last_heartbeat_at < stale_before,
                    )
                    .order_by(WorkflowRun.last_heartbeat_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                ).all()
                for row in rows:
                    previous = row.status
                    old_task_id = row.task_id
                    row.status = "queued"
                    row.current_stage = "created"
                    row.dispatch_status = "pending"
                    row.task_id = None
                    row.dispatched_at = None
                    row.claimed_at = None
                    row.last_heartbeat_at = None
                    row.worker_id = None
                    row.retry_count += 1
                    row.dispatch_error = None
                    row.queued_at = now
                    row.updated_at = now
                    row.version += 1
                    self._append_event(
                        session,
                        row,
                        event_type="workflow.stale_claim_recovered",
                        previous_status=previous,
                        payload={"previous_task_id": old_task_id},
                        actor_type="system",
                        actor_id="stale-claim-recovery",
                    )
                    recovered.append(_snapshot(row))
                return tuple(recovered)
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to recover stale claims") from exc
