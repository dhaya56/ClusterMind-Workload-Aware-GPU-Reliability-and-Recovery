from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    AgentExecutionRecord,
    ApprovalDecisionRecord,
    AuditEvent,
    OutboxEvent,
    WorkflowRun,
)
from app.db.session import DatabaseRuntime
from app.schemas.agents import (
    ApprovalDecision,
    ApprovalRecord,
    RecoveryExecutionStatus,
    WorkflowResult,
    utc_now,
)


class WorkflowRepositoryError(RuntimeError):
    """Base error for durable workflow persistence."""


class WorkflowRepositoryNotFoundError(WorkflowRepositoryError):
    """Raised when a workflow does not exist."""


class WorkflowRepositoryConflictError(WorkflowRepositoryError):
    """Raised when a state transition conflicts with persisted state."""


class WorkflowRepository(Protocol):
    def save(self, result: WorkflowResult) -> None: ...
    def get(self, workflow_id: str) -> WorkflowResult: ...
    def list(self, *, limit: int = 100) -> tuple[WorkflowResult, ...]: ...


_AGENT_RESULT_FIELDS = (
    ("telemetry", "telemetry_result"),
    ("workload_impact", "workload_impact_result"),
    ("root_cause", "root_cause_result"),
    ("recovery_plan", "recovery_plan_result"),
)


def _json(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _stage_for(result: WorkflowResult) -> str:
    if result.status == "failed":
        return "failed"
    if result.status in {"completed", "suppressed", "rejected"}:
        return "completed"
    if result.status in {"awaiting_approval", "approved", "more_evidence_requested"}:
        return "human_approval"
    if result.recovery_plan is not None:
        return "recovery"
    if result.root_cause is not None:
        return "root_cause"
    if result.workload_impact is not None:
        return "workload_impact"
    if result.telemetry is not None:
        return "gpu_telemetry"
    return "created"


def _is_terminal(status: str) -> bool:
    return status in {"suppressed", "rejected", "completed", "failed", "cancelled"}


def _result_to_row(result: WorkflowResult, row: WorkflowRun | None = None) -> WorkflowRun:
    target = row or WorkflowRun(
        workflow_id=result.workflow_id,
        incident_id=result.incident_id,
        status=result.status,
        current_stage=_stage_for(result),
        execution_status=result.execution_status.value,
        requested_use_llm=bool(result.metadata.get("use_llm", True)),
        started_at=result.started_at,
        updated_at=result.updated_at,
    )
    target.incident_id = result.incident_id
    target.status = result.status
    target.current_stage = _stage_for(result)
    target.execution_status = result.execution_status.value
    target.requested_use_llm = bool(result.metadata.get("use_llm", True))
    target.started_at = result.started_at
    target.updated_at = result.updated_at
    target.completed_at = result.updated_at if _is_terminal(result.status) else None
    target.trace_id = result.trace_id
    target.failure_code = "workflow_failed" if result.status == "failed" else None
    target.failure_summary = result.errors[0] if result.status == "failed" and result.errors else None
    target.telemetry_result = _json(result.telemetry)
    target.workload_impact_result = _json(result.workload_impact)
    target.root_cause_result = _json(result.root_cause)
    target.recovery_plan_result = _json(result.recovery_plan)
    target.errors = list(result.errors)
    target.metadata_payload = dict(result.metadata)
    return target


def _agent_records(result: WorkflowResult, workflow: WorkflowRun) -> list[AgentExecutionRecord]:
    records: list[AgentExecutionRecord] = []
    for result_field, _ in _AGENT_RESULT_FIELDS:
        agent_result = getattr(result, result_field)
        if agent_result is None:
            continue
        execution = agent_result.execution
        records.append(
            AgentExecutionRecord(
                workflow=workflow,
                agent_name=execution.agent.value,
                attempt_number=1,
                status=execution.status.value,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_ms=execution.duration_ms,
                trace_id=execution.trace_id,
                used_llm=bool(getattr(agent_result, "used_llm", False)),
                fallback_used=bool(getattr(agent_result, "fallback_used", False)),
                error_code=execution.error_code,
                error_message=execution.error_message,
                result_payload=_json(agent_result),
                updated_at=result.updated_at,
            )
        )
    return records


def _row_to_result(row: WorkflowRun) -> WorkflowResult:
    decision = row.approval_decisions[0] if row.approval_decisions else None
    approval = None
    if decision is not None:
        approval = ApprovalRecord(
            incident_id=row.incident_id,
            recommendation_id=decision.recommendation_id,
            decision=ApprovalDecision(decision.decision),
            actor=decision.actor,
            reason=decision.reason,
            decided_at=decision.decided_at,
        )
    payload = {
        "workflow_id": row.workflow_id,
        "incident_id": row.incident_id,
        "started_at": row.started_at,
        "updated_at": row.updated_at,
        "status": row.status,
        "telemetry": row.telemetry_result,
        "workload_impact": row.workload_impact_result,
        "root_cause": row.root_cause_result,
        "recovery_plan": row.recovery_plan_result,
        "approval": _json(approval),
        "execution_status": row.execution_status,
        "trace_id": row.trace_id,
        "errors": row.errors,
        "metadata": row.metadata_payload,
    }
    return WorkflowResult.model_validate(payload)


class PostgresWorkflowRepository:
    """Transactional PostgreSQL repository for ClusterMind workflow aggregates."""

    def __init__(self, runtime: DatabaseRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _validated_id(workflow_id: str) -> str:
        normalized = workflow_id.strip()
        if not normalized:
            raise WorkflowRepositoryError("workflow_id cannot be blank")
        if len(normalized) > 150:
            raise WorkflowRepositoryError("workflow_id is too long")
        return normalized

    @staticmethod
    def _load_statement(workflow_id: str):
        return (
            select(WorkflowRun)
            .where(WorkflowRun.workflow_id == workflow_id)
            .options(selectinload(WorkflowRun.approval_decisions))
        )

    @staticmethod
    def _next_sequence(session: Session, workflow_run_id: Any) -> int:
        current = session.scalar(
            select(func.max(AuditEvent.sequence_number)).where(
                AuditEvent.workflow_run_id == workflow_run_id
            )
        )
        return int(current or 0) + 1

    @staticmethod
    def _append_events(
        session: Session,
        row: WorkflowRun,
        *,
        event_type: str,
        previous_status: str | None,
        actor_type: str,
        actor_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        session.add(
            AuditEvent(
                workflow=row,
                sequence_number=PostgresWorkflowRepository._next_sequence(session, row.id),
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
                    "version": row.version,
                },
                available_at=row.updated_at,
            )
        )

    def save(self, result: WorkflowResult) -> None:
        validated = WorkflowResult.model_validate(result)
        try:
            with self.runtime.transaction() as session:
                existing = session.scalar(
                    self._load_statement(validated.workflow_id).with_for_update()
                )
                previous_status = existing.status if existing is not None else None
                if existing is None:
                    row = _result_to_row(validated)
                    session.add(row)
                    session.flush()
                    event_type = "workflow.created"
                else:
                    if existing.status not in {"queued", "running"}:
                        persisted = _row_to_result(existing)
                        if (
                            persisted.model_dump(mode="json")
                            == validated.model_dump(mode="json")
                        ):
                            return
                    if (
                        existing.approval_decisions
                        and validated.approval is None
                    ):
                        raise WorkflowRepositoryConflictError(
                            "A finalized workflow cannot lose its approval decision"
                        )
                    row = _result_to_row(validated, existing)
                    row.version += 1
                    event_type = (
                        "workflow.status_changed"
                        if previous_status != row.status
                        else "workflow.updated"
                    )

                session.execute(
                    delete(AgentExecutionRecord).where(
                        AgentExecutionRecord.workflow_run_id == row.id
                    )
                )
                session.add_all(_agent_records(validated, row))

                if validated.approval is not None:
                    self._persist_approval(session, row, validated.approval)

                self._append_events(
                    session,
                    row,
                    event_type=event_type,
                    previous_status=previous_status,
                    actor_type="system",
                    actor_id="workflow-repository",
                    payload={"source": "aggregate_save"},
                )
        except IntegrityError as exc:
            raise WorkflowRepositoryConflictError(
                f"Workflow persistence conflict: {validated.workflow_id}"
            ) from exc
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError(
                f"Unable to persist workflow: {validated.workflow_id}"
            ) from exc

    @staticmethod
    def _persist_approval(
        session: Session,
        row: WorkflowRun,
        approval: ApprovalRecord,
    ) -> ApprovalDecisionRecord:
        existing = session.scalar(
            select(ApprovalDecisionRecord)
            .where(ApprovalDecisionRecord.workflow_run_id == row.id)
            .with_for_update()
        )
        if existing is not None:
            if existing.decision == approval.decision.value:
                return existing
            raise WorkflowRepositoryConflictError(
                "A different final approval decision has already been recorded"
            )
        if approval.decision == ApprovalDecision.PENDING:
            raise WorkflowRepositoryConflictError(
                "pending is not a submitted approval decision"
            )
        record = ApprovalDecisionRecord(
            workflow=row,
            recommendation_id=approval.recommendation_id,
            decision=approval.decision.value,
            actor=approval.actor,
            reason=approval.reason,
            decided_at=approval.decided_at,
            workflow_version=row.version,
        )
        session.add(record)
        return record

    def get(self, workflow_id: str) -> WorkflowResult:
        normalized = self._validated_id(workflow_id)
        try:
            with self.runtime.read_session() as session:
                row = session.scalar(self._load_statement(normalized))
                if row is None:
                    raise WorkflowRepositoryNotFoundError(
                        f"Workflow does not exist: {normalized}"
                    )
                return _row_to_result(row)
        except WorkflowRepositoryNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError(
                f"Unable to read workflow: {normalized}"
            ) from exc

    def list(self, *, limit: int = 100) -> tuple[WorkflowResult, ...]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise WorkflowRepositoryError("limit must be between 1 and 1000")
        try:
            with self.runtime.read_session() as session:
                rows: Sequence[WorkflowRun] = session.scalars(
                    select(WorkflowRun)
                    .options(selectinload(WorkflowRun.approval_decisions))
                    .order_by(WorkflowRun.updated_at.desc())
                    .limit(limit)
                ).all()
                return tuple(_row_to_result(row) for row in rows)
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError("Unable to list workflows") from exc

    def decide(
        self,
        workflow_id: str,
        *,
        decision: ApprovalDecision | str,
        actor: str,
        reason: str | None = None,
        expected_version: int | None = None,
    ) -> WorkflowResult:
        normalized = self._validated_id(workflow_id)
        requested = ApprovalDecision(decision)
        if requested == ApprovalDecision.PENDING:
            raise WorkflowRepositoryConflictError(
                "pending is not a valid submitted decision"
            )
        if requested != ApprovalDecision.APPROVED and not (reason or "").strip():
            raise WorkflowRepositoryConflictError(
                "non-approved decision requires a reason"
            )
        try:
            with self.runtime.transaction() as session:
                row = session.scalar(
                    self._load_statement(normalized).with_for_update()
                )
                if row is None:
                    raise WorkflowRepositoryNotFoundError(
                        f"Workflow does not exist: {normalized}"
                    )
                if expected_version is not None and row.version != expected_version:
                    raise WorkflowRepositoryConflictError(
                        "workflow version does not match expected_version"
                    )
                if row.recovery_plan_result is None:
                    raise WorkflowRepositoryConflictError(
                        "Workflow has no recovery recommendation"
                    )
                existing = row.approval_decisions[0] if row.approval_decisions else None
                if existing is not None:
                    if existing.decision == requested.value:
                        return _row_to_result(row)
                    raise WorkflowRepositoryConflictError(
                        "A different final approval decision has already been recorded"
                    )

                recommendation_id = str(
                    row.recovery_plan_result.get("recommendation_id", "")
                ).strip()
                if not recommendation_id:
                    raise WorkflowRepositoryConflictError(
                        "Recovery recommendation has no recommendation_id"
                    )
                previous_status = row.status
                status_map = {
                    ApprovalDecision.APPROVED: "approved",
                    ApprovalDecision.REJECTED: "rejected",
                    ApprovalDecision.MORE_EVIDENCE_REQUESTED: "more_evidence_requested",
                }
                execution_map = {
                    ApprovalDecision.APPROVED: RecoveryExecutionStatus.QUEUED.value,
                    ApprovalDecision.REJECTED: RecoveryExecutionStatus.CANCELLED.value,
                    ApprovalDecision.MORE_EVIDENCE_REQUESTED: RecoveryExecutionStatus.NOT_STARTED.value,
                }
                decided_at = utc_now()
                row.status = status_map[requested]
                row.execution_status = execution_map[requested]
                row.current_stage = "human_approval"
                row.updated_at = decided_at
                row.version += 1
                approval = ApprovalRecord(
                    incident_id=row.incident_id,
                    recommendation_id=recommendation_id,
                    decision=requested,
                    actor=actor,
                    reason=reason,
                    decided_at=decided_at,
                )
                self._persist_approval(session, row, approval)
                self._append_events(
                    session,
                    row,
                    event_type="workflow.decision_recorded",
                    previous_status=previous_status,
                    actor_type="human",
                    actor_id=actor,
                    payload={
                        "decision": requested.value,
                        "recommendation_id": recommendation_id,
                    },
                )
                session.flush()
                return _row_to_result(row)
        except (WorkflowRepositoryNotFoundError, WorkflowRepositoryConflictError):
            raise
        except IntegrityError as exc:
            raise WorkflowRepositoryConflictError(
                f"Workflow decision conflict: {normalized}"
            ) from exc
        except SQLAlchemyError as exc:
            raise WorkflowRepositoryError(
                f"Unable to persist workflow decision: {normalized}"
            ) from exc
