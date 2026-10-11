from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


WORKFLOW_STATUSES = (
    "created",
    "queued",
    "running",
    "suppressed",
    "awaiting_approval",
    "approved",
    "rejected",
    "more_evidence_requested",
    "completed",
    "failed",
    "cancelled",
)
CURRENT_STAGES = (
    "created",
    "loading_evidence",
    "gpu_telemetry",
    "workload_impact",
    "root_cause",
    "recovery",
    "human_approval",
    "execution",
    "completed",
    "failed",
)
EXECUTION_STATUSES = (
    "not_started",
    "queued",
    "simulating",
    "completed",
    "failed",
    "cancelled",
)
AGENT_NAMES = (
    "gpu_telemetry_agent",
    "workload_impact_agent",
    "root_cause_agent",
    "recovery_agent",
)
AGENT_STATUSES = ("pending", "running", "completed", "failed", "skipped")
APPROVAL_DECISIONS = ("approved", "rejected", "more_evidence_requested")


def _utc_server_now():
    return text("timezone('utc', now())")


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN (" + ",".join(f"'{value}'" for value in WORKFLOW_STATUSES) + ")",
            name="workflow_status_allowed",
        ),
        CheckConstraint(
            "current_stage IN (" + ",".join(f"'{value}'" for value in CURRENT_STAGES) + ")",
            name="workflow_current_stage_allowed",
        ),
        CheckConstraint(
            "execution_status IN (" + ",".join(f"'{value}'" for value in EXECUTION_STATUSES) + ")",
            name="workflow_execution_status_allowed",
        ),
        CheckConstraint("version >= 1", name="workflow_version_positive"),
        CheckConstraint("jsonb_typeof(errors) = 'array'", name="workflow_errors_array"),
        CheckConstraint("jsonb_typeof(metadata) = 'object'", name="workflow_metadata_object"),
        CheckConstraint("telemetry_result IS NULL OR jsonb_typeof(telemetry_result) = 'object'", name="workflow_telemetry_object"),
        CheckConstraint("workload_impact_result IS NULL OR jsonb_typeof(workload_impact_result) = 'object'", name="workflow_impact_object"),
        CheckConstraint("root_cause_result IS NULL OR jsonb_typeof(root_cause_result) = 'object'", name="workflow_root_cause_object"),
        CheckConstraint("recovery_plan_result IS NULL OR jsonb_typeof(recovery_plan_result) = 'object'", name="workflow_recovery_object"),
        CheckConstraint("updated_at >= started_at", name="workflow_timestamp_order"),
        CheckConstraint("completed_at IS NULL OR completed_at >= started_at", name="workflow_completed_timestamp_order"),
        UniqueConstraint("workflow_id", name="uq_workflow_runs_workflow_public_id"),
        Index("ix_workflow_runs_incident_updated", "incident_id", "updated_at"),
        Index("ix_workflow_runs_status_updated", "status", "updated_at"),
        Index(
            "uq_workflow_runs_active_incident",
            "incident_id",
            unique=True,
            postgresql_where=text(
                "status IN ('created','queued','running','awaiting_approval','approved','more_evidence_requested')"
            ),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[str] = mapped_column(String(150), nullable=False)
    incident_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(40), nullable=False, default="created")
    execution_status: Mapped[str] = mapped_column(String(30), nullable=False, default="not_started")
    requested_use_llm: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())
    trace_id: Mapped[str | None] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    failure_code: Mapped[str | None] = mapped_column(String(100))
    failure_summary: Mapped[str | None] = mapped_column(String(1000))
    telemetry_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    workload_impact_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    root_cause_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    recovery_plan_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    errors: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    metadata_payload: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))

    agent_executions: Mapped[list[AgentExecutionRecord]] = relationship(back_populates="workflow", cascade="all, delete-orphan")
    approval_decisions: Mapped[list[ApprovalDecisionRecord]] = relationship(back_populates="workflow", cascade="all, delete-orphan")
    audit_events: Mapped[list[AuditEvent]] = relationship(back_populates="workflow", cascade="all, delete-orphan")


class AgentExecutionRecord(Base):
    __tablename__ = "agent_executions"
    __table_args__ = (
        CheckConstraint("agent_name IN (" + ",".join(f"'{value}'" for value in AGENT_NAMES) + ")", name="agent_name_allowed"),
        CheckConstraint("status IN (" + ",".join(f"'{value}'" for value in AGENT_STATUSES) + ")", name="agent_status_allowed"),
        CheckConstraint("attempt_number >= 1", name="agent_attempt_positive"),
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="agent_duration_nonnegative"),
        CheckConstraint("completed_at IS NULL OR started_at IS NOT NULL", name="agent_completed_requires_started"),
        CheckConstraint("completed_at IS NULL OR completed_at >= started_at", name="agent_timestamp_order"),
        CheckConstraint("result_payload IS NULL OR jsonb_typeof(result_payload) = 'object'", name="agent_result_object"),
        CheckConstraint("NOT (used_llm AND fallback_used)", name="agent_llm_fallback_exclusive"),
        UniqueConstraint("workflow_run_id", "agent_name", "attempt_number", name="uq_agent_executions_agent_attempt"),
        Index("ix_agent_executions_workflow_status", "workflow_run_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    trace_id: Mapped[str | None] = mapped_column(String(32))
    used_llm: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())

    workflow: Mapped[WorkflowRun] = relationship(back_populates="agent_executions")


class ApprovalDecisionRecord(Base):
    __tablename__ = "approval_decisions"
    __table_args__ = (
        CheckConstraint("decision IN (" + ",".join(f"'{value}'" for value in APPROVAL_DECISIONS) + ")", name="approval_decision_allowed"),
        CheckConstraint("decision = 'approved' OR (reason IS NOT NULL AND length(trim(reason)) > 0)", name="approval_reason_required"),
        CheckConstraint("workflow_version >= 1", name="approval_workflow_version_positive"),
        UniqueConstraint("workflow_run_id", name="uq_approval_decisions_approval_one_per_workflow"),
        Index("ix_approval_decisions_decided_at", "decided_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflow_runs.id", ondelete="RESTRICT"), nullable=False)
    recommendation_id: Mapped[str] = mapped_column(String(150), nullable=False)
    decision: Mapped[str] = mapped_column(String(40), nullable=False)
    actor: Mapped[str] = mapped_column(String(150), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(1000))
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    workflow_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())

    workflow: Mapped[WorkflowRun] = relationship(back_populates="approval_decisions")


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        CheckConstraint("sequence_number >= 1", name="audit_sequence_positive"),
        CheckConstraint("jsonb_typeof(event_payload) = 'object'", name="audit_payload_object"),
        UniqueConstraint("workflow_run_id", "sequence_number", name="uq_audit_events_audit_sequence"),
        Index("ix_audit_events_workflow_occurred", "workflow_run_id", "occurred_at"),
        Index("ix_audit_events_event_type_occurred", "event_type", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False)
    sequence_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String(40))
    new_status: Mapped[str | None] = mapped_column(String(40))
    actor_type: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(150))
    trace_id: Mapped[str | None] = mapped_column(String(32))
    event_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())

    workflow: Mapped[WorkflowRun] = relationship(back_populates="audit_events")


class OutboxEvent(Base):
    __tablename__ = "outbox_events"
    __table_args__ = (
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="outbox_payload_object"),
        CheckConstraint("attempt_count >= 0", name="outbox_attempt_nonnegative"),
        Index("ix_outbox_events_pending", "available_at", postgresql_where=text("published_at IS NULL")),
        Index("ix_outbox_events_aggregate", "aggregate_type", "aggregate_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    aggregate_type: Mapped[str] = mapped_column(String(80), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(150), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=_utc_server_now())
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(String(1000))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(150))
