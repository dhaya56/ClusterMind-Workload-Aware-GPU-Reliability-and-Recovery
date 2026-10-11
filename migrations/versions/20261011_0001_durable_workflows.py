"""Create durable workflow persistence tables.

Revision ID: 20261011_0001
Revises: None
Create Date: 2026-10-11 21:40:00+05:30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20261011_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UTC_NOW = sa.text("timezone('utc', now())")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.create_table(
        "workflow_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("workflow_id", sa.String(150), nullable=False),
        sa.Column("incident_id", sa.String(100), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("current_stage", sa.String(40), nullable=False, server_default="created"),
        sa.Column("execution_status", sa.String(30), nullable=False, server_default="not_started"),
        sa.Column("requested_use_llm", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.Column("trace_id", sa.String(32)),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("failure_code", sa.String(100)),
        sa.Column("failure_summary", sa.String(1000)),
        sa.Column("telemetry_result", postgresql.JSONB()),
        sa.Column("workload_impact_result", postgresql.JSONB()),
        sa.Column("root_cause_result", postgresql.JSONB()),
        sa.Column("recovery_plan_result", postgresql.JSONB()),
        sa.Column("errors", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.CheckConstraint("status IN ('created','queued','running','suppressed','awaiting_approval','approved','rejected','more_evidence_requested','completed','failed','cancelled')", name="ck_workflow_runs_workflow_status_allowed"),
        sa.CheckConstraint("current_stage IN ('created','loading_evidence','gpu_telemetry','workload_impact','root_cause','recovery','human_approval','execution','completed','failed')", name="ck_workflow_runs_workflow_current_stage_allowed"),
        sa.CheckConstraint("execution_status IN ('not_started','queued','simulating','completed','failed','cancelled')", name="ck_workflow_runs_workflow_execution_status_allowed"),
        sa.CheckConstraint("version >= 1", name="ck_workflow_runs_workflow_version_positive"),
        sa.CheckConstraint("jsonb_typeof(errors) = 'array'", name="ck_workflow_runs_workflow_errors_array"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_workflow_runs_workflow_metadata_object"),
        sa.CheckConstraint("telemetry_result IS NULL OR jsonb_typeof(telemetry_result) = 'object'", name="ck_workflow_runs_workflow_telemetry_object"),
        sa.CheckConstraint("workload_impact_result IS NULL OR jsonb_typeof(workload_impact_result) = 'object'", name="ck_workflow_runs_workflow_impact_object"),
        sa.CheckConstraint("root_cause_result IS NULL OR jsonb_typeof(root_cause_result) = 'object'", name="ck_workflow_runs_workflow_root_cause_object"),
        sa.CheckConstraint("recovery_plan_result IS NULL OR jsonb_typeof(recovery_plan_result) = 'object'", name="ck_workflow_runs_workflow_recovery_object"),
        sa.CheckConstraint("updated_at >= started_at", name="ck_workflow_runs_workflow_timestamp_order"),
        sa.CheckConstraint("completed_at IS NULL OR completed_at >= started_at", name="ck_workflow_runs_workflow_completed_timestamp_order"),
        sa.UniqueConstraint("workflow_id", name="uq_workflow_runs_workflow_public_id"),
    )
    op.create_index("ix_workflow_runs_incident_id", "workflow_runs", ["incident_id"])
    op.create_index("ix_workflow_runs_incident_updated", "workflow_runs", ["incident_id", "updated_at"])
    op.create_index("ix_workflow_runs_status_updated", "workflow_runs", ["status", "updated_at"])
    op.create_index("uq_workflow_runs_active_incident", "workflow_runs", ["incident_id"], unique=True, postgresql_where=sa.text("status IN ('created','queued','running','awaiting_approval','approved','more_evidence_requested')"))

    op.create_table(
        "agent_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("workflow_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_name", sa.String(50), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("duration_ms", sa.Float()),
        sa.Column("trace_id", sa.String(32)),
        sa.Column("used_llm", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fallback_used", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_code", sa.String(100)),
        sa.Column("error_message", sa.String(1000)),
        sa.Column("result_payload", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.CheckConstraint("agent_name IN ('gpu_telemetry_agent','workload_impact_agent','root_cause_agent','recovery_agent')", name="ck_agent_executions_agent_name_allowed"),
        sa.CheckConstraint("status IN ('pending','running','completed','failed','skipped')", name="ck_agent_executions_agent_status_allowed"),
        sa.CheckConstraint("attempt_number >= 1", name="ck_agent_executions_agent_attempt_positive"),
        sa.CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_agent_executions_agent_duration_nonnegative"),
        sa.CheckConstraint("completed_at IS NULL OR started_at IS NOT NULL", name="ck_agent_executions_agent_completed_requires_started"),
        sa.CheckConstraint("completed_at IS NULL OR completed_at >= started_at", name="ck_agent_executions_agent_timestamp_order"),
        sa.CheckConstraint("result_payload IS NULL OR jsonb_typeof(result_payload) = 'object'", name="ck_agent_executions_agent_result_object"),
        sa.CheckConstraint("NOT (used_llm AND fallback_used)", name="ck_agent_executions_agent_llm_fallback_exclusive"),
        sa.UniqueConstraint("workflow_run_id", "agent_name", "attempt_number", name="uq_agent_executions_agent_attempt"),
    )
    op.create_index("ix_agent_executions_workflow_status", "agent_executions", ["workflow_run_id", "status"])

    op.create_table(
        "approval_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("workflow_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("recommendation_id", sa.String(150), nullable=False),
        sa.Column("decision", sa.String(40), nullable=False),
        sa.Column("actor", sa.String(150), nullable=False),
        sa.Column("reason", sa.String(1000)),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("workflow_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.CheckConstraint("decision IN ('approved','rejected','more_evidence_requested')", name="ck_approval_decisions_approval_decision_allowed"),
        sa.CheckConstraint("decision = 'approved' OR (reason IS NOT NULL AND length(trim(reason)) > 0)", name="ck_approval_decisions_approval_reason_required"),
        sa.CheckConstraint("workflow_version >= 1", name="ck_approval_decisions_approval_workflow_version_positive"),
        sa.UniqueConstraint("workflow_run_id", name="uq_approval_decisions_approval_one_per_workflow"),
    )
    op.create_index("ix_approval_decisions_decided_at", "approval_decisions", ["decided_at"])

    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("workflow_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence_number", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("previous_status", sa.String(40)),
        sa.Column("new_status", sa.String(40)),
        sa.Column("actor_type", sa.String(40), nullable=False),
        sa.Column("actor_id", sa.String(150)),
        sa.Column("trace_id", sa.String(32)),
        sa.Column("event_payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.CheckConstraint("sequence_number >= 1", name="ck_audit_events_audit_sequence_positive"),
        sa.CheckConstraint("jsonb_typeof(event_payload) = 'object'", name="ck_audit_events_audit_payload_object"),
        sa.UniqueConstraint("workflow_run_id", "sequence_number", name="uq_audit_events_audit_sequence"),
    )
    op.create_index("ix_audit_events_workflow_occurred", "audit_events", ["workflow_run_id", "occurred_at"])
    op.create_index("ix_audit_events_event_type_occurred", "audit_events", ["event_type", "occurred_at"])

    op.create_table(
        "outbox_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("aggregate_type", sa.String(80), nullable=False),
        sa.Column("aggregate_id", sa.String(150), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=UTC_NOW),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.String(1000)),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("locked_by", sa.String(150)),
        sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_outbox_events_outbox_payload_object"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_outbox_events_outbox_attempt_nonnegative"),
    )
    op.create_index("ix_outbox_events_pending", "outbox_events", ["available_at"], postgresql_where=sa.text("published_at IS NULL"))
    op.create_index("ix_outbox_events_aggregate", "outbox_events", ["aggregate_type", "aggregate_id"])


def downgrade() -> None:
    op.drop_index("ix_outbox_events_aggregate", table_name="outbox_events")
    op.drop_index("ix_outbox_events_pending", table_name="outbox_events")
    op.drop_table("outbox_events")
    op.drop_index("ix_audit_events_event_type_occurred", table_name="audit_events")
    op.drop_index("ix_audit_events_workflow_occurred", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_approval_decisions_decided_at", table_name="approval_decisions")
    op.drop_table("approval_decisions")
    op.drop_index("ix_agent_executions_workflow_status", table_name="agent_executions")
    op.drop_table("agent_executions")
    op.drop_index("uq_workflow_runs_active_incident", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_status_updated", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_incident_updated", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_incident_id", table_name="workflow_runs")
    op.drop_table("workflow_runs")
