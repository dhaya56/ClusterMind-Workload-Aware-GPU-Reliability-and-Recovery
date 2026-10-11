"""Add durable asynchronous workflow lifecycle metadata.

Revision ID: 20261012_0002
Revises: 20261011_0001
Create Date: 2026-10-12 02:20:00+05:30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20261012_0002"
down_revision: Union[str, None] = "20261011_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "workflow_runs",
        sa.Column("task_id", sa.String(150), nullable=True),
    )
    op.add_column(
        "workflow_runs",
        sa.Column(
            "dispatch_status",
            sa.String(30),
            nullable=False,
            server_default="not_dispatched",
        ),
    )
    op.add_column(
        "workflow_runs",
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_runs",
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_runs",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_runs",
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "workflow_runs",
        sa.Column("worker_id", sa.String(150), nullable=True),
    )
    op.add_column(
        "workflow_runs",
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "workflow_runs",
        sa.Column("dispatch_error", sa.String(1000), nullable=True),
    )

    op.create_check_constraint(
        "ck_workflow_runs_dispatch_status_allowed",
        "workflow_runs",
        "dispatch_status IN ("
        "'not_dispatched','pending','dispatched','claimed','completed','failed'"
        ")",
    )
    op.create_check_constraint(
        "ck_workflow_runs_retry_count_nonnegative",
        "workflow_runs",
        "retry_count >= 0",
    )
    op.create_check_constraint(
        "ck_workflow_runs_dispatched_requires_task",
        "workflow_runs",
        "dispatch_status NOT IN ('dispatched','claimed','completed') "
        "OR task_id IS NOT NULL",
    )
    op.create_check_constraint(
        "ck_workflow_runs_claimed_requires_worker",
        "workflow_runs",
        "dispatch_status != 'claimed' OR "
        "(worker_id IS NOT NULL AND claimed_at IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_workflow_runs_dispatch_error_state",
        "workflow_runs",
        "dispatch_error IS NULL OR dispatch_status = 'failed'",
    )
    op.create_check_constraint(
        "ck_workflow_runs_async_timestamp_order",
        "workflow_runs",
        "(dispatched_at IS NULL OR queued_at IS NULL OR dispatched_at >= queued_at) "
        "AND (claimed_at IS NULL OR dispatched_at IS NULL OR claimed_at >= dispatched_at) "
        "AND (last_heartbeat_at IS NULL OR claimed_at IS NULL "
        "OR last_heartbeat_at >= claimed_at)",
    )

    op.create_unique_constraint(
        "uq_workflow_runs_task_id",
        "workflow_runs",
        ["task_id"],
    )
    op.create_index(
        "ix_workflow_runs_dispatch_status_queued",
        "workflow_runs",
        ["dispatch_status", "queued_at"],
    )
    op.create_index(
        "ix_workflow_runs_stale_claims",
        "workflow_runs",
        ["last_heartbeat_at"],
        postgresql_where=sa.text("dispatch_status = 'claimed'"),
    )


def downgrade() -> None:
    op.drop_index("ix_workflow_runs_stale_claims", table_name="workflow_runs")
    op.drop_index(
        "ix_workflow_runs_dispatch_status_queued",
        table_name="workflow_runs",
    )
    op.drop_constraint(
        "uq_workflow_runs_task_id",
        "workflow_runs",
        type_="unique",
    )
    op.drop_constraint(
        "ck_workflow_runs_async_timestamp_order",
        "workflow_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_workflow_runs_dispatch_error_state",
        "workflow_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_workflow_runs_claimed_requires_worker",
        "workflow_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_workflow_runs_dispatched_requires_task",
        "workflow_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_workflow_runs_retry_count_nonnegative",
        "workflow_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_workflow_runs_dispatch_status_allowed",
        "workflow_runs",
        type_="check",
    )
    for column in (
        "dispatch_error",
        "retry_count",
        "worker_id",
        "last_heartbeat_at",
        "claimed_at",
        "dispatched_at",
        "queued_at",
        "dispatch_status",
        "task_id",
    ):
        op.drop_column("workflow_runs", column)
