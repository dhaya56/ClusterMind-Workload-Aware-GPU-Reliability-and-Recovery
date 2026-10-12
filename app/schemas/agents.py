from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentName(StrEnum):
    GPU_TELEMETRY = "gpu_telemetry_agent"
    WORKLOAD_IMPACT = "workload_impact_agent"
    ROOT_CAUSE = "root_cause_agent"
    RECOVERY = "recovery_agent"


class AgentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IncidentSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalDecision(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MORE_EVIDENCE_REQUESTED = "more_evidence_requested"


class RecoveryExecutionStatus(StrEnum):
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    SIMULATING = "simulating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceItem(BaseModel):
    """One auditable fact used by an agent to reach a decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=100)
    observation: str = Field(min_length=1, max_length=500)
    value: float | int | str | bool | None = None
    baseline: float | int | str | bool | None = None
    unit: str | None = Field(default=None, max_length=30)
    source: Literal[
        "gpu_telemetry",
        "node_telemetry",
        "workload_telemetry",
        "detector_results",
        "incident_record",
        "runbook",
        "system",
    ]
    timestamp: datetime | None = None


class AgentExecution(BaseModel):
    """Lifecycle metadata shared by every agent result."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    status: AgentStatus = AgentStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "AgentExecution":
        if self.completed_at and not self.started_at:
            raise ValueError("completed_at requires started_at")
        if self.started_at and self.completed_at:
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot precede started_at")
        if self.status == AgentStatus.FAILED and not self.error_message:
            raise ValueError("failed execution requires error_message")
        if self.status != AgentStatus.FAILED and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("error fields are only valid for failed execution")
        return self


class TelemetryAgentResult(BaseModel):
    """Structured interpretation of GPU and detector telemetry."""

    model_config = ConfigDict(extra="forbid")

    execution: AgentExecution
    incident_id: str = Field(min_length=1, max_length=100)
    node_id: str = Field(min_length=1, max_length=100)
    gpu_id: str = Field(min_length=1, max_length=100)
    condition: str = Field(min_length=1, max_length=200)
    severity: IncidentSeverity
    anomaly_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    hardware_override: bool = False
    override_reasons: list[str] = Field(default_factory=list, max_length=10)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=20)
    summary: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_override(self) -> "TelemetryAgentResult":
        if self.hardware_override and not self.override_reasons:
            raise ValueError("hardware_override requires at least one reason")
        if not self.hardware_override and self.override_reasons:
            raise ValueError("override_reasons require hardware_override=true")
        return self


class WorkloadImpactResult(BaseModel):
    """Structured impact assessment for the workload assigned to a GPU."""

    model_config = ConfigDict(extra="forbid")

    execution: AgentExecution
    incident_id: str = Field(min_length=1, max_length=100)
    workload_id: str = Field(min_length=1, max_length=100)
    workload_type: str = Field(min_length=1, max_length=100)
    harmful_impact: bool
    healthy_productive_spike: bool
    throughput_change_percent: float | None = Field(default=None, ge=-100.0)
    latency_change_percent: float | None = Field(default=None, ge=-100.0)
    error_rate_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    checkpoint_available: bool
    impact_level: Literal["none", "low", "medium", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=20)
    summary: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_impact(self) -> "WorkloadImpactResult":
        if self.healthy_productive_spike and self.harmful_impact:
            raise ValueError(
                "healthy_productive_spike and harmful_impact cannot both be true"
            )
        if self.healthy_productive_spike and self.impact_level != "none":
            raise ValueError("healthy spike must have impact_level='none'")
        if self.harmful_impact and self.impact_level == "none":
            raise ValueError("harmful impact cannot have impact_level='none'")
        return self


class RootCauseCandidate(BaseModel):
    """One ranked root-cause hypothesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cause: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(min_length=1, max_length=10)
    contradicting_evidence: list[str] = Field(default_factory=list, max_length=10)


class RootCauseResult(BaseModel):
    """Evidence-grounded diagnosis generated from telemetry and runbooks."""

    model_config = ConfigDict(extra="forbid")

    execution: AgentExecution
    incident_id: str = Field(min_length=1, max_length=100)
    probable_cause: str = Field(min_length=1, max_length=300)
    runbook_id: str = Field(min_length=1, max_length=150)
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: list[RootCauseCandidate] = Field(min_length=1, max_length=5)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=25)
    explanation: str = Field(min_length=1, max_length=2000)
    used_llm: bool
    fallback_used: bool

    @model_validator(mode="after")
    def validate_confidence(self) -> "RootCauseResult":
        ranked = sorted(
            self.candidates,
            key=lambda item: item.confidence,
            reverse=True,
        )
        if ranked[0].cause != self.probable_cause:
            raise ValueError("probable_cause must be the highest-confidence candidate")
        if abs(ranked[0].confidence - self.confidence) > 1e-6:
            raise ValueError("confidence must match the top candidate")
        if self.used_llm and self.fallback_used:
            raise ValueError("used_llm and fallback_used cannot both be true")
        return self


class RecoveryStep(BaseModel):
    """One allowlisted, non-shell recovery step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order: int = Field(ge=1, le=20)
    action: Literal[
        "monitor",
        "reduce_batch_size",
        "resume_from_checkpoint",
        "migrate_workload",
        "release_allocation",
        "restart_workload",
        "quarantine_gpu",
        "quarantine_node",
        "inspect_cooling",
        "inspect_nvlink",
        "review_power_profile",
        "escalate_to_operator",
    ]
    target: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=600)
    reversible: bool
    rollback: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def validate_rollback(self) -> "RecoveryStep":
        if self.reversible and not self.rollback:
            raise ValueError("reversible recovery step requires rollback")
        return self


class RecoveryPlanResult(BaseModel):
    """Checkpoint-aware recovery plan that always requires human approval."""

    model_config = ConfigDict(extra="forbid")

    execution: AgentExecution
    incident_id: str = Field(min_length=1, max_length=100)
    recommendation_id: str = Field(min_length=1, max_length=150)
    title: str = Field(min_length=1, max_length=300)
    risk_level: Literal["low", "medium", "high", "critical"]
    expected_downtime_minutes: int = Field(ge=0, le=1440)
    checkpoint_required: bool
    checkpoint_available: bool
    approval_required: Literal[True] = True
    confidence: float = Field(ge=0.0, le=1.0)
    steps: list[RecoveryStep] = Field(min_length=1, max_length=20)
    safety_checks: list[str] = Field(min_length=1, max_length=20)
    expected_outcome: str = Field(min_length=1, max_length=1000)
    rollback_plan: str = Field(min_length=1, max_length=1000)
    summary: str = Field(min_length=1, max_length=1600)
    used_llm: bool
    fallback_used: bool

    @model_validator(mode="after")
    def validate_plan(self) -> "RecoveryPlanResult":
        orders = [step.order for step in self.steps]
        if len(orders) != len(set(orders)):
            raise ValueError("recovery step order values must be unique")
        if orders != sorted(orders):
            raise ValueError("recovery steps must be sorted by order")
        if self.checkpoint_required and not self.checkpoint_available:
            if not any(
                step.action == "escalate_to_operator" for step in self.steps
            ):
                raise ValueError(
                    "missing checkpoint requires an escalation recovery step"
                )
        if self.used_llm and self.fallback_used:
            raise ValueError("used_llm and fallback_used cannot both be true")
        return self


class ApprovalRecord(BaseModel):
    """Auditable human decision for one recovery recommendation."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(min_length=1, max_length=100)
    recommendation_id: str = Field(min_length=1, max_length=150)
    decision: ApprovalDecision = ApprovalDecision.PENDING
    actor: str = Field(default="review-operator", min_length=1, max_length=150)
    reason: str | None = Field(default=None, max_length=1000)
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ApprovalRecord":
        if self.decision == ApprovalDecision.PENDING:
            if self.decided_at is not None or self.reason is not None:
                raise ValueError("pending approval cannot have decision metadata")
        else:
            if self.decided_at is None:
                raise ValueError("completed approval requires decided_at")
            if self.decision != ApprovalDecision.APPROVED and not self.reason:
                raise ValueError("non-approved decision requires a reason")
        return self


class WorkflowResult(BaseModel):
    """Complete persisted result of one ClusterMind agent workflow."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1, max_length=150)
    incident_id: str = Field(min_length=1, max_length=100)
    started_at: datetime
    updated_at: datetime
    status: Literal[
        "running",
        "suppressed",
        "awaiting_approval",
        "approved",
        "rejected",
        "more_evidence_requested",
        "completed",
        "failed",
    ]
    telemetry: TelemetryAgentResult | None = None
    workload_impact: WorkloadImpactResult | None = None
    root_cause: RootCauseResult | None = None
    recovery_plan: RecoveryPlanResult | None = None
    approval: ApprovalRecord | None = None
    execution_status: RecoveryExecutionStatus = (
        RecoveryExecutionStatus.NOT_STARTED
    )
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    errors: list[str] = Field(default_factory=list, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("updated_at")
    @classmethod
    def updated_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_workflow(self) -> "WorkflowResult":
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        if self.updated_at < self.started_at:
            raise ValueError("updated_at cannot precede started_at")
        if self.telemetry and self.telemetry.incident_id != self.incident_id:
            raise ValueError("telemetry incident_id does not match workflow")
        if self.workload_impact and (
            self.workload_impact.incident_id != self.incident_id
        ):
            raise ValueError("workload impact incident_id does not match workflow")
        if self.root_cause and self.root_cause.incident_id != self.incident_id:
            raise ValueError("root cause incident_id does not match workflow")
        if self.recovery_plan and (
            self.recovery_plan.incident_id != self.incident_id
        ):
            raise ValueError("recovery plan incident_id does not match workflow")
        if self.approval and self.approval.incident_id != self.incident_id:
            raise ValueError("approval incident_id does not match workflow")
        if self.status == "suppressed":
            if not self.workload_impact or not (
                self.workload_impact.healthy_productive_spike
            ):
                raise ValueError(
                    "suppressed workflow requires a healthy productive spike"
                )
        if self.status == "awaiting_approval" and not self.recovery_plan:
            raise ValueError("awaiting approval requires a recovery plan")
        if self.status == "failed" and not self.errors:
            raise ValueError("failed workflow requires at least one error")
        return self


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp without naive datetime risk."""

    return datetime.now(timezone.utc)
