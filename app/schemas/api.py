from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.agents import ApprovalDecision, WorkflowResult


class AnalyzeIncidentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_llm: bool = True
    force: bool = False


class AnalysisDispatchResponse(BaseModel):
    """Accepted durable asynchronous analysis request."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    task_id: str
    status: Literal["queued", "running"]
    dispatch_status: Literal["pending", "dispatched", "claimed"]
    status_url: str


class WorkflowDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected", "more_evidence_requested"]
    actor: str = Field(min_length=1, max_length=150)
    reason: str | None = Field(default=None, max_length=1000)

    @field_validator("actor")
    @classmethod
    def normalize_actor(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("actor cannot be blank")
        return normalized

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def approval_decision(self) -> ApprovalDecision:
        return ApprovalDecision(self.decision)


class IncidentListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    incident_type: str
    severity: str
    node_id: str
    gpu_id: str
    affected_workload: str
    start_timestamp: datetime
    end_timestamp: datetime
    incident_status: str


class IncidentListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[IncidentListItem]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=1000)
    offset: int = Field(ge=0)


class IncidentDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    node_id: str
    gpu_id: str
    incident_type: str
    severity: str
    start_timestamp: datetime
    end_timestamp: datetime
    root_cause: str
    affected_workload: str
    gpu_hours_lost: float
    recovery_action: str
    incident_status: str
    runbook_id: str
    runbook_title: str


class OverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_count: int = Field(ge=0)
    severity_counts: dict[str, int]
    incident_type_counts: dict[str, int]
    node_count: int = Field(ge=0)
    gpu_count: int = Field(ge=0)
    workload_count: int = Field(ge=0)
    sampling_interval_minutes: int = Field(ge=1)
    workflow_count: int = Field(ge=0)
    workflow_status_counts: dict[str, int]
    pending_approval_count: int = Field(ge=0)


class WorkflowListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[WorkflowResult]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=1000)


class PlatformComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["healthy", "degraded", "unavailable"]
    detail: str


class PlatformStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["healthy", "degraded", "unavailable"]
    timestamp: datetime
    components: list[PlatformComponent]
    capabilities: dict[str, Any]
