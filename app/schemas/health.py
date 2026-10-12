from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ServiceStatus = Literal["healthy", "degraded", "unavailable", "not_configured"]


class ComponentHealth(BaseModel):
    """Health state of one ClusterMind dependency or component."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    status: ServiceStatus
    message: str = Field(default="", max_length=500)
    latency_ms: float | None = Field(default=None, ge=0.0)


class HealthResponse(BaseModel):
    """Liveness response for the ClusterMind API process."""

    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    environment: str
    status: Literal["healthy"]
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness response containing application dependency states."""

    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    environment: str
    status: Literal["healthy", "degraded", "unavailable"]
    ready: bool
    timestamp: datetime
    components: list[ComponentHealth]


class RootResponse(BaseModel):
    """Metadata returned from the API root endpoint."""

    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    environment: str
    status: Literal["running"]
    documentation_url: str
    health_url: str
    readiness_url: str
    metrics_url: str