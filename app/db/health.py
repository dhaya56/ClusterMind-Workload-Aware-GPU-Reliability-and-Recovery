from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DatabaseStatus = Literal["healthy", "unavailable"]


@dataclass(frozen=True, slots=True)
class DatabaseHealth:
    """Small database readiness result without credential disclosure."""

    status: DatabaseStatus
    latency_ms: float
    database: str | None = None
    server_version: str | None = None
    error_type: str | None = None
