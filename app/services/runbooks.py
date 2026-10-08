from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import Settings, get_settings
from app.schemas.agents import IncidentSeverity, RecoveryStep


class RunbookError(RuntimeError):
    """Raised when the runbook catalog is missing, invalid, or inconsistent."""


class SymptomRule(BaseModel):
    """One deterministic evidence rule declared by a runbook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=100)
    operator: Literal[
        "increase_from_baseline",
        "increase_from_baseline_percent",
        "decrease_from_baseline_percent",
        "greater_than",
        "greater_than_or_equal",
        "less_than_or_equal",
        "monotonic_increase",
        "non_zero",
    ]
    threshold: float
    unit: str = Field(min_length=1, max_length=30)
    weight: float = Field(gt=0.0, le=1.0)


class ProbableCause(BaseModel):
    """A runbook-backed diagnosis candidate and its prior confidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cause: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=1000)
    confidence_prior: float = Field(ge=0.0, le=1.0)


class SeverityPolicy(BaseModel):
    """Supported severities and minimum evidence confidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: tuple[IncidentSeverity, ...] = Field(min_length=1)
    minimum_agent_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_unique_severities(self) -> "SeverityPolicy":
        if len(self.supported) != len(set(self.supported)):
            raise ValueError("supported severity values must be unique")
        return self


class RecoveryPolicy(BaseModel):
    """Human-controlled, allowlisted recovery policy from a runbook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_required: Literal[True]
    checkpoint_required: bool
    expected_downtime_minutes: int = Field(ge=0, le=1440)
    risk_level: Literal["low", "medium", "high", "critical"]
    actions: tuple[RecoveryStep, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_action_order(self) -> "RecoveryPolicy":
        orders = [action.order for action in self.actions]
        expected = list(range(1, len(orders) + 1))
        if orders != expected:
            raise ValueError(
                "recovery action order must be contiguous and start at 1"
            )
        return self


class Runbook(BaseModel):
    """Immutable, schema-validated operational runbook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    runbook_id: str = Field(
        min_length=1,
        max_length=150,
        pattern=r"^gpu\.[a-z0-9_]+\.v[1-9][0-9]*$",
    )
    scenario: Literal[
        "thermal_throttling",
        "ecc_memory_degradation",
        "memory_saturation",
        "power_capping",
        "nvlink_degradation",
        "idle_gpu_waste",
    ]
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1200)
    severity_policy: SeverityPolicy
    symptoms: tuple[SymptomRule, ...] = Field(min_length=1, max_length=20)
    probable_causes: tuple[ProbableCause, ...] = Field(min_length=1, max_length=10)
    safety_checks: tuple[str, ...] = Field(min_length=1, max_length=20)
    recovery: RecoveryPolicy
    rollback_plan: str = Field(min_length=1, max_length=1500)
    escalation_conditions: tuple[str, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_semantics(self) -> "Runbook":
        expected_id = f"gpu.{self.scenario}.v1"
        if self.runbook_id != expected_id:
            raise ValueError(
                f"runbook_id must match scenario and version: {expected_id}"
            )

        metrics = [rule.metric for rule in self.symptoms]
        if len(metrics) != len(set(metrics)):
            raise ValueError("symptom metrics must be unique within a runbook")

        weight_sum = sum(rule.weight for rule in self.symptoms)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"symptom weights must sum to 1.0, found {weight_sum:.6f}"
            )

        causes = [candidate.cause for candidate in self.probable_causes]
        if len(causes) != len(set(causes)):
            raise ValueError("probable cause names must be unique")

        if any(not item.strip() for item in self.safety_checks):
            raise ValueError("safety checks cannot contain blank values")
        if any(not item.strip() for item in self.escalation_conditions):
            raise ValueError("escalation conditions cannot contain blank values")

        return self


class RunbookCatalog:
    """Read-only catalog providing total scenario coverage."""

    REQUIRED_SCENARIOS = frozenset(
        {
            "thermal_throttling",
            "ecc_memory_degradation",
            "memory_saturation",
            "power_capping",
            "nvlink_degradation",
            "idle_gpu_waste",
        }
    )

    def __init__(self, runbooks: Mapping[str, Runbook]) -> None:
        copied = dict(runbooks)
        scenarios = set(copied)
        missing = self.REQUIRED_SCENARIOS.difference(scenarios)
        extra = scenarios.difference(self.REQUIRED_SCENARIOS)
        if missing or extra:
            raise RunbookError(
                "Runbook scenario coverage mismatch. "
                f"Missing: {sorted(missing)}. Extra: {sorted(extra)}"
            )

        ids = [runbook.runbook_id for runbook in copied.values()]
        if len(ids) != len(set(ids)):
            raise RunbookError("Runbook IDs must be unique")

        self._runbooks: Mapping[str, Runbook] = MappingProxyType(copied)

    def get(self, scenario: str) -> Runbook:
        try:
            return self._runbooks[scenario]
        except KeyError as exc:
            available = ", ".join(sorted(self._runbooks))
            raise RunbookError(
                f"Unknown runbook scenario '{scenario}'. Available: {available}"
            ) from exc

    def all(self) -> tuple[Runbook, ...]:
        return tuple(self._runbooks[key] for key in sorted(self._runbooks))

    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._runbooks))

    def supports_severity(
        self,
        scenario: str,
        severity: IncidentSeverity | str,
    ) -> bool:
        try:
            value = IncidentSeverity(severity)
        except ValueError:
            return False
        return value in self.get(scenario).severity_policy.supported

    def __len__(self) -> int:
        return len(self._runbooks)


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunbookError(f"Unable to read runbook: {path}") from exc

    if not content.strip():
        raise RunbookError(f"Runbook file is empty: {path}")

    try:
        loaded = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise RunbookError(f"Invalid YAML in runbook: {path}") from exc

    if not isinstance(loaded, dict):
        raise RunbookError(f"Runbook root must be a mapping: {path}")
    return loaded


def load_runbook_catalog(directory: Path) -> RunbookCatalog:
    """Load exactly one validated YAML runbook per required scenario."""

    root = directory.expanduser().resolve()
    if not root.exists():
        raise RunbookError(f"Runbook directory does not exist: {root}")
    if not root.is_dir():
        raise RunbookError(f"Runbook path is not a directory: {root}")

    paths = sorted((*root.glob("*.yaml"), *root.glob("*.yml")))
    if not paths:
        raise RunbookError(f"No YAML runbooks found in: {root}")

    by_scenario: dict[str, Runbook] = {}
    source_by_scenario: dict[str, Path] = {}

    for path in paths:
        raw = _read_yaml_mapping(path)
        try:
            runbook = Runbook.model_validate(raw)
        except Exception as exc:
            raise RunbookError(
                f"Runbook validation failed for {path.name}: {exc}"
            ) from exc

        if runbook.scenario in by_scenario:
            previous = source_by_scenario[runbook.scenario]
            raise RunbookError(
                f"Duplicate scenario '{runbook.scenario}' in "
                f"{previous.name} and {path.name}"
            )

        by_scenario[runbook.scenario] = runbook
        source_by_scenario[runbook.scenario] = path

    return RunbookCatalog(by_scenario)


@lru_cache(maxsize=4)
def _cached_catalog(directory: str) -> RunbookCatalog:
    return load_runbook_catalog(Path(directory))


def get_runbook_catalog(settings: Settings | None = None) -> RunbookCatalog:
    """Return the cached immutable catalog for the configured runbook path."""

    runtime_settings = settings or get_settings()
    return _cached_catalog(str(runtime_settings.runbook_directory.resolve()))


def clear_runbook_cache() -> None:
    """Clear the catalog cache for tests and controlled configuration reloads."""

    _cached_catalog.cache_clear()
