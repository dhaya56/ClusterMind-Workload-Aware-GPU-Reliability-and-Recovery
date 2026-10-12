from __future__ import annotations

import math
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable, Mapping

from opentelemetry.trace import Span, Status, StatusCode

from app.observability.tracing import get_tracer
from app.schemas.agents import (
    AgentExecution,
    AgentName,
    AgentStatus,
    EvidenceItem,
    TelemetryAgentResult,
    WorkloadImpactResult,
    utc_now,
)
from app.services.incidents import IncidentEvidencePackage


class WorkloadImpactAgentError(RuntimeError):
    """Raised when workload impact cannot be evaluated safely."""


class WorkloadImpactAgent:
    """Correlate GPU anomalies with useful workload progress and degradation."""

    HARMFUL_THROUGHPUT_REDUCTION_PERCENT = 10.0
    HARMFUL_LATENCY_INCREASE_PERCENT = 15.0
    HARMFUL_STEP_INCREASE_PERCENT = 15.0
    HARMFUL_ERROR_RATE_PERCENT = 2.0
    PRODUCTIVE_UTILIZATION_PERCENT = 85.0
    PRODUCTIVE_THROUGHPUT_IMPROVEMENT_PERCENT = 5.0

    def __init__(self) -> None:
        self._tracer = get_tracer("clustermind.agents.workload_impact")

    @staticmethod
    def _numeric_values(
        records: Iterable[Mapping[str, Any]],
        metric: str,
    ) -> list[float]:
        values: list[float] = []
        for record in records:
            value = record.get(metric)
            if value is None or isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        return values

    @classmethod
    def _mean(
        cls,
        records: Iterable[Mapping[str, Any]],
        metric: str,
    ) -> float | None:
        values = cls._numeric_values(records, metric)
        return fmean(values) if values else None

    @staticmethod
    def _percent_change(baseline: float | None, current: float | None) -> float | None:
        if baseline is None or current is None or abs(baseline) < 1e-12:
            return None
        return ((current - baseline) / abs(baseline)) * 100.0

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "", "none", "nan"}:
            return False
        raise WorkloadImpactAgentError(
            f"Unrecognized checkpoint_available value: {value!r}"
        )

    @staticmethod
    def _trace_id(span: Span) -> str | None:
        context = span.get_span_context()
        if not context.is_valid:
            return None
        return f"{context.trace_id:032x}"

    @staticmethod
    def _latest_timestamp(records: tuple[Mapping[str, Any], ...]) -> Any:
        return records[-1].get("timestamp") if records else None

    @staticmethod
    def _impact_level(score: float, hardware_override: bool) -> str:
        if hardware_override or score >= 0.85:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.40:
            return "medium"
        if score > 0.0:
            return "low"
        return "none"

    def analyze(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
    ) -> WorkloadImpactResult:
        """Measure operational impact and identify productive high utilization."""

        if telemetry.incident_id != evidence.incident.incident_id:
            raise WorkloadImpactAgentError(
                "Telemetry result and evidence package incident IDs do not match"
            )

        started_at = utc_now()
        started_clock = perf_counter()
        incident = evidence.incident
        workload_rows = evidence.workload_incident_window
        workload_baseline = evidence.workload_baseline

        assessment_sample_count = min(3, len(workload_rows))
        assessment_rows = workload_rows[-assessment_sample_count:]

        with self._tracer.start_as_current_span(
            "agent.workload_impact.analyze"
        ) as span:
            span.set_attribute(
                "clustermind.agent",
                AgentName.WORKLOAD_IMPACT.value,
            )
            span.set_attribute("clustermind.incident_id", incident.incident_id)
            span.set_attribute(
                "clustermind.workload_id",
                incident.affected_workload,
            )

            try:
                if not workload_rows or not workload_baseline:
                    raise WorkloadImpactAgentError(
                        "Workload baseline and incident-window evidence are required"
                    )

                workload_ids = {
                    str(row.get("job_id", "")).strip() for row in workload_rows
                }
                workload_ids.discard("")
                if workload_ids != {incident.affected_workload}:
                    raise WorkloadImpactAgentError(
                        "Workload evidence contains inconsistent job identifiers"
                    )

                workload_types = {
                    str(row.get("job_type", "")).strip() for row in workload_rows
                }
                workload_types.discard("")
                if len(workload_types) != 1:
                    raise WorkloadImpactAgentError(
                        "Workload evidence must contain exactly one job type"
                    )
                workload_type = next(iter(workload_types))

                checkpoint_values = {
                    self._as_bool(row.get("checkpoint_available"))
                    for row in workload_rows
                }
                if len(checkpoint_values) != 1:
                    raise WorkloadImpactAgentError(
                        "Checkpoint availability changed within one incident window"
                    )
                checkpoint_available = next(iter(checkpoint_values))

                throughput_metric = None
                if self._numeric_values(assessment_rows, "training_throughput"):
                    throughput_metric = "training_throughput"
                elif self._numeric_values(assessment_rows, "tokens_per_second"):
                    throughput_metric = "tokens_per_second"

                baseline_throughput = (
                    self._mean(workload_baseline, throughput_metric)
                    if throughput_metric
                    else None
                )
                incident_throughput = (
                    self._mean(assessment_rows, throughput_metric)
                    if throughput_metric
                    else None
                )
                throughput_change = self._percent_change(
                    baseline_throughput,
                    incident_throughput,
                )

                latency_metric = None
                if self._numeric_values(assessment_rows, "inference_latency_ms"):
                    latency_metric = "inference_latency_ms"
                elif self._numeric_values(assessment_rows, "step_duration_ms"):
                    latency_metric = "step_duration_ms"

                baseline_latency = (
                    self._mean(workload_baseline, latency_metric)
                    if latency_metric
                    else None
                )
                incident_latency = (
                    self._mean(assessment_rows, latency_metric)
                    if latency_metric
                    else None
                )
                latency_change = self._percent_change(
                    baseline_latency,
                    incident_latency,
                )

                error_rate = self._mean(assessment_rows, "error_rate_percent")
                restart_count = max(
                    self._numeric_values(assessment_rows, "restart_count") or [0.0]
                )
                gpu_utilization = self._mean(
                    evidence.gpu_incident_window,
                    "gpu_utilization_percent",
                )
                uncorrectable_ecc = max(
                    self._numeric_values(
                        evidence.gpu_incident_window,
                        "ecc_uncorrectable_errors",
                    )
                    or [0.0]
                )
                xid_code = max(
                    self._numeric_values(
                        evidence.gpu_incident_window,
                        "xid_error_code",
                    )
                    or [0.0]
                )

                throughput_reduction = (
                    max(0.0, -throughput_change)
                    if throughput_change is not None
                    else 0.0
                )
                latency_increase = (
                    max(0.0, latency_change)
                    if latency_change is not None
                    else 0.0
                )
                effective_error_rate = max(0.0, error_rate or 0.0)

                throughput_harm = (
                    throughput_change is not None
                    and throughput_reduction
                    >= self.HARMFUL_THROUGHPUT_REDUCTION_PERCENT
                )
                latency_harm = (
                    latency_change is not None
                    and latency_increase
                    >= self.HARMFUL_LATENCY_INCREASE_PERCENT
                )
                error_harm = (
                    effective_error_rate >= self.HARMFUL_ERROR_RATE_PERCENT
                    or restart_count > 0
                )
                hardware_harm = (
                    telemetry.hardware_override
                    or uncorrectable_ecc > 0
                    or xid_code != 0
                )

                healthy_productive_spike = bool(
                    gpu_utilization is not None
                    and gpu_utilization >= self.PRODUCTIVE_UTILIZATION_PERCENT
                    and throughput_change is not None
                    and throughput_change
                    >= self.PRODUCTIVE_THROUGHPUT_IMPROVEMENT_PERCENT
                    and not latency_harm
                    and not error_harm
                    and not hardware_harm
                )
                harmful_impact = bool(
                    not healthy_productive_spike
                    and (
                        throughput_harm
                        or latency_harm
                        or error_harm
                        or hardware_harm
                    )
                )

                evidence_items: list[EvidenceItem] = []
                timestamp = self._latest_timestamp(workload_rows)

                if throughput_change is not None and throughput_metric is not None:
                    direction = "increased" if throughput_change >= 0 else "decreased"
                    evidence_items.append(
                        EvidenceItem(
                            metric=throughput_metric,
                            observation=(
                                f"Useful workload throughput {direction} by "
                                f"{abs(throughput_change):.2f}% in the latest sustained window relative to baseline."
                            ),
                            value=(
                                None
                                if incident_throughput is None
                                else round(incident_throughput, 4)
                            ),
                            baseline=(
                                None
                                if baseline_throughput is None
                                else round(baseline_throughput, 4)
                            ),
                            unit=(
                                "samples_per_second"
                                if throughput_metric == "training_throughput"
                                else "tokens_per_second"
                            ),
                            source="workload_telemetry",
                            timestamp=timestamp,
                        )
                    )

                if latency_change is not None and latency_metric is not None:
                    direction = "increased" if latency_change >= 0 else "decreased"
                    evidence_items.append(
                        EvidenceItem(
                            metric=latency_metric,
                            observation=(
                                f"{latency_metric} {direction} by "
                                f"{abs(latency_change):.2f}% in the latest sustained window relative to baseline."
                            ),
                            value=(
                                None
                                if incident_latency is None
                                else round(incident_latency, 4)
                            ),
                            baseline=(
                                None
                                if baseline_latency is None
                                else round(baseline_latency, 4)
                            ),
                            unit="milliseconds",
                            source="workload_telemetry",
                            timestamp=timestamp,
                        )
                    )

                evidence_items.append(
                    EvidenceItem(
                        metric="error_rate_percent",
                        observation=(
                            f"Workload error rate was {effective_error_rate:.3f}%."
                        ),
                        value=round(effective_error_rate, 4),
                        unit="percent",
                        source="workload_telemetry",
                        timestamp=timestamp,
                    )
                )
                evidence_items.append(
                    EvidenceItem(
                        metric="checkpoint_available",
                        observation=(
                            "A recovery checkpoint is available."
                            if checkpoint_available
                            else "No recovery checkpoint is available."
                        ),
                        value=checkpoint_available,
                        source="workload_telemetry",
                        timestamp=timestamp,
                    )
                )

                harm_score = min(
                    1.0,
                    (min(throughput_reduction / 40.0, 1.0) * 0.35)
                    + (min(latency_increase / 60.0, 1.0) * 0.25)
                    + (min(effective_error_rate / 10.0, 1.0) * 0.20)
                    + (0.10 if restart_count > 0 else 0.0)
                    + (0.10 if hardware_harm else 0.0),
                )
                if healthy_productive_spike:
                    impact_level = "none"
                    confidence = min(
                        0.99,
                        0.70
                        + min(
                            max(throughput_change or 0.0, 0.0) / 100.0,
                            0.20,
                        )
                        + 0.09,
                    )
                elif harmful_impact:
                    impact_level = self._impact_level(
                        harm_score,
                        hardware_harm,
                    )
                    evidence_strength = sum(
                        (
                            throughput_harm,
                            latency_harm,
                            error_harm,
                            hardware_harm,
                        )
                    ) / 4.0
                    confidence = min(
                        0.99,
                        0.55
                        + (0.25 * evidence_strength)
                        + (0.20 * telemetry.confidence),
                    )
                else:
                    impact_level = "low"
                    confidence = min(
                        0.80,
                        0.45 + (0.25 * telemetry.confidence),
                    )

                if healthy_productive_spike:
                    summary = (
                        f"{incident.affected_workload} is productively using the GPU; "
                        "useful throughput improved without material latency, error, "
                        "ECC, or XID degradation. The candidate alert should be suppressed."
                    )
                elif harmful_impact:
                    changes: list[str] = []
                    if throughput_change is not None:
                        changes.append(
                            f"throughput changed by {throughput_change:.2f}%"
                        )
                    if latency_change is not None:
                        changes.append(f"latency changed by {latency_change:.2f}%")
                    if error_harm:
                        changes.append("errors or restarts were observed")
                    if hardware_harm:
                        changes.append("a hardware safety signal was present")
                    summary = (
                        f"{incident.affected_workload} experienced harmful "
                        f"{impact_level} impact: " + "; ".join(changes) + "."
                    )
                else:
                    summary = (
                        f"{incident.affected_workload} shows limited workload impact. "
                        "Continue monitoring and request additional evidence before recovery."
                    )

                completed_at = utc_now()
                result = WorkloadImpactResult(
                    execution=AgentExecution(
                        agent=AgentName.WORKLOAD_IMPACT,
                        status=AgentStatus.COMPLETED,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=round(
                            (perf_counter() - started_clock) * 1000.0,
                            3,
                        ),
                        trace_id=self._trace_id(span),
                    ),
                    incident_id=incident.incident_id,
                    workload_id=incident.affected_workload,
                    workload_type=workload_type,
                    harmful_impact=harmful_impact,
                    healthy_productive_spike=healthy_productive_spike,
                    throughput_change_percent=(
                        None
                        if throughput_change is None
                        else round(throughput_change, 6)
                    ),
                    latency_change_percent=(
                        None
                        if latency_change is None
                        else round(latency_change, 6)
                    ),
                    error_rate_percent=round(effective_error_rate, 6),
                    checkpoint_available=checkpoint_available,
                    impact_level=impact_level,
                    confidence=round(confidence, 6),
                    evidence=evidence_items,
                    summary=summary,
                )

                span.set_attribute(
                    "clustermind.harmful_impact",
                    result.harmful_impact,
                )
                span.set_attribute(
                    "clustermind.healthy_productive_spike",
                    result.healthy_productive_spike,
                )
                span.set_attribute(
                    "clustermind.impact_level",
                    result.impact_level,
                )
                span.set_attribute(
                    "clustermind.confidence",
                    result.confidence,
                )
                span.set_attribute(
                    "clustermind.checkpoint_available",
                    result.checkpoint_available,
                )
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                if isinstance(exc, WorkloadImpactAgentError):
                    raise
                raise WorkloadImpactAgentError(
                    f"Workload impact analysis failed for {incident.incident_id}"
                ) from exc
