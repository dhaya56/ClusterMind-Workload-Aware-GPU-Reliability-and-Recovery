from __future__ import annotations

import math
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable, Mapping

from opentelemetry.trace import Span, Status, StatusCode, get_current_span

from app.observability.tracing import get_tracer
from app.schemas.agents import (
    AgentExecution,
    AgentName,
    AgentStatus,
    EvidenceItem,
    IncidentSeverity,
    TelemetryAgentResult,
    utc_now,
)
from app.services.incidents import IncidentEvidencePackage
from app.services.runbooks import SymptomRule


class TelemetryAgentError(RuntimeError):
    """Raised when GPU evidence cannot be evaluated safely."""


class GPUTelemetryAgent:
    """Deterministic first-stage agent for GPU anomaly interpretation."""

    HARDWARE_OVERRIDE_CONFIDENCE = 0.99
    EXTREME_GPU_TEMPERATURE_C = 90.0
    EXTREME_MEMORY_TEMPERATURE_C = 95.0

    def __init__(self) -> None:
        self._tracer = get_tracer("clustermind.agents.telemetry")

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
    def _mean(cls, records: Iterable[Mapping[str, Any]], metric: str) -> float | None:
        values = cls._numeric_values(records, metric)
        return fmean(values) if values else None

    @classmethod
    def _max(cls, records: Iterable[Mapping[str, Any]], metric: str) -> float | None:
        values = cls._numeric_values(records, metric)
        return max(values) if values else None

    @classmethod
    def _min(cls, records: Iterable[Mapping[str, Any]], metric: str) -> float | None:
        values = cls._numeric_values(records, metric)
        return min(values) if values else None

    @staticmethod
    def _percent_change(baseline: float, current: float) -> float | None:
        if abs(baseline) < 1e-12:
            return None
        return ((current - baseline) / abs(baseline)) * 100.0

    @staticmethod
    def _latest_timestamp(records: tuple[Mapping[str, Any], ...]) -> Any:
        return records[-1].get("timestamp") if records else None

    def _evaluate_rule(
        self,
        rule: SymptomRule,
        evidence: IncidentEvidencePackage,
    ) -> tuple[bool, float | None, float | None, str]:
        metric = rule.metric
        baseline_records = (
            evidence.workload_baseline
            if metric in evidence.workload_baseline[0]
            else evidence.gpu_baseline
        )
        incident_records = (
            evidence.workload_incident_window
            if metric in evidence.workload_incident_window[0]
            else evidence.gpu_incident_window
        )

        baseline = self._mean(baseline_records, metric)
        current_mean = self._mean(incident_records, metric)
        current_max = self._max(incident_records, metric)
        current_min = self._min(incident_records, metric)
        incident_values = self._numeric_values(incident_records, metric)

        if not incident_values:
            return False, None, baseline, f"No numeric values available for {metric}."

        matched = False
        observed: float | None = current_mean
        observation = ""

        if rule.operator == "increase_from_baseline":
            if baseline is not None and current_max is not None:
                delta = current_max - baseline
                observed = current_max
                matched = delta >= rule.threshold
                observation = (
                    f"{metric} increased by {delta:.2f} {rule.unit} "
                    f"from the pre-incident baseline."
                )
        elif rule.operator == "increase_from_baseline_percent":
            if baseline is not None and current_max is not None:
                change = self._percent_change(baseline, current_max)
                if change is not None:
                    observed = current_max
                    matched = change >= rule.threshold
                    observation = (
                        f"{metric} increased by {change:.2f}% from baseline."
                    )
        elif rule.operator == "decrease_from_baseline_percent":
            if baseline is not None and current_min is not None:
                change = self._percent_change(baseline, current_min)
                if change is not None:
                    reduction = -change
                    observed = current_min
                    matched = reduction >= rule.threshold
                    observation = (
                        f"{metric} decreased by {reduction:.2f}% from baseline."
                    )
        elif rule.operator == "greater_than":
            observed = current_max
            matched = current_max is not None and current_max > rule.threshold
            observation = f"{metric} reached {current_max:.2f} {rule.unit}."
        elif rule.operator == "greater_than_or_equal":
            observed = current_max
            matched = current_max is not None and current_max >= rule.threshold
            observation = f"{metric} reached {current_max:.2f} {rule.unit}."
        elif rule.operator == "less_than_or_equal":
            observed = current_min
            matched = current_min is not None and current_min <= rule.threshold
            observation = f"{metric} fell to {current_min:.2f} {rule.unit}."
        elif rule.operator == "non_zero":
            observed = current_max
            matched = current_max is not None and current_max != 0
            observation = f"{metric} reported non-zero value {current_max:.0f}."
        elif rule.operator == "monotonic_increase":
            observed = incident_values[-1]
            comparisons = [
                current >= previous
                for previous, current in zip(incident_values, incident_values[1:])
            ]
            matched = (
                len(incident_values) >= 2
                and all(comparisons)
                and incident_values[-1] - incident_values[0] >= rule.threshold
            )
            observation = (
                f"{metric} progressed from {incident_values[0]:.2f} "
                f"to {incident_values[-1]:.2f} {rule.unit}."
            )
        else:
            raise TelemetryAgentError(f"Unsupported symptom operator: {rule.operator}")

        if not observation:
            observation = f"Unable to compare {metric} with its baseline."
        return matched, observed, baseline, observation

    def _hardware_overrides(
        self,
        evidence: IncidentEvidencePackage,
    ) -> tuple[list[str], list[EvidenceItem]]:
        gpu_rows = evidence.gpu_incident_window
        reasons: list[str] = []
        items: list[EvidenceItem] = []
        timestamp = self._latest_timestamp(gpu_rows)

        uncorrectable = self._max(gpu_rows, "ecc_uncorrectable_errors") or 0.0
        xid = self._max(gpu_rows, "xid_error_code") or 0.0
        gpu_temperature = self._max(gpu_rows, "gpu_temperature_c") or 0.0
        memory_temperature = self._max(gpu_rows, "memory_temperature_c") or 0.0

        if uncorrectable > 0:
            reasons.append("uncorrectable_ecc_error")
            items.append(
                EvidenceItem(
                    metric="ecc_uncorrectable_errors",
                    observation="An uncorrectable GPU memory error was observed.",
                    value=int(uncorrectable),
                    unit="count",
                    source="gpu_telemetry",
                    timestamp=timestamp,
                )
            )
        if xid != 0:
            reasons.append("xid_error")
            items.append(
                EvidenceItem(
                    metric="xid_error_code",
                    observation=f"GPU XID error code {int(xid)} was observed.",
                    value=int(xid),
                    unit="code",
                    source="gpu_telemetry",
                    timestamp=timestamp,
                )
            )
        if gpu_temperature >= self.EXTREME_GPU_TEMPERATURE_C:
            reasons.append("extreme_gpu_temperature")
            items.append(
                EvidenceItem(
                    metric="gpu_temperature_c",
                    observation="GPU temperature reached the critical safety threshold.",
                    value=round(gpu_temperature, 3),
                    unit="celsius",
                    source="gpu_telemetry",
                    timestamp=timestamp,
                )
            )
        if memory_temperature >= self.EXTREME_MEMORY_TEMPERATURE_C:
            reasons.append("extreme_memory_temperature")
            items.append(
                EvidenceItem(
                    metric="memory_temperature_c",
                    observation="GPU memory temperature reached the critical threshold.",
                    value=round(memory_temperature, 3),
                    unit="celsius",
                    source="gpu_telemetry",
                    timestamp=timestamp,
                )
            )
        return reasons, items

    @staticmethod
    def _span_trace_id(span: Span) -> str | None:
        context = span.get_span_context()
        if not context.is_valid:
            return None
        return f"{context.trace_id:032x}"

    def analyze(self, evidence: IncidentEvidencePackage) -> TelemetryAgentResult:
        """Evaluate one incident package and return a validated agent result."""

        started_at = utc_now()
        started_clock = perf_counter()
        incident = evidence.incident

        with self._tracer.start_as_current_span(
            "agent.gpu_telemetry.analyze"
        ) as span:
            span.set_attribute("clustermind.agent", AgentName.GPU_TELEMETRY.value)
            span.set_attribute("clustermind.incident_id", incident.incident_id)
            span.set_attribute("clustermind.scenario", incident.incident_type)
            span.set_attribute("clustermind.node_id", incident.node_id)
            span.set_attribute("clustermind.gpu_id", incident.gpu_id)

            try:
                matched_weight = 0.0
                evaluated_weight = 0.0
                evidence_items: list[EvidenceItem] = []

                for rule in evidence.runbook.symptoms:
                    matched, value, baseline, observation = self._evaluate_rule(
                        rule,
                        evidence,
                    )
                    evaluated_weight += rule.weight
                    if matched:
                        matched_weight += rule.weight
                        evidence_items.append(
                            EvidenceItem(
                                metric=rule.metric,
                                observation=observation,
                                value=None if value is None else round(value, 4),
                                baseline=(
                                    None if baseline is None else round(baseline, 4)
                                ),
                                unit=rule.unit,
                                source=(
                                    "workload_telemetry"
                                    if rule.metric
                                    in evidence.workload_incident_window[0]
                                    else "gpu_telemetry"
                                ),
                                timestamp=self._latest_timestamp(
                                    evidence.workload_incident_window
                                    if rule.metric
                                    in evidence.workload_incident_window[0]
                                    else evidence.gpu_incident_window
                                ),
                            )
                        )

                if evaluated_weight <= 0.0:
                    raise TelemetryAgentError("No weighted symptom rules were evaluated")

                detector_scores = self._numeric_values(
                    evidence.detector_incident_window,
                    "anomaly_score",
                )
                if not detector_scores:
                    raise TelemetryAgentError(
                        "Detector evidence does not contain anomaly scores"
                    )
                anomaly_score = max(detector_scores)
                predicted_rows = sum(
                    bool(row.get("predicted_anomaly"))
                    for row in evidence.detector_incident_window
                )
                detector_rate = predicted_rows / len(
                    evidence.detector_incident_window
                )

                override_reasons, override_evidence = self._hardware_overrides(
                    evidence
                )
                evidence_items.extend(override_evidence)
                hardware_override = bool(override_reasons)

                rule_score = matched_weight / evaluated_weight
                confidence = min(
                    0.98,
                    (0.58 * rule_score)
                    + (0.27 * anomaly_score)
                    + (0.15 * detector_rate),
                )
                if hardware_override:
                    confidence = max(
                        confidence,
                        self.HARDWARE_OVERRIDE_CONFIDENCE,
                    )

                minimum_confidence = (
                    evidence.runbook.severity_policy.minimum_agent_confidence
                )
                confidence = max(confidence, min(rule_score, minimum_confidence))

                if not evidence_items:
                    raise TelemetryAgentError(
                        "Incident did not match any runbook symptom or hardware override"
                    )

                severity = IncidentSeverity(incident.severity)
                if hardware_override:
                    severity = IncidentSeverity.CRITICAL

                summary = (
                    f"{evidence.runbook.title} evidence was detected on "
                    f"{incident.node_id}/{incident.gpu_id}. "
                    f"{len(evidence_items)} evidence signals matched; "
                    f"the maximum detector anomaly score was {anomaly_score:.3f}."
                )
                if hardware_override:
                    summary += (
                        " A deterministic hardware safety override raised the "
                        "condition to critical severity."
                    )

                completed_at = utc_now()
                execution = AgentExecution(
                    agent=AgentName.GPU_TELEMETRY,
                    status=AgentStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=round(
                        (perf_counter() - started_clock) * 1000.0,
                        3,
                    ),
                    trace_id=self._span_trace_id(span),
                )
                result = TelemetryAgentResult(
                    execution=execution,
                    incident_id=incident.incident_id,
                    node_id=incident.node_id,
                    gpu_id=incident.gpu_id,
                    condition=incident.incident_type,
                    severity=severity,
                    anomaly_score=round(anomaly_score, 6),
                    confidence=round(confidence, 6),
                    hardware_override=hardware_override,
                    override_reasons=override_reasons,
                    evidence=evidence_items,
                    summary=summary,
                )

                span.set_attribute("clustermind.confidence", result.confidence)
                span.set_attribute(
                    "clustermind.anomaly_score",
                    result.anomaly_score,
                )
                span.set_attribute(
                    "clustermind.hardware_override",
                    result.hardware_override,
                )
                span.set_attribute(
                    "clustermind.evidence_count",
                    len(result.evidence),
                )
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                if isinstance(exc, TelemetryAgentError):
                    raise
                raise TelemetryAgentError(
                    f"Telemetry analysis failed for {incident.incident_id}"
                ) from exc


def current_trace_id() -> str | None:
    """Return the current valid trace ID for API and workflow correlation."""

    span = get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"
