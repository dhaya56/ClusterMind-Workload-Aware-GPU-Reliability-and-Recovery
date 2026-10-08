from __future__ import annotations

import json
from time import perf_counter
from typing import Literal

from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.observability.tracing import get_tracer
from app.schemas.agents import (
    AgentExecution,
    AgentName,
    AgentStatus,
    EvidenceItem,
    RootCauseCandidate,
    RootCauseResult,
    TelemetryAgentResult,
    WorkloadImpactResult,
    utc_now,
)
from app.services.incidents import IncidentEvidencePackage
from app.services.ollama import (
    OllamaGeneration,
    OllamaResponseError,
    OllamaService,
    OllamaServiceError,
    OllamaUnavailableError,
)


class RootCauseAgentError(RuntimeError):
    """Raised when root-cause analysis cannot produce a safe diagnosis."""


class _LLMCandidate(BaseModel):
    """Schema-constrained diagnosis candidate returned by Ollama."""

    model_config = ConfigDict(extra="forbid")

    cause: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(min_length=1, max_length=8)
    contradicting_evidence: list[str] = Field(default_factory=list, max_length=8)


class _LLMRootCauseResponse(BaseModel):
    """Strict structured response accepted from the local reasoning model."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[_LLMCandidate] = Field(min_length=1, max_length=5)
    explanation: str = Field(min_length=1, max_length=1600)

    @model_validator(mode="after")
    def validate_unique_candidates(self) -> "_LLMRootCauseResponse":
        causes = [candidate.cause for candidate in self.candidates]
        if len(causes) != len(set(causes)):
            raise ValueError("candidate causes must be unique")
        return self


class RootCauseAgent:
    """Runbook-grounded root-cause agent with bounded Ollama reasoning."""

    LLM_WEIGHT = 0.55
    RUNBOOK_PRIOR_WEIGHT = 0.30
    UPSTREAM_EVIDENCE_WEIGHT = 0.15
    MAX_EVIDENCE_ITEMS = 10

    def __init__(self, ollama: OllamaService | None = None) -> None:
        self._ollama = ollama
        self._owns_ollama = ollama is None
        self._tracer = get_tracer("clustermind.agents.root_cause")

    def close(self) -> None:
        if self._owns_ollama and self._ollama is not None:
            self._ollama.close()
            self._ollama = None

    def __enter__(self) -> "RootCauseAgent":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _trace_id(span: Span) -> str | None:
        context = span.get_span_context()
        if not context.is_valid:
            return None
        return f"{context.trace_id:032x}"

    @staticmethod
    def _validate_inputs(
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
    ) -> None:
        incident_id = evidence.incident.incident_id
        if telemetry.incident_id != incident_id:
            raise RootCauseAgentError(
                "Telemetry result and evidence package incident IDs do not match"
            )
        if workload.incident_id != incident_id:
            raise RootCauseAgentError(
                "Workload result and evidence package incident IDs do not match"
            )
        if workload.healthy_productive_spike:
            raise RootCauseAgentError(
                "Root-cause diagnosis is not permitted for a healthy productive spike"
            )
        if telemetry.execution.status != AgentStatus.COMPLETED:
            raise RootCauseAgentError("Telemetry Agent did not complete successfully")
        if workload.execution.status != AgentStatus.COMPLETED:
            raise RootCauseAgentError(
                "Workload Impact Agent did not complete successfully"
            )

    @staticmethod
    def _evidence_text(items: list[EvidenceItem]) -> list[str]:
        output: list[str] = []
        for item in items:
            value_text = ""
            if item.value is not None:
                value_text = f" value={item.value}"
            if item.baseline is not None:
                value_text += f" baseline={item.baseline}"
            if item.unit:
                value_text += f" unit={item.unit}"
            output.append(
                f"[{item.source}] {item.metric}: {item.observation}{value_text}"
            )
        return output

    def _prompt_payload(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
    ) -> tuple[str, str]:
        runbook = evidence.runbook
        permitted = [
            {
                "cause": candidate.cause,
                "description": candidate.description,
                "confidence_prior": candidate.confidence_prior,
            }
            for candidate in runbook.probable_causes
        ]
        evidence_lines = self._evidence_text(
            (telemetry.evidence + workload.evidence)[: self.MAX_EVIDENCE_ITEMS]
        )
        payload = {
            "incident": {
                "incident_id": evidence.incident.incident_id,
                "scenario": evidence.incident.incident_type,
                "severity": telemetry.severity.value,
                "node_id": evidence.incident.node_id,
                "gpu_id": evidence.incident.gpu_id,
                "workload_id": evidence.incident.affected_workload,
            },
            "telemetry_assessment": {
                "condition": telemetry.condition,
                "confidence": telemetry.confidence,
                "anomaly_score": telemetry.anomaly_score,
                "hardware_override": telemetry.hardware_override,
                "override_reasons": telemetry.override_reasons,
            },
            "workload_assessment": {
                "harmful_impact": workload.harmful_impact,
                "impact_level": workload.impact_level,
                "confidence": workload.confidence,
                "throughput_change_percent": workload.throughput_change_percent,
                "latency_change_percent": workload.latency_change_percent,
                "error_rate_percent": workload.error_rate_percent,
                "checkpoint_available": workload.checkpoint_available,
            },
            "permitted_causes": permitted,
            "evidence": evidence_lines,
        }
        system_prompt = (
            "You are the ClusterMind Root Cause Agent for GPU operations. "
            "Use only the supplied evidence and only the permitted cause names. "
            "Do not invent metrics, causes, remediation steps, or missing facts. "
            "Rank every permitted cause that is materially plausible. "
            "Confidence must reflect evidence quality and uncertainty. "
            "Supporting and contradicting evidence must quote or closely paraphrase "
            "the supplied evidence. Return only data matching the required schema."
        )
        user_prompt = (
            "Diagnose the most probable root cause for this incident. "
            "The candidate cause field must exactly match one of the permitted cause "
            "names. Explain the conclusion in concise operator language.\n\n"
            + json.dumps(payload, indent=2, sort_keys=True)
        )
        return system_prompt, user_prompt

    @staticmethod
    def _upstream_evidence_score(
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
    ) -> float:
        workload_factor = workload.confidence
        if not workload.harmful_impact:
            workload_factor *= 0.75
        return min(
            1.0,
            max(0.0, (telemetry.confidence + workload_factor) / 2.0),
        )

    def _reconcile_candidates(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
        response: _LLMRootCauseResponse,
    ) -> list[RootCauseCandidate]:
        runbook_candidates = {
            item.cause: item for item in evidence.runbook.probable_causes
        }
        returned_causes = {item.cause for item in response.candidates}
        unknown = returned_causes.difference(runbook_candidates)
        if unknown:
            raise RootCauseAgentError(
                "Ollama returned causes outside the runbook allowlist: "
                + ", ".join(sorted(unknown))
            )

        evidence_score = self._upstream_evidence_score(telemetry, workload)
        reconciled: list[RootCauseCandidate] = []
        llm_by_cause = {item.cause: item for item in response.candidates}

        for cause, runbook_candidate in runbook_candidates.items():
            llm_candidate = llm_by_cause.get(cause)
            if llm_candidate is None:
                llm_confidence = 0.0
                supporting = [
                    "The model did not select this runbook candidate as materially plausible."
                ]
                contradicting = [
                    "No model-selected evidence supported this alternative."
                ]
            else:
                llm_confidence = llm_candidate.confidence
                supporting = list(llm_candidate.supporting_evidence)

                # Contradiction polarity cannot be trusted solely to the LLM.
                # Until contradiction evidence is deterministically validated
                # against runbook symptom rules, do not expose it to operators.
                contradicting = []

            confidence = min(
                0.99,
                max(
                    0.0,
                    (self.LLM_WEIGHT * llm_confidence)
                    + (
                        self.RUNBOOK_PRIOR_WEIGHT
                        * runbook_candidate.confidence_prior
                    )
                    + (self.UPSTREAM_EVIDENCE_WEIGHT * evidence_score),
                ),
            )
            reconciled.append(
                RootCauseCandidate(
                    cause=cause,
                    confidence=round(confidence, 6),
                    supporting_evidence=supporting,
                    contradicting_evidence=contradicting,
                )
            )

        reconciled.sort(key=lambda item: item.confidence, reverse=True)
        return reconciled

    @staticmethod
    def _fallback_candidates(
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
    ) -> list[RootCauseCandidate]:
        upstream = (telemetry.confidence + workload.confidence) / 2.0
        candidates: list[RootCauseCandidate] = []
        for index, item in enumerate(evidence.runbook.probable_causes):
            confidence = min(
                0.98,
                max(
                    0.0,
                    (0.75 * item.confidence_prior) + (0.25 * upstream),
                ),
            )
            supporting = [
                evidence_item.observation
                for evidence_item in (
                    telemetry.evidence + workload.evidence
                )[:6]
            ]
            contradicting = []
            if index > 0:
                contradicting.append(
                    "The primary runbook candidate has a stronger prior match."
                )
            candidates.append(
                RootCauseCandidate(
                    cause=item.cause,
                    confidence=round(confidence, 6),
                    supporting_evidence=supporting or [evidence.runbook.description],
                    contradicting_evidence=contradicting,
                )
            )
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates

    @staticmethod
    def _result_evidence(
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
        evidence: IncidentEvidencePackage,
    ) -> list[EvidenceItem]:
        merged = list(telemetry.evidence) + list(workload.evidence)
        merged.append(
            EvidenceItem(
                metric="runbook_match",
                observation=(
                    f"Validated runbook {evidence.runbook.runbook_id} matched "
                    f"scenario {evidence.incident.incident_type}."
                ),
                value=evidence.runbook.runbook_id,
                source="runbook",
            )
        )
        return merged[:25]

    def _get_ollama(self) -> OllamaService:
        if self._ollama is None:
            self._ollama = OllamaService()
        return self._ollama

    def analyze(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
        *,
        use_llm: bool = True,
    ) -> RootCauseResult:
        """Return an allowlisted diagnosis, with deterministic fallback."""

        self._validate_inputs(evidence, telemetry, workload)
        started_at = utc_now()
        started_clock = perf_counter()
        incident = evidence.incident

        with self._tracer.start_as_current_span(
            "agent.root_cause.analyze"
        ) as span:
            span.set_attribute("clustermind.agent", AgentName.ROOT_CAUSE.value)
            span.set_attribute("clustermind.incident_id", incident.incident_id)
            span.set_attribute("clustermind.runbook_id", evidence.runbook.runbook_id)
            span.set_attribute("clustermind.llm_requested", use_llm)

            used_llm = False
            fallback_used = False
            generation: OllamaGeneration | None = None
            explanation: str

            try:
                if use_llm:
                    try:
                        system_prompt, user_prompt = self._prompt_payload(
                            evidence,
                            telemetry,
                            workload,
                        )
                        response, generation = self._get_ollama().generate_structured(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            response_model=_LLMRootCauseResponse,
                        )
                        candidates = self._reconcile_candidates(
                            evidence,
                            telemetry,
                            workload,
                            response,
                        )
                        top_candidate = candidates[0]
                        supporting_summary = " ".join(
                            top_candidate.supporting_evidence[:3]
                        ).strip()

                        if not supporting_summary:
                            supporting_summary = (
                                "The selected cause has the strongest combined "
                                "runbook prior and upstream evidence score."
                            )

                        explanation = (
                            f"The most probable cause is "
                            f"{top_candidate.cause.replace('_', ' ')}. "
                            f"{supporting_summary}"
                        )
                        used_llm = True
                    except (
                        OllamaUnavailableError,
                        OllamaResponseError,
                        OllamaServiceError,
                        RootCauseAgentError,
                    ) as exc:
                        span.add_event(
                            "root_cause_llm_fallback",
                            {"reason": str(exc)[:1000]},
                        )
                        candidates = self._fallback_candidates(
                            evidence,
                            telemetry,
                            workload,
                        )
                        explanation = (
                            f"Deterministic runbook fallback selected "
                            f"{candidates[0].cause} because local model reasoning "
                            "was unavailable or failed validation. The diagnosis is "
                            "grounded in the validated telemetry, workload impact, "
                            "and runbook evidence."
                        )
                        fallback_used = True
                else:
                    candidates = self._fallback_candidates(
                        evidence,
                        telemetry,
                        workload,
                    )
                    explanation = (
                        f"Deterministic runbook analysis selected "
                        f"{candidates[0].cause} from validated incident evidence."
                    )
                    fallback_used = True

                if not candidates:
                    raise RootCauseAgentError(
                        "Root-cause analysis produced no candidates"
                    )

                top = candidates[0]
                completed_at = utc_now()
                result = RootCauseResult(
                    execution=AgentExecution(
                        agent=AgentName.ROOT_CAUSE,
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
                    probable_cause=top.cause,
                    runbook_id=evidence.runbook.runbook_id,
                    confidence=top.confidence,
                    candidates=candidates,
                    evidence=self._result_evidence(
                        telemetry,
                        workload,
                        evidence,
                    ),
                    explanation=explanation,
                    used_llm=used_llm,
                    fallback_used=fallback_used,
                )

                span.set_attribute("clustermind.used_llm", result.used_llm)
                span.set_attribute(
                    "clustermind.fallback_used",
                    result.fallback_used,
                )
                span.set_attribute(
                    "clustermind.probable_cause",
                    result.probable_cause,
                )
                span.set_attribute(
                    "clustermind.confidence",
                    result.confidence,
                )
                if generation is not None:
                    span.set_attribute(
                        "gen_ai.request.model",
                        generation.model,
                    )
                    span.set_attribute(
                        "gen_ai.usage.input_tokens",
                        generation.prompt_tokens or 0,
                    )
                    span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        generation.completion_tokens or 0,
                    )
                    span.set_attribute(
                        "clustermind.llm_attempts",
                        generation.attempts,
                    )
                    span.set_attribute(
                        "clustermind.llm_duration_ms",
                        generation.duration_ms,
                    )
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                if isinstance(exc, RootCauseAgentError):
                    raise
                raise RootCauseAgentError(
                    f"Root-cause analysis failed for {incident.incident_id}"
                ) from exc
