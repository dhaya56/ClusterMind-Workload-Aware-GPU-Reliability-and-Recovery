from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.observability.tracing import get_tracer
from app.schemas.agents import (
    AgentExecution,
    AgentName,
    AgentStatus,
    RecoveryPlanResult,
    RecoveryStep,
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


class RecoveryAgentError(RuntimeError):
    """Raised when a safe, human-approved recovery plan cannot be produced."""


class _LLMRecoveryResponse(BaseModel):
    """Schema-constrained operator wording returned by Ollama."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=1200)
    expected_outcome: str = Field(min_length=1, max_length=800)

    @model_validator(mode="after")
    def strip_values(self) -> "_LLMRecoveryResponse":
        self.title = self.title.strip()
        self.summary = self.summary.strip()
        self.expected_outcome = self.expected_outcome.strip()
        return self


class RecoveryAgent:
    """Create checkpoint-aware, allowlisted plans that require human approval."""

    MAX_LLM_ACTIONS = 12

    def __init__(self, ollama: OllamaService | None = None) -> None:
        self._ollama = ollama
        self._owns_ollama = ollama is None
        self._tracer = get_tracer("clustermind.agents.recovery")

    def close(self) -> None:
        if self._owns_ollama and self._ollama is not None:
            self._ollama.close()
            self._ollama = None

    def __enter__(self) -> "RecoveryAgent":
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
        root_cause: RootCauseResult,
    ) -> None:
        incident_id = evidence.incident.incident_id
        results = (telemetry, workload, root_cause)
        if any(result.incident_id != incident_id for result in results):
            raise RecoveryAgentError(
                "Recovery inputs do not reference the same incident"
            )
        if any(result.execution.status != AgentStatus.COMPLETED for result in results):
            raise RecoveryAgentError(
                "All upstream agents must complete before recovery planning"
            )
        if workload.healthy_productive_spike:
            raise RecoveryAgentError(
                "Recovery planning is forbidden for a healthy productive spike"
            )
        allowed_causes = {
            candidate.cause for candidate in evidence.runbook.probable_causes
        }
        if root_cause.probable_cause not in allowed_causes:
            raise RecoveryAgentError(
                "Root cause is not allowlisted by the selected runbook"
            )
        if root_cause.runbook_id != evidence.runbook.runbook_id:
            raise RecoveryAgentError(
                "Root-cause and evidence runbook IDs do not match"
            )

    @staticmethod
    def _resolve_target(
        target: str,
        evidence: IncidentEvidencePackage,
        workload: WorkloadImpactResult,
    ) -> str:
        mapping = {
            "affected_gpu": (
                f"{evidence.incident.node_id}/{evidence.incident.gpu_id}"
            ),
            "affected_node": evidence.incident.node_id,
            "affected_workload": workload.workload_id,
            "hardware_operations": "hardware-operations-team",
            "healthy_gpu": "operator-selected-healthy-gpu",
            "healthy_node": "operator-selected-healthy-node",
            "healthy_nvlink_domain": "operator-selected-healthy-nvlink-domain",
        }
        return mapping.get(target, target)

    def _materialize_steps(
        self,
        evidence: IncidentEvidencePackage,
        workload: WorkloadImpactResult,
    ) -> tuple[list[RecoveryStep], bool]:
        policy = evidence.runbook.recovery
        missing_required_checkpoint = (
            policy.checkpoint_required and not workload.checkpoint_available
        )
        steps: list[RecoveryStep] = []

        for template in policy.actions:
            if missing_required_checkpoint and template.action in {
                "migrate_workload",
                "restart_workload",
                "resume_from_checkpoint",
            }:
                continue
            if (
                template.action == "resume_from_checkpoint"
                and not workload.checkpoint_available
            ):
                continue
            steps.append(
                RecoveryStep(
                    order=len(steps) + 1,
                    action=template.action,
                    target=self._resolve_target(
                        template.target,
                        evidence,
                        workload,
                    ),
                    description=template.description,
                    reversible=template.reversible,
                    rollback=template.rollback,
                )
            )

        if missing_required_checkpoint:
            if not any(step.action == "monitor" for step in steps):
                steps.insert(
                    0,
                    RecoveryStep(
                        order=1,
                        action="monitor",
                        target=(
                            f"{evidence.incident.node_id}/"
                            f"{evidence.incident.gpu_id}"
                        ),
                        description=(
                            "Maintain enhanced monitoring while an operator determines "
                            "a safe workload preservation strategy."
                        ),
                        reversible=True,
                        rollback=(
                            "Return to standard monitoring after a safe recovery "
                            "strategy is approved."
                        ),
                    ),
                )

            if not any(step.action == "escalate_to_operator" for step in steps):
                steps.append(
                    RecoveryStep(
                        order=1,
                        action="escalate_to_operator",
                        target="cluster-operations-team",
                        description=(
                            "No verified checkpoint is available. Escalate for a "
                            "manual decision before migration, restart, or release."
                        ),
                        reversible=False,
                    )
                )

        normalized: list[RecoveryStep] = []
        for index, step in enumerate(steps, start=1):
            normalized.append(step.model_copy(update={"order": index}))
        if not normalized:
            raise RecoveryAgentError("Recovery policy produced no safe actions")
        return normalized, missing_required_checkpoint

    @staticmethod
    def _risk_rank(value: str) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}[value]

    def _effective_risk(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
        missing_required_checkpoint: bool,
    ) -> str:
        values = [evidence.runbook.recovery.risk_level]
        if telemetry.hardware_override:
            values.append("critical")
        if workload.impact_level == "critical":
            values.append("critical")
        elif workload.impact_level == "high":
            values.append("high")
        if missing_required_checkpoint:
            values.append("critical")
        return max(values, key=self._risk_rank)

    @staticmethod
    def _fallback_wording(
        evidence: IncidentEvidencePackage,
        workload: WorkloadImpactResult,
        root_cause: RootCauseResult,
        steps: list[RecoveryStep],
        missing_required_checkpoint: bool,
    ) -> tuple[str, str, str]:
        cause = root_cause.probable_cause.replace("_", " ")
        title = f"Human-approved recovery for {cause}"
        if missing_required_checkpoint:
            summary = (
                f"A recovery checkpoint is required by {evidence.runbook.title}, "
                "but no verified checkpoint is available. Unsafe migration, restart, "
                "and checkpoint-resume steps were removed. Continue safe monitoring "
                "and escalate to an operator for workload preservation."
            )
            outcome = (
                "The affected resource remains controlled while an operator selects "
                "a safe, data-preserving recovery strategy."
            )
        else:
            actions = ", ".join(step.action.replace("_", " ") for step in steps)
            summary = (
                f"Execute the validated {evidence.runbook.title} recovery sequence "
                f"for workload {workload.workload_id}: {actions}. All actions remain "
                "blocked until a human approves the recommendation."
            )
            outcome = (
                f"Restore stable execution for {workload.workload_id}, preserve the "
                "latest verified state, and isolate the affected resource when required."
            )
        return title, summary, outcome

    def _prompt_payload(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
        root_cause: RootCauseResult,
        steps: list[RecoveryStep],
        risk_level: str,
        missing_required_checkpoint: bool,
    ) -> tuple[str, str]:
        payload: dict[str, Any] = {
            "incident": {
                "incident_id": evidence.incident.incident_id,
                "node_id": evidence.incident.node_id,
                "gpu_id": evidence.incident.gpu_id,
                "workload_id": workload.workload_id,
                "severity": telemetry.severity.value,
                "impact_level": workload.impact_level,
            },
            "diagnosis": {
                "probable_cause": root_cause.probable_cause,
                "confidence": root_cause.confidence,
                "runbook_id": root_cause.runbook_id,
            },
            "recovery_constraints": {
                "human_approval_required": True,
                "checkpoint_required": evidence.runbook.recovery.checkpoint_required,
                "checkpoint_available": workload.checkpoint_available,
                "missing_required_checkpoint": missing_required_checkpoint,
                "risk_level": risk_level,
                "expected_downtime_minutes": (
                    evidence.runbook.recovery.expected_downtime_minutes
                ),
            },
            "approved_action_sequence": [
                {
                    "order": step.order,
                    "action": step.action,
                    "target": step.target,
                    "description": step.description,
                }
                for step in steps[: self.MAX_LLM_ACTIONS]
            ],
            "safety_checks": list(evidence.runbook.safety_checks),
            "rollback_plan": evidence.runbook.rollback_plan,
        }
        system_prompt = (
            "You are the ClusterMind Recovery Agent wording assistant. "
            "The application has already selected and validated the only permitted "
            "actions. Do not add, remove, reorder, or execute actions. Do not claim "
            "that recovery has already happened. Human approval is mandatory. If a "
            "required checkpoint is missing, clearly state that migration, restart, "
            "or checkpoint resume is blocked. Return only the requested structured data."
        )
        user_prompt = (
            "Write a concise operator-facing title, summary, and expected outcome for "
            "the validated recovery plan below. Do not invent facts or actions.\n\n"
            + json.dumps(payload, indent=2, sort_keys=True)
        )
        return system_prompt, user_prompt

    def _get_ollama(self) -> OllamaService:
        if self._ollama is None:
            self._ollama = OllamaService()
        return self._ollama

    def analyze(
        self,
        evidence: IncidentEvidencePackage,
        telemetry: TelemetryAgentResult,
        workload: WorkloadImpactResult,
        root_cause: RootCauseResult,
        *,
        use_llm: bool = True,
    ) -> RecoveryPlanResult:
        """Create a safe plan without executing infrastructure changes."""

        self._validate_inputs(evidence, telemetry, workload, root_cause)
        started_at = utc_now()
        started_clock = perf_counter()
        incident = evidence.incident

        with self._tracer.start_as_current_span(
            "agent.recovery.analyze"
        ) as span:
            span.set_attribute("clustermind.agent", AgentName.RECOVERY.value)
            span.set_attribute("clustermind.incident_id", incident.incident_id)
            span.set_attribute("clustermind.runbook_id", evidence.runbook.runbook_id)
            span.set_attribute("clustermind.llm_requested", use_llm)

            used_llm = False
            fallback_used = False
            generation: OllamaGeneration | None = None

            try:
                steps, missing_required_checkpoint = self._materialize_steps(
                    evidence,
                    workload,
                )
                risk_level = self._effective_risk(
                    evidence,
                    telemetry,
                    workload,
                    missing_required_checkpoint,
                )
                title, summary, expected_outcome = self._fallback_wording(
                    evidence,
                    workload,
                    root_cause,
                    steps,
                    missing_required_checkpoint,
                )

                # Recovery actions and operator wording remain deterministic.
                # A language model must not introduce, omit, or imply execution of
                # safety-critical recovery actions. The Root Cause Agent already
                # provides bounded LLM reasoning; recovery planning is policy-driven.
                fallback_used = True
                if use_llm:
                    span.add_event(
                        "recovery_llm_wording_suppressed",
                        {
                            "reason": (
                                "Safety-critical recovery wording is generated "
                                "deterministically from the validated runbook"
                            )
                        },
                    )

                action_fingerprint = ",".join(
                    f"{step.order}:{step.action}:{step.target}" for step in steps
                )
                recommendation_uuid = uuid5(
                    NAMESPACE_URL,
                    (
                        f"clustermind:{incident.incident_id}:"
                        f"{evidence.runbook.runbook_id}:"
                        f"{workload.checkpoint_available}:{action_fingerprint}"
                    ),
                )
                recommendation_id = (
                    f"REC-{incident.incident_id}-"
                    f"{recommendation_uuid.hex[:12].upper()}"
                )
                confidence = min(
                    0.99,
                    max(
                        0.0,
                        (0.50 * root_cause.confidence)
                        + (0.25 * workload.confidence)
                        + (0.25 * telemetry.confidence),
                    ),
                )
                if missing_required_checkpoint:
                    confidence = min(confidence, 0.85)

                completed_at = utc_now()
                result = RecoveryPlanResult(
                    execution=AgentExecution(
                        agent=AgentName.RECOVERY,
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
                    recommendation_id=recommendation_id,
                    title=title,
                    risk_level=risk_level,
                    expected_downtime_minutes=(
                        evidence.runbook.recovery.expected_downtime_minutes
                    ),
                    checkpoint_required=(
                        evidence.runbook.recovery.checkpoint_required
                    ),
                    checkpoint_available=workload.checkpoint_available,
                    approval_required=True,
                    confidence=round(confidence, 6),
                    steps=steps,
                    safety_checks=(
                        list(evidence.runbook.safety_checks)
                        + (
                            [
                                "Do not interrupt the running workload until an operator "
                                "accepts the data-loss risk caused by the missing checkpoint."
                            ]
                            if missing_required_checkpoint
                            else []
                        )
                    ),
                    expected_outcome=expected_outcome,
                    rollback_plan=evidence.runbook.rollback_plan,
                    summary=summary,
                    used_llm=used_llm,
                    fallback_used=fallback_used,
                )

                span.set_attribute("clustermind.used_llm", result.used_llm)
                span.set_attribute(
                    "clustermind.fallback_used",
                    result.fallback_used,
                )
                span.set_attribute(
                    "clustermind.risk_level",
                    result.risk_level,
                )
                span.set_attribute(
                    "clustermind.checkpoint_required",
                    result.checkpoint_required,
                )
                span.set_attribute(
                    "clustermind.checkpoint_available",
                    result.checkpoint_available,
                )
                span.set_attribute(
                    "clustermind.recovery_step_count",
                    len(result.steps),
                )
                span.set_attribute(
                    "clustermind.confidence",
                    result.confidence,
                )
                if generation is not None:
                    span.set_attribute("gen_ai.request.model", generation.model)
                    span.set_attribute(
                        "gen_ai.usage.input_tokens",
                        generation.prompt_tokens or 0,
                    )
                    span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        generation.completion_tokens or 0,
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
                if isinstance(exc, RecoveryAgentError):
                    raise
                raise RecoveryAgentError(
                    f"Recovery planning failed for {incident.incident_id}"
                ) from exc
