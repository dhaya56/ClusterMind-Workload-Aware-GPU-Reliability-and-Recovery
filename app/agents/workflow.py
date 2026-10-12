from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Protocol, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph import END, START, StateGraph
from opentelemetry.trace import Status, StatusCode

from app.agents.recovery import RecoveryAgent
from app.agents.root_cause import RootCauseAgent
from app.agents.telemetry import GPUTelemetryAgent
from app.agents.workload_impact import WorkloadImpactAgent
from app.observability.tracing import get_tracer
from app.schemas.agents import (
    ApprovalDecision,
    ApprovalRecord,
    RecoveryExecutionStatus,
    RecoveryPlanResult,
    RootCauseResult,
    TelemetryAgentResult,
    WorkflowResult,
    WorkloadImpactResult,
    utc_now,
)
from app.services.incidents import IncidentRepository, get_incident_repository


class WorkflowError(RuntimeError):
    """Raised when an incident workflow cannot progress safely."""


class WorkflowConflictError(WorkflowError):
    """Raised when a requested transition conflicts with persisted state."""


class WorkflowNotFoundError(WorkflowError):
    """Raised when a persisted workflow cannot be found."""


class AgentWorkflowState(TypedDict, total=False):
    """JSON-serializable state passed between LangGraph nodes."""

    workflow_id: str
    incident_id: str
    started_at: str
    updated_at: str
    use_llm: bool
    route: str
    status: str
    telemetry: dict[str, Any]
    workload_impact: dict[str, Any]
    root_cause: dict[str, Any]
    recovery_plan: dict[str, Any]
    errors: list[str]
    trace_id: str | None


class WorkflowStoreProtocol(Protocol):
    def save(self, result: WorkflowResult) -> None: ...
    def get(self, workflow_id: str) -> WorkflowResult: ...
    def list(self, *, limit: int = 100) -> tuple[WorkflowResult, ...]: ...


class WorkflowStore:
    """Atomic, process-local synchronized JSON persistence for workflow results."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @staticmethod
    def _safe_workflow_id(workflow_id: str) -> str:
        normalized = workflow_id.strip()
        if not normalized:
            raise WorkflowError("workflow_id cannot be blank")
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in normalized):
            raise WorkflowError("workflow_id contains unsupported characters")
        return normalized

    def _path(self, workflow_id: str) -> Path:
        return self.directory / f"{self._safe_workflow_id(workflow_id)}.json"

    def save(self, result: WorkflowResult) -> None:
        """Atomically replace one workflow document after schema validation."""

        validated = WorkflowResult.model_validate(result)
        destination = self._path(validated.workflow_id)
        temporary = destination.with_suffix(
            f".json.tmp-{os.getpid()}-{RLock().__hash__()}"
        )
        payload = validated.model_dump_json(indent=2)
        with self._lock:
            try:
                temporary.write_text(payload + "\n", encoding="utf-8")
                os.replace(temporary, destination)
            except OSError as exc:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise WorkflowError(
                    f"Unable to persist workflow {validated.workflow_id}"
                ) from exc

    def get(self, workflow_id: str) -> WorkflowResult:
        path = self._path(workflow_id)
        with self._lock:
            if not path.is_file():
                raise WorkflowNotFoundError(
                    f"Workflow does not exist: {workflow_id}"
                )
            try:
                return WorkflowResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise WorkflowError(
                    f"Persisted workflow is unreadable: {workflow_id}"
                ) from exc

    def list(self, *, limit: int = 100) -> tuple[WorkflowResult, ...]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise WorkflowError("limit must be between 1 and 1000")
        with self._lock:
            results: list[WorkflowResult] = []
            for path in self.directory.glob("WF-*.json"):
                try:
                    results.append(
                        WorkflowResult.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
                    )
                except (OSError, ValueError) as exc:
                    raise WorkflowError(
                        f"Persisted workflow is unreadable: {path.name}"
                    ) from exc
        results.sort(key=lambda item: item.updated_at, reverse=True)
        return tuple(results[:limit])


class ClusterMindWorkflow:
    """LangGraph orchestration for analysis, diagnosis, and approval gating."""

    EVIDENCE_CONFIDENCE_THRESHOLD = 0.55

    def __init__(
        self,
        *,
        repository: IncidentRepository | None = None,
        store: WorkflowStoreProtocol | None = None,
        store_directory: Path | None = None,
    ) -> None:
        self.repository = repository or get_incident_repository()
        if store is not None and store_directory is not None:
            raise ValueError("Provide store or store_directory, not both")
        default_directory = (
            self.repository.settings.artifact_directory / "workflows"
        )
        self.store = store or WorkflowStore(store_directory or default_directory)
        self._telemetry_agent = GPUTelemetryAgent()
        self._impact_agent = WorkloadImpactAgent()
        self._root_agent = RootCauseAgent()
        self._recovery_agent = RecoveryAgent()
        self._tracer = get_tracer("clustermind.workflow")
        self.graph = self._build_graph()

    def close(self) -> None:
        self._root_agent.close()
        self._recovery_agent.close()

    def __enter__(self) -> "ClusterMindWorkflow":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def workflow_id_for(incident_id: str) -> str:
        normalized = incident_id.strip()
        if not normalized:
            raise WorkflowError("incident_id cannot be blank")
        identifier = uuid5(
            NAMESPACE_URL,
            f"clustermind-workflow-v1:{normalized}",
        )
        return f"WF-{normalized}-{identifier.hex[:12].upper()}"

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise WorkflowError("workflow state contains a naive timestamp")
        return parsed

    def _build_graph(self):
        builder = StateGraph(AgentWorkflowState)
        builder.add_node("telemetry_agent", self._telemetry_node)
        builder.add_node("workload_impact_agent", self._workload_node)
        builder.add_node("root_cause_agent", self._root_cause_node)
        builder.add_node("recovery_agent", self._recovery_node)
        builder.add_node("suppressed", self._suppressed_node)
        builder.add_node("more_evidence", self._more_evidence_node)
        builder.add_node("awaiting_approval", self._awaiting_approval_node)

        builder.add_edge(START, "telemetry_agent")
        builder.add_edge("telemetry_agent", "workload_impact_agent")
        builder.add_conditional_edges(
            "workload_impact_agent",
            self._route_after_workload,
            {
                "suppressed": "suppressed",
                "more_evidence": "more_evidence",
                "diagnose": "root_cause_agent",
            },
        )
        builder.add_edge("root_cause_agent", "recovery_agent")
        builder.add_edge("recovery_agent", "awaiting_approval")
        builder.add_edge("suppressed", END)
        builder.add_edge("more_evidence", END)
        builder.add_edge("awaiting_approval", END)
        return builder.compile()

    def _telemetry_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        evidence = self.repository.build_evidence(state["incident_id"])
        result = self._telemetry_agent.analyze(evidence)
        return {
            "telemetry": result.model_dump(mode="json"),
            "updated_at": utc_now().isoformat(),
        }

    def _workload_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        evidence = self.repository.build_evidence(state["incident_id"])
        telemetry = TelemetryAgentResult.model_validate(state["telemetry"])
        result = self._impact_agent.analyze(evidence, telemetry)
        return {
            "workload_impact": result.model_dump(mode="json"),
            "updated_at": utc_now().isoformat(),
        }

    def _route_after_workload(
        self,
        state: AgentWorkflowState,
    ) -> Literal["suppressed", "more_evidence", "diagnose"]:
        telemetry = TelemetryAgentResult.model_validate(state["telemetry"])
        workload = WorkloadImpactResult.model_validate(state["workload_impact"])
        if workload.healthy_productive_spike:
            return "suppressed"
        if (
            not workload.harmful_impact
            and not telemetry.hardware_override
            and min(telemetry.confidence, workload.confidence)
            < self.EVIDENCE_CONFIDENCE_THRESHOLD
        ):
            return "more_evidence"
        return "diagnose"

    def _root_cause_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        evidence = self.repository.build_evidence(state["incident_id"])
        telemetry = TelemetryAgentResult.model_validate(state["telemetry"])
        workload = WorkloadImpactResult.model_validate(state["workload_impact"])
        result = self._root_agent.analyze(
            evidence,
            telemetry,
            workload,
            use_llm=bool(state.get("use_llm", True)),
        )
        return {
            "root_cause": result.model_dump(mode="json"),
            "updated_at": utc_now().isoformat(),
        }

    def _recovery_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        evidence = self.repository.build_evidence(state["incident_id"])
        telemetry = TelemetryAgentResult.model_validate(state["telemetry"])
        workload = WorkloadImpactResult.model_validate(state["workload_impact"])
        root_cause = RootCauseResult.model_validate(state["root_cause"])
        result = self._recovery_agent.analyze(
            evidence,
            telemetry,
            workload,
            root_cause,
            use_llm=False,
        )
        return {
            "recovery_plan": result.model_dump(mode="json"),
            "updated_at": utc_now().isoformat(),
        }

    @staticmethod
    def _suppressed_node(state: AgentWorkflowState) -> AgentWorkflowState:
        return {
            "route": "suppressed",
            "status": "suppressed",
            "updated_at": utc_now().isoformat(),
        }

    @staticmethod
    def _more_evidence_node(state: AgentWorkflowState) -> AgentWorkflowState:
        return {
            "route": "more_evidence",
            "status": "more_evidence_requested",
            "updated_at": utc_now().isoformat(),
        }

    @staticmethod
    def _awaiting_approval_node(state: AgentWorkflowState) -> AgentWorkflowState:
        if "recovery_plan" not in state:
            raise WorkflowError("Approval gate requires a recovery plan")
        return {
            "route": "awaiting_approval",
            "status": "awaiting_approval",
            "updated_at": utc_now().isoformat(),
        }

    @staticmethod
    def _state_to_result(state: AgentWorkflowState) -> WorkflowResult:
        status = state.get("status", "failed")
        errors = list(state.get("errors", []))
        return WorkflowResult(
            workflow_id=state["workflow_id"],
            incident_id=state["incident_id"],
            started_at=ClusterMindWorkflow._parse_datetime(state["started_at"]),
            updated_at=ClusterMindWorkflow._parse_datetime(state["updated_at"]),
            status=status,
            telemetry=(
                TelemetryAgentResult.model_validate(state["telemetry"])
                if "telemetry" in state
                else None
            ),
            workload_impact=(
                WorkloadImpactResult.model_validate(state["workload_impact"])
                if "workload_impact" in state
                else None
            ),
            root_cause=(
                RootCauseResult.model_validate(state["root_cause"])
                if "root_cause" in state
                else None
            ),
            recovery_plan=(
                RecoveryPlanResult.model_validate(state["recovery_plan"])
                if "recovery_plan" in state
                else None
            ),
            approval=None,
            execution_status=RecoveryExecutionStatus.NOT_STARTED,
            trace_id=state.get("trace_id"),
            errors=errors,
            metadata={
                "route": state.get("route"),
                "use_llm": bool(state.get("use_llm", True)),
                "orchestrator": "langgraph",
                "workflow_schema_version": "1.0",
            },
        )

    def analyze(
        self,
        incident_id: str,
        *,
        use_llm: bool = True,
        force: bool = False,
    ) -> WorkflowResult:
        """Run analysis to a terminal route or the human approval gate."""

        incident = self.repository.get_incident(incident_id)
        workflow_id = self.workflow_id_for(incident.incident_id)
        try:
            existing = self.store.get(workflow_id)
        except WorkflowNotFoundError:
            existing = None

        if existing is not None:
            immutable_statuses = {
                "approved",
                "rejected",
                "completed",
            }

            if existing.status in immutable_statuses:
                if force:
                    raise WorkflowConflictError(
                        f"Finalized workflow cannot be reanalyzed: "
                        f"{existing.status}"
                    )
                return existing

            reusable_statuses = {
                "awaiting_approval",
                "suppressed",
                "more_evidence_requested",
            }

            if not force and existing.status in reusable_statuses:
                return existing

        started_at = utc_now()
        initial: AgentWorkflowState = {
            "workflow_id": workflow_id,
            "incident_id": incident.incident_id,
            "started_at": started_at.isoformat(),
            "updated_at": started_at.isoformat(),
            "use_llm": use_llm,
            "route": "running",
            "status": "running",
            "errors": [],
        }

        with self._tracer.start_as_current_span(
            "workflow.incident_analysis"
        ) as span:
            span.set_attribute("clustermind.workflow_id", workflow_id)
            span.set_attribute("clustermind.incident_id", incident.incident_id)
            span.set_attribute("clustermind.use_llm", use_llm)
            context = span.get_span_context()
            if context.is_valid:
                initial["trace_id"] = f"{context.trace_id:032x}"
            try:
                final_state = self.graph.invoke(initial)
                result = self._state_to_result(final_state)
                self.store.save(result)
                span.set_attribute("clustermind.workflow_status", result.status)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                failed_state = dict(initial)
                failed_state.update(
                    {
                        "updated_at": utc_now().isoformat(),
                        "status": "failed",
                        "route": "failed",
                        "errors": [f"{type(exc).__name__}: {exc}"[:2000]],
                    }
                )
                result = self._state_to_result(failed_state)
                self.store.save(result)
                raise WorkflowError(
                    f"Workflow failed for incident {incident.incident_id}"
                ) from exc

    def decide(
        self,
        workflow_id: str,
        *,
        decision: ApprovalDecision | str,
        actor: str,
        reason: str | None = None,
    ) -> WorkflowResult:
        """Apply an idempotent human decision without executing recovery actions."""

        current = self.store.get(workflow_id)
        requested = ApprovalDecision(decision)
        if requested == ApprovalDecision.PENDING:
            raise WorkflowConflictError("pending is not a valid submitted decision")
        if current.status not in {
            "awaiting_approval",
            "approved",
            "rejected",
            "more_evidence_requested",
        }:
            raise WorkflowConflictError(
                f"Workflow status does not allow a decision: {current.status}"
            )
        if current.recovery_plan is None:
            raise WorkflowConflictError("Workflow has no recovery recommendation")

        if current.approval is not None and current.approval.decision != ApprovalDecision.PENDING:
            if current.approval.decision == requested:
                return current
            raise WorkflowConflictError(
                "A different final approval decision has already been recorded"
            )

        approval = ApprovalRecord(
            incident_id=current.incident_id,
            recommendation_id=current.recovery_plan.recommendation_id,
            decision=requested,
            actor=actor,
            reason=reason,
            decided_at=utc_now(),
        )
        status_map = {
            ApprovalDecision.APPROVED: "approved",
            ApprovalDecision.REJECTED: "rejected",
            ApprovalDecision.MORE_EVIDENCE_REQUESTED: "more_evidence_requested",
        }
        execution_map = {
            ApprovalDecision.APPROVED: RecoveryExecutionStatus.QUEUED,
            ApprovalDecision.REJECTED: RecoveryExecutionStatus.CANCELLED,
            ApprovalDecision.MORE_EVIDENCE_REQUESTED: RecoveryExecutionStatus.NOT_STARTED,
        }
        updated = current.model_copy(
            update={
                "updated_at": utc_now(),
                "status": status_map[requested],
                "approval": approval,
                "execution_status": execution_map[requested],
            }
        )
        validated = WorkflowResult.model_validate(updated)
        self.store.save(validated)
        return validated
