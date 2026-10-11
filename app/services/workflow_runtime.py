from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import BoundedSemaphore, Lock, RLock

from app.agents.workflow import (
    ClusterMindWorkflow,
    WorkflowConflictError,
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowStoreProtocol,
)
from app.db.repositories import (
    PostgresWorkflowRepository,
    WorkflowRepositoryConflictError,
    WorkflowRepositoryError,
    WorkflowRepositoryNotFoundError,
)
from app.schemas.agents import ApprovalDecision, WorkflowResult
from app.services.incidents import IncidentRepository, get_incident_repository


class WorkflowRuntime:
    """Thread-safe single-instance runtime around the LangGraph workflow."""

    def __init__(
        self,
        repository: IncidentRepository | None = None,
        *,
        store_directory: Path | None = None,
        workflow_repository: WorkflowStoreProtocol | None = None,
        max_concurrent_analyses: int = 1,
    ) -> None:
        if isinstance(max_concurrent_analyses, bool) or max_concurrent_analyses < 1:
            raise ValueError("max_concurrent_analyses must be a positive integer")
        self.repository = repository or get_incident_repository()
        if workflow_repository is not None and store_directory is not None:
            raise ValueError(
                "Provide workflow_repository or store_directory, not both"
            )
        self.workflow = ClusterMindWorkflow(
            repository=self.repository,
            store=workflow_repository,
            store_directory=store_directory,
        )
        self.workflow_store_detail = (
            "postgresql"
            if isinstance(self.workflow.store, PostgresWorkflowRepository)
            else str(self.workflow.store.directory)
        )
        self._analysis_slots = BoundedSemaphore(max_concurrent_analyses)
        self._decision_lock = RLock()
        self._locks_guard = Lock()
        self._incident_locks: defaultdict[str, RLock] = defaultdict(RLock)

    def _incident_lock(self, incident_id: str) -> RLock:
        with self._locks_guard:
            return self._incident_locks[incident_id]

    def analyze(
        self,
        incident_id: str,
        *,
        use_llm: bool,
        force: bool,
    ) -> WorkflowResult:
        normalized = incident_id.strip()
        with self._incident_lock(normalized):
            with self._analysis_slots:
                return self.workflow.analyze(
                    normalized,
                    use_llm=use_llm,
                    force=force,
                )

    def decide(
        self,
        workflow_id: str,
        *,
        decision: ApprovalDecision,
        actor: str,
        reason: str | None,
    ) -> WorkflowResult:
        with self._decision_lock:
            if isinstance(self.workflow.store, PostgresWorkflowRepository):
                try:
                    return self.workflow.store.decide(
                        workflow_id,
                        decision=decision,
                        actor=actor,
                        reason=reason,
                    )
                except WorkflowRepositoryNotFoundError as exc:
                    raise WorkflowNotFoundError(str(exc)) from exc
                except WorkflowRepositoryConflictError as exc:
                    raise WorkflowConflictError(str(exc)) from exc
                except WorkflowRepositoryError as exc:
                    raise WorkflowError(str(exc)) from exc
            return self.workflow.decide(
                workflow_id,
                decision=decision,
                actor=actor,
                reason=reason,
            )

    def close(self) -> None:
        self.workflow.close()
