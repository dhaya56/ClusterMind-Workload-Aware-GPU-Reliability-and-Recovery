from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.db.repositories import WorkflowRepositoryConflictError
from app.tasks import recovery_tasks, workflow_tasks


class FakeRuntime:
    disposed = False

    def __init__(self, settings) -> None:
        self.settings = settings

    def dispose(self) -> None:
        self.disposed = True


def task_request(task_id: str = "task-1", hostname: str = "worker-1"):
    return SimpleNamespace(request=SimpleNamespace(id=task_id, hostname=hostname))


def test_duplicate_delivery_returns_existing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(SimpleNamespace())
    lifecycle = SimpleNamespace(
        claim=lambda *args, **kwargs: (_ for _ in ()).throw(
            WorkflowRepositoryConflictError("already claimed")
        )
    )
    repository = SimpleNamespace(
        get=lambda workflow_id: SimpleNamespace(status="awaiting_approval")
    )

    monkeypatch.setattr(workflow_tasks, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(workflow_tasks, "DatabaseRuntime", lambda settings: runtime)
    monkeypatch.setattr(workflow_tasks, "WorkflowLifecycleRepository", lambda value: lifecycle)
    monkeypatch.setattr(workflow_tasks, "PostgresWorkflowRepository", lambda value: repository)

    result = workflow_tasks.analyze_workflow.run(
        workflow_id="WF-1",
        incident_id="INC-1",
        use_llm=False,
    )

    assert result == {
        "workflow_id": "WF-1",
        "status": "awaiting_approval",
        "duplicate_delivery": True,
    }
    assert runtime.disposed is True


def test_worker_failure_is_persisted_and_runtime_disposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(SimpleNamespace())
    failures: list[dict[str, object]] = []
    lifecycle = SimpleNamespace(
        claim=lambda *args, **kwargs: None,
        heartbeat=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("heartbeat failed")
        ),
        mark_failed=lambda workflow_id, **kwargs: failures.append(
            {"workflow_id": workflow_id, **kwargs}
        ),
    )

    monkeypatch.setattr(workflow_tasks, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(workflow_tasks, "DatabaseRuntime", lambda settings: runtime)
    monkeypatch.setattr(workflow_tasks, "WorkflowLifecycleRepository", lambda value: lifecycle)
    monkeypatch.setattr(
        workflow_tasks,
        "PostgresWorkflowRepository",
        lambda value: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        workflow_tasks.analyze_workflow.run(
            workflow_id="WF-1",
            incident_id="INC-1",
            use_llm=False,
        )

    assert failures[0]["workflow_id"] == "WF-1"
    assert "RuntimeError: heartbeat failed" in str(failures[0]["error"])
    assert runtime.disposed is True


def test_successful_worker_saves_and_completes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime = FakeRuntime(SimpleNamespace())
    completed: list[dict[str, object]] = []
    saved: list[object] = []
    result_object = SimpleNamespace(status="awaiting_approval")
    lifecycle = SimpleNamespace(
        claim=lambda *args, **kwargs: None,
        heartbeat=lambda *args, **kwargs: None,
        mark_completed=lambda workflow_id, **kwargs: completed.append(
            {"workflow_id": workflow_id, **kwargs}
        ),
    )
    repository = SimpleNamespace(save=lambda result: saved.append(result))
    workflow = SimpleNamespace(
        analyze=lambda *args, **kwargs: result_object,
        close=lambda: None,
    )

    monkeypatch.setattr(workflow_tasks, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(workflow_tasks, "DatabaseRuntime", lambda settings: runtime)
    monkeypatch.setattr(workflow_tasks, "WorkflowLifecycleRepository", lambda value: lifecycle)
    monkeypatch.setattr(workflow_tasks, "PostgresWorkflowRepository", lambda value: repository)
    monkeypatch.setattr(workflow_tasks, "get_incident_repository", lambda settings: SimpleNamespace())
    monkeypatch.setattr(workflow_tasks, "WorkflowStore", lambda path: SimpleNamespace())
    monkeypatch.setattr(workflow_tasks, "ClusterMindWorkflow", lambda **kwargs: workflow)

    result = workflow_tasks.analyze_workflow.run(
        workflow_id="WF-1",
        incident_id="INC-1",
        use_llm=False,
    )

    assert result["duplicate_delivery"] is False
    assert result["status"] == "awaiting_approval"
    assert saved == [result_object]
    assert completed[0]["workflow_id"] == "WF-1"
    assert runtime.disposed is True


def test_recovery_task_counts_partial_dispatch_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(celery_workflow_stale_seconds=300)
    runtime = FakeRuntime(settings)
    pending = [SimpleNamespace(workflow_id="WF-1"), SimpleNamespace(workflow_id="WF-2")]
    lifecycle = SimpleNamespace(
        recover_stale_claims=lambda **kwargs: (SimpleNamespace(),),
        list_pending=lambda **kwargs: pending,
    )

    class Dispatcher:
        def __init__(self, value) -> None:
            pass

        def dispatch_pending(self, workflow_id: str) -> None:
            if workflow_id == "WF-2":
                raise recovery_tasks.AnalysisDispatchError("broker unavailable")

    monkeypatch.setattr(recovery_tasks, "get_settings", lambda: settings)
    monkeypatch.setattr(recovery_tasks, "DatabaseRuntime", lambda value: runtime)
    monkeypatch.setattr(recovery_tasks, "WorkflowLifecycleRepository", lambda value: lifecycle)
    monkeypatch.setattr(recovery_tasks, "AnalysisDispatcher", Dispatcher)

    result = recovery_tasks.recover_workflows.run(limit=20)

    assert result == {"recovered": 1, "dispatched": 1, "failed": 1}
    assert runtime.disposed is True
