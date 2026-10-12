from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest

from app.services.analysis_dispatcher import AnalysisDispatchError, AnalysisDispatcher
from app.services.outbox import OutboxMessage, OutboxPublisher, PublishBatchResult


MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000001")


def _message() -> OutboxMessage:
    return OutboxMessage(
        id=MESSAGE_ID,
        aggregate_id="WF-1",
        event_type="workflow.updated",
        payload={"status": "queued"},
        attempt_count=0,
    )


def test_outbox_success_marks_message_published() -> None:
    message = _message()
    calls: list[tuple[object, str]] = []
    repository = SimpleNamespace(
        claim_batch=lambda **_: (message,),
        mark_published=lambda message_id, worker_id: calls.append(
            (message_id, worker_id)
        ),
        mark_failed=lambda *args, **kwargs: pytest.fail(
            "mark_failed must not be called"
        ),
    )
    publisher = SimpleNamespace(
        settings=SimpleNamespace(ntfy_enabled=True),
        publish=lambda item: None,
    )

    result = OutboxPublisher(repository, publisher).publish_batch(
        worker_id="worker-1",
        limit=1,
    )

    assert result == PublishBatchResult(claimed=1, published=1, failed=0)
    assert calls == [(MESSAGE_ID, "worker-1")]


def test_outbox_failure_records_retryable_error() -> None:
    message = _message()
    failures: list[dict[str, object]] = []
    repository = SimpleNamespace(
        claim_batch=lambda **_: (message,),
        mark_published=lambda *args, **kwargs: pytest.fail(
            "mark_published must not be called"
        ),
        mark_failed=lambda message_id, **kwargs: failures.append(
            {"message_id": message_id, **kwargs}
        ),
    )

    def fail_publish(item: OutboxMessage) -> None:
        raise RuntimeError("ntfy unavailable")

    publisher = SimpleNamespace(
        settings=SimpleNamespace(ntfy_enabled=True),
        publish=fail_publish,
    )

    result = OutboxPublisher(repository, publisher).publish_batch(
        worker_id="worker-1",
        limit=1,
    )

    assert result == PublishBatchResult(claimed=1, published=0, failed=1)
    assert failures[0]["message_id"] == MESSAGE_ID
    assert failures[0]["worker_id"] == "worker-1"
    assert "RuntimeError: ntfy unavailable" in str(failures[0]["error"])


def test_dispatch_broker_failure_is_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow_id = "WF-INC-1-455BE5409C0A"
    queued = SimpleNamespace(
        workflow_id=workflow_id,
        incident_id="INC-1",
        status="queued",
        dispatch_status="pending",
        task_id=None,
    )
    marked = SimpleNamespace(
        workflow_id=workflow_id,
        incident_id="INC-1",
        status="queued",
        dispatch_status="dispatched",
        task_id="task-1",
    )
    failures: list[tuple[str, str]] = []
    lifecycle = SimpleNamespace(
        create_queued=lambda **_: queued,
        mark_dispatched=lambda *args, **kwargs: marked,
        mark_dispatch_failed=lambda item, error: failures.append((item, error)),
    )

    monkeypatch.setattr(
        "app.services.analysis_dispatcher.uuid4",
        lambda: "task-1",
    )

    def fail_send(*args, **kwargs):
        raise OSError("broker unavailable")

    monkeypatch.setattr(
        "app.services.analysis_dispatcher.celery_app.send_task",
        fail_send,
    )

    with pytest.raises(AnalysisDispatchError, match="Unable to dispatch"):
        AnalysisDispatcher(lifecycle).dispatch("INC-1", use_llm=False)

    assert failures == [(workflow_id, "OSError: broker unavailable")]


def test_pending_redispatch_uses_persisted_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        workflow_id="WF-1",
        incident_id="INC-1",
        status="queued",
        dispatch_status="pending",
        task_id=None,
        requested_use_llm=False,
    )
    marked = SimpleNamespace(
        status="queued",
        dispatch_status="dispatched",
        requested_use_llm=False,
    )
    sent: list[dict[str, object]] = []
    lifecycle = SimpleNamespace(
        get_status=lambda workflow_id: state,
        mark_dispatched=lambda *args, **kwargs: marked,
    )

    monkeypatch.setattr(
        "app.services.analysis_dispatcher.uuid4",
        lambda: "task-retry",
    )
    monkeypatch.setattr(
        "app.services.analysis_dispatcher.celery_app.send_task",
        lambda *args, **kwargs: sent.append({"args": args, **kwargs}),
    )

    result = AnalysisDispatcher(lifecycle).dispatch_pending("WF-1")

    assert result.task_id == "task-retry"
    assert result.dispatch_status == "dispatched"
    assert sent[0]["kwargs"]["use_llm"] is False
    assert sent[0]["task_id"] == "task-retry"
