from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.schemas.api import AnalysisDispatchResponse, WorkflowStatusResponse
from app.services.analysis_dispatcher import AnalysisDispatcher
from app.services.outbox import OutboxPublisher, PublishBatchResult
from app.tasks.celery_app import ANALYSIS_QUEUE, celery_app


def test_celery_uses_json_and_safe_delivery_configuration() -> None:
    config = celery_app.conf

    assert tuple(config.accept_content) == ("json",)
    assert config.task_serializer == "json"
    assert config.result_serializer == "json"
    assert config.task_acks_late is True
    assert config.task_reject_on_worker_lost is True
    assert config.worker_prefetch_multiplier == 1


def test_celery_periodic_schedules_are_routed() -> None:
    schedule = celery_app.conf.beat_schedule

    assert schedule["publish-outbox-events"]["task"] == "clustermind.publish_outbox"
    assert schedule["recover-stale-workflows"]["task"] == "clustermind.recover_workflows"
    assert schedule["publish-outbox-events"]["options"]["queue"] == ANALYSIS_QUEUE
    assert schedule["recover-stale-workflows"]["options"]["queue"] == ANALYSIS_QUEUE


def test_analysis_dispatch_response_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        AnalysisDispatchResponse(
            workflow_id="WF-1",
            task_id="task-1",
            status="queued",
            dispatch_status="dispatched",
            status_url="/api/v1/workflows/WF-1",
            unexpected=True,
        )


def test_transient_workflow_status_supports_missing_result() -> None:
    response = WorkflowStatusResponse(
        workflow_id="WF-1",
        incident_id="INC-1",
        status="queued",
        current_stage="created",
        dispatch_status="dispatched",
        task_id="task-1",
        retry_count=0,
        version=2,
        queued_at=datetime.now(timezone.utc),
        dispatched_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        result_available=False,
        result=None,
    )

    assert response.result_available is False
    assert response.result is None


def test_dispatcher_returns_existing_active_delivery() -> None:
    state = SimpleNamespace(
        workflow_id="WF-INC-1-455BE5409C0A",
        incident_id="INC-1",
        status="queued",
        dispatch_status="dispatched",
        task_id="task-existing",
    )
    lifecycle = SimpleNamespace(create_queued=lambda **_: state)

    result = AnalysisDispatcher(lifecycle).dispatch("INC-1", use_llm=False)

    assert result.workflow_id == "WF-INC-1-455BE5409C0A"
    assert result.task_id == "task-existing"
    assert result.dispatch_status == "dispatched"


def test_outbox_publisher_does_no_work_when_notifications_disabled() -> None:
    repository = SimpleNamespace()
    publisher = SimpleNamespace(settings=SimpleNamespace(ntfy_enabled=False))

    result = OutboxPublisher(repository, publisher).publish_batch(
        worker_id="worker-1",
        limit=20,
    )

    assert result == PublishBatchResult(claimed=0, published=0, failed=0)
