from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import health as health_module
from app.api.v1 import product as product_module
from app.api.v1.product import get_runtime
from app.core.config import Settings
from app.main import create_app
from app.schemas.health import ComponentHealth
from app.services.analysis_dispatcher import (
    AnalysisDispatch,
    AnalysisDispatchError,
    AnalysisDispatcher,
)
from app.services.incidents import IncidentRepositoryError


class FakeIncidentRepository:
    def get_incident(self, incident_id: str):
        if incident_id == "INC-NOT-FOUND":
            raise IncidentRepositoryError(
                "Unknown incident_id: INC-NOT-FOUND"
            )
        return SimpleNamespace(incident_id=incident_id)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(product_module, "DatabaseRuntime", SimpleNamespace)
    
    settings = Settings(
        workflow_repository_backend="postgres",
        otel_enabled=False,
    )
    application = create_app(settings)

    @asynccontextmanager
    async def empty_lifespan(_):
        application.state.database_runtime = SimpleNamespace()
        application.state.workflow_runtime = SimpleNamespace(
            repository=FakeIncidentRepository(),
            workflow_store_detail="postgresql",
        )
        yield

    application.router.lifespan_context = empty_lifespan
    application.dependency_overrides[get_runtime] = lambda: (
        application.state.workflow_runtime
    )
    return application


def test_analysis_endpoint_returns_202(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AnalysisDispatcher,
        "dispatch",
        lambda self, incident_id, use_llm: AnalysisDispatch(
            workflow_id="WF-1",
            task_id="task-1",
            status="queued",
            dispatch_status="dispatched",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/incidents/INC-1/analyze",
            json={"use_llm": False, "force": False},
        )

    assert response.status_code == 202
    assert response.json() == {
        "workflow_id": "WF-1",
        "task_id": "task-1",
        "status": "queued",
        "dispatch_status": "dispatched",
        "status_url": "/api/v1/workflows/WF-1",
    }


def test_analysis_endpoint_rejects_force(app) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/incidents/INC-1/analyze",
            json={"use_llm": False, "force": True},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Forced asynchronous reanalysis is not enabled"
    )


def test_analysis_endpoint_returns_404_for_unknown_incident(app) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/incidents/INC-NOT-FOUND/analyze",
            json={"use_llm": False, "force": False},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Unknown incident_id: INC-NOT-FOUND"
    )


def test_analysis_endpoint_returns_503_on_dispatch_failure(
    app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dispatch(self, incident_id: str, use_llm: bool):
        raise AnalysisDispatchError("broker unavailable")

    monkeypatch.setattr(AnalysisDispatcher, "dispatch", fail_dispatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/incidents/INC-1/analyze",
            json={"use_llm": False, "force": False},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Incident analysis could not be queued safely"
    )


@pytest.mark.asyncio
async def test_worker_health_reports_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = SimpleNamespace(
        ping=lambda: {"worker-1": {"ok": "pong"}}
    )
    monkeypatch.setattr(
        health_module.celery_app.control,
        "inspect",
        lambda timeout: inspector,
    )

    result = await health_module._celery_worker_health()

    assert result.status == "healthy"
    assert result.message == "1 worker(s) available"


@pytest.mark.asyncio
async def test_worker_health_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = SimpleNamespace(ping=lambda: {})
    monkeypatch.setattr(
        health_module.celery_app.control,
        "inspect",
        lambda timeout: inspector,
    )

    result = await health_module._celery_worker_health()

    assert result.status == "unavailable"
    assert result.message == "No Celery worker responded"


def test_worker_unavailable_makes_readiness_false() -> None:
    components = [
        ComponentHealth(name="dataset", status="healthy"),
        ComponentHealth(name="detector", status="healthy"),
        ComponentHealth(name="postgresql", status="healthy"),
        ComponentHealth(name="celery_worker", status="unavailable"),
    ]

    status, ready = health_module._overall_readiness(components)

    assert status == "unavailable"
    assert ready is False
