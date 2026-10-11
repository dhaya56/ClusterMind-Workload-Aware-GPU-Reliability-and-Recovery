from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import func, select

from app.db.models import OutboxEvent, WorkflowRun
from app.db.session import DatabaseRuntime
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import Match


HTTP_REQUESTS_TOTAL = Counter(
    "clustermind_http_requests_total",
    "Total HTTP requests handled by ClusterMind.",
    labelnames=("method", "route", "status_code"),
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "clustermind_http_request_duration_seconds",
    "ClusterMind HTTP request duration in seconds.",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "clustermind_http_requests_in_progress",
    "Current number of ClusterMind HTTP requests in progress.",
    labelnames=("method", "route"),
)

APPLICATION_INFO = Gauge(
    "clustermind_application_info",
    "Static ClusterMind application information.",
    labelnames=("version", "environment"),
)


WORKFLOW_LIFECYCLE_TOTAL = Gauge(
    "clustermind_workflow_lifecycle_total",
    "Current workflows grouped by workflow and dispatch status.",
    labelnames=("status", "dispatch_status"),
)

WORKFLOW_STALE_CLAIMS_TOTAL = Gauge(
    "clustermind_workflow_stale_claims_total",
    "Current claimed workflows with an expired heartbeat.",
)

OUTBOX_PENDING_TOTAL = Gauge(
    "clustermind_outbox_pending_total",
    "Current unpublished outbox events.",
)


def _refresh_database_metrics(app: FastAPI) -> None:
    runtime = getattr(app.state, "database_runtime", None)
    if not isinstance(runtime, DatabaseRuntime):
        return
    with runtime.read_session() as session:
        rows = session.execute(
            select(
                WorkflowRun.status,
                WorkflowRun.dispatch_status,
                func.count(WorkflowRun.id),
            ).group_by(WorkflowRun.status, WorkflowRun.dispatch_status)
        ).all()
        WORKFLOW_LIFECYCLE_TOTAL.clear()
        for status_value, dispatch_status, count in rows:
            WORKFLOW_LIFECYCLE_TOTAL.labels(
                status=status_value,
                dispatch_status=dispatch_status,
            ).set(count)
        stale = session.scalar(
            select(func.count(WorkflowRun.id)).where(
                WorkflowRun.dispatch_status == "claimed",
                WorkflowRun.last_heartbeat_at
                < func.timezone("utc", func.now()) - func.make_interval(0, 0, 0, 0, 0, 5),
            )
        )
        pending = session.scalar(
            select(func.count(OutboxEvent.id)).where(
                OutboxEvent.published_at.is_(None)
            )
        )
        WORKFLOW_STALE_CLAIMS_TOTAL.set(int(stale or 0))
        OUTBOX_PENDING_TOTAL.set(int(pending or 0))


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record bounded-label HTTP metrics for FastAPI routes."""

    def __init__(self, app: FastAPI, *, metrics_path: str) -> None:
        super().__init__(app)
        self.metrics_path = metrics_path

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path == self.metrics_path:
            return await call_next(request)

        method = request.method
        route = _route_template(request)
        started_at = perf_counter()

        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, route=route).inc()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = max(0.0, perf_counter() - started_at)
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, route=route).dec()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                route=route,
            ).observe(elapsed)
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                route=route,
                status_code=str(status_code),
            ).inc()


def _route_template(request: Request) -> str:
    """Resolve the matched route template without using raw URL labels."""

    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                return path
    return "unmatched"


def configure_metrics(
    app: FastAPI,
    *,
    metrics_path: str,
    version: str,
    environment: str,
) -> None:
    """Attach Prometheus middleware and expose the scrape endpoint."""

    normalized_path = metrics_path if metrics_path.startswith("/") else f"/{metrics_path}"

    APPLICATION_INFO.labels(
        version=version,
        environment=environment,
    ).set(1)

    app.add_middleware(
        PrometheusMiddleware,
        metrics_path=normalized_path,
    )

    @app.get(
        normalized_path,
        include_in_schema=False,
        name="prometheus_metrics",
    )
    async def prometheus_metrics() -> Response:
        _refresh_database_metrics(app)
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
