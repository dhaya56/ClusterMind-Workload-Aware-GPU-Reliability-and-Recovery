from __future__ import annotations

import logging
from threading import Lock

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from app.core.config import Settings


LOGGER = logging.getLogger(__name__)

_CONFIGURE_LOCK = Lock()
_PROVIDER: TracerProvider | None = None
_GLOBAL_INSTRUMENTATION_ENABLED = False


def configure_tracing(app: FastAPI, settings: Settings) -> TracerProvider | None:
    """Configure OTLP tracing and instrument FastAPI, HTTPX, and Redis."""

    global _PROVIDER
    global _GLOBAL_INSTRUMENTATION_ENABLED

    if not settings.otel_enabled:
        LOGGER.info("OpenTelemetry tracing is disabled")
        return None

    with _CONFIGURE_LOCK:
        if _PROVIDER is None:
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: settings.otel_service_name,
                    ResourceAttributes.SERVICE_VERSION: settings.app_version,
                    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: (
                        settings.app_environment
                    ),
                    "clustermind.component": "api",
                }
            )
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(
                endpoint=settings.otel_exporter_otlp_endpoint,
                insecure=settings.otel_exporter_otlp_insecure,
                timeout=10,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _PROVIDER = provider

        if not _GLOBAL_INSTRUMENTATION_ENABLED:
            HTTPXClientInstrumentor().instrument(
                tracer_provider=_PROVIDER,
            )
            RedisInstrumentor().instrument(
                tracer_provider=_PROVIDER,
            )
            _GLOBAL_INSTRUMENTATION_ENABLED = True

        if not getattr(app.state, "otel_instrumented", False):
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=_PROVIDER,
                excluded_urls=(
                    f"{settings.prometheus_metrics_path},"
                    f"{settings.api_v1_prefix}/health"
                ),
            )
            app.state.otel_instrumented = True

    LOGGER.info(
        "OpenTelemetry tracing configured",
        extra={
            "service_name": settings.otel_service_name,
            "otlp_endpoint": settings.otel_exporter_otlp_endpoint,
        },
    )
    return _PROVIDER


def get_tracer(name: str = "clustermind") -> trace.Tracer:
    """Return a tracer for custom service and agent spans."""

    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the configured provider."""

    if _PROVIDER is None:
        return

    try:
        _PROVIDER.force_flush(timeout_millis=5000)
    except Exception:
        LOGGER.exception("Unable to flush OpenTelemetry spans")
