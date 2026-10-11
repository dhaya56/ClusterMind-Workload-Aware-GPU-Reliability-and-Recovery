from __future__ import annotations

from celery.signals import worker_process_init, worker_process_shutdown
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import get_settings

_worker_provider: TracerProvider | None = None


@worker_process_init.connect(weak=False)
def initialize_worker_telemetry(**_: object) -> None:
    global _worker_provider
    settings = get_settings()
    if settings.otel_enabled:
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.celery_worker_service_name,
                    "service.version": settings.app_version,
                    "deployment.environment": settings.app_environment,
                }
            )
        )
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=settings.otel_exporter_otlp_insecure,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _worker_provider = provider
    CeleryInstrumentor().instrument()


@worker_process_shutdown.connect(weak=False)
def shutdown_worker_telemetry(**_: object) -> None:
    global _worker_provider
    if _worker_provider is not None:
        _worker_provider.shutdown()
        _worker_provider = None
