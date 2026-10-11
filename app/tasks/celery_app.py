from __future__ import annotations

from celery import Celery
from kombu import Exchange, Queue

from app.core.config import get_settings

settings = get_settings()

ANALYSIS_QUEUE = "clustermind.analysis"
analysis_exchange = Exchange(ANALYSIS_QUEUE, type="direct", durable=True)

celery_app = Celery(
    "clustermind",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=("app.tasks.probe_tasks", "app.tasks.workflow_tasks"),
)

celery_app.conf.update(
    accept_content=("json",),
    task_serializer="json",
    result_serializer="json",
    event_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue=ANALYSIS_QUEUE,
    task_default_exchange=ANALYSIS_QUEUE,
    task_default_exchange_type="direct",
    task_default_routing_key=ANALYSIS_QUEUE,
    task_queues=(
        Queue(
            ANALYSIS_QUEUE,
            exchange=analysis_exchange,
            routing_key=ANALYSIS_QUEUE,
            durable=True,
        ),
    ),
    task_routes={
        "clustermind.probe": {
            "queue": ANALYSIS_QUEUE,
            "routing_key": ANALYSIS_QUEUE,
        },
    },
    task_acks_late=True,
    task_acks_on_failure_or_timeout=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.celery_worker_concurrency,
    task_soft_time_limit=settings.celery_task_soft_time_limit_seconds,
    task_time_limit=settings.celery_task_time_limit_seconds,
    result_expires=settings.celery_result_expires_seconds,
    result_extended=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout_seconds,
    },
    result_backend_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout_seconds,
    },
    task_track_started=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    worker_enable_remote_control=True,
)
