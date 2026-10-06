from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx
from fastapi import APIRouter, Depends
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.schemas.health import ComponentHealth, HealthResponse, ReadinessResponse


router = APIRouter(tags=["health"])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000.0, 3)


def _check_required_files(
    name: str,
    paths: tuple[Path, ...],
) -> ComponentHealth:
    started_at = perf_counter()
    missing = [path.name for path in paths if not path.is_file()]

    if missing:
        return ComponentHealth(
            name=name,
            status="unavailable",
            message="Missing required files: " + ", ".join(sorted(missing)),
            latency_ms=_elapsed_ms(started_at),
        )

    empty = [path.name for path in paths if path.stat().st_size == 0]
    if empty:
        return ComponentHealth(
            name=name,
            status="unavailable",
            message="Empty required files: " + ", ".join(sorted(empty)),
            latency_ms=_elapsed_ms(started_at),
        )

    return ComponentHealth(
        name=name,
        status="healthy",
        message=f"{len(paths)} required files are available",
        latency_ms=_elapsed_ms(started_at),
    )


def _dataset_health(settings: Settings) -> ComponentHealth:
    names = (
        "gpu_telemetry.csv",
        "node_telemetry.csv",
        "workloads.csv",
        "incidents.csv",
        "agent_evaluation.csv",
        "generation_manifest.json",
    )
    paths = tuple(settings.data_directory / name for name in names)
    return _check_required_files("dataset", paths)


def _detector_health(settings: Settings) -> ComponentHealth:
    names = (
        "isolation_forest.joblib",
        "standard_scaler.joblib",
        "detector_metadata.json",
        "feature_order.json",
        "evaluation_metrics.json",
    )
    paths = tuple(settings.model_artifact_directory / name for name in names)
    return _check_required_files("detector", paths)


async def _http_health(
    *,
    name: str,
    url: str,
    timeout_seconds: float,
    healthy_statuses: set[int],
) -> ComponentHealth:
    started_at = perf_counter()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        ) as client:
            response = await client.get(url)
        if response.status_code not in healthy_statuses:
            return ComponentHealth(
                name=name,
                status="degraded",
                message=f"Unexpected HTTP status {response.status_code}",
                latency_ms=_elapsed_ms(started_at),
            )
        return ComponentHealth(
            name=name,
            status="healthy",
            message="Service responded successfully",
            latency_ms=_elapsed_ms(started_at),
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        return ComponentHealth(
            name=name,
            status="unavailable",
            message=f"Connection failed: {type(exc).__name__}",
            latency_ms=_elapsed_ms(started_at),
        )
    except httpx.HTTPError as exc:
        return ComponentHealth(
            name=name,
            status="degraded",
            message=f"HTTP check failed: {type(exc).__name__}",
            latency_ms=_elapsed_ms(started_at),
        )


async def _influxdb_health(settings: Settings) -> ComponentHealth:
    return await _http_health(
        name="influxdb",
        url=f"{settings.influxdb_url}/health",
        timeout_seconds=min(settings.influxdb_timeout_ms / 1000.0, 5.0),
        healthy_statuses={200},
    )


async def _ollama_health(settings: Settings) -> ComponentHealth:
    return await _http_health(
        name="ollama",
        url=f"{settings.ollama_base_url}/api/tags",
        timeout_seconds=5.0,
        healthy_statuses={200},
    )


async def _redis_health(settings: Settings) -> ComponentHealth:
    started_at = perf_counter()
    client: Redis | None = None
    try:
        client = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=3.0,
            socket_timeout=3.0,
            decode_responses=True,
        )
        response = await client.ping()
        if response is not True:
            return ComponentHealth(
                name="redis",
                status="degraded",
                message="PING returned an unexpected response",
                latency_ms=_elapsed_ms(started_at),
            )
        return ComponentHealth(
            name="redis",
            status="healthy",
            message="PING succeeded",
            latency_ms=_elapsed_ms(started_at),
        )
    except Exception as exc:
        return ComponentHealth(
            name="redis",
            status="unavailable",
            message=f"Connection failed: {type(exc).__name__}",
            latency_ms=_elapsed_ms(started_at),
        )
    finally:
        if client is not None:
            await client.aclose()


def _overall_readiness(
    components: list[ComponentHealth],
) -> tuple[str, bool]:
    required_names = {"dataset", "detector"}
    required_unavailable = any(
        component.name in required_names and component.status == "unavailable"
        for component in components
    )
    if required_unavailable:
        return "unavailable", False

    external_unavailable = any(
        component.status in {"degraded", "unavailable"}
        for component in components
        if component.name not in required_names
    )
    if external_unavailable:
        return "degraded", True

    return "healthy", True


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return process liveness without checking external dependencies."""

    return HealthResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_environment,
        status="healthy",
        timestamp=_utc_now(),
    )


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Return local artifact and external dependency readiness states."""

    local_components = [
        _dataset_health(settings),
        _detector_health(settings),
    ]
    external_components = await asyncio.gather(
        _influxdb_health(settings),
        _redis_health(settings),
        _ollama_health(settings),
    )
    components = local_components + list(external_components)
    status, ready = _overall_readiness(components)

    return ReadinessResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_environment,
        status=status,
        ready=ready,
        timestamp=_utc_now(),
        components=components,
    )
