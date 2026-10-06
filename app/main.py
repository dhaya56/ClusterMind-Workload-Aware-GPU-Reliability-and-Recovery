from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.observability.metrics import configure_metrics
from app.schemas.health import RootResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize local runtime state for the API process."""

    settings: Settings = app.state.settings
    settings.ensure_runtime_directories()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the ClusterMind FastAPI application."""

    runtime_settings = settings or get_settings()

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        description=(
            "Multi-agent GPU workload assurance and recovery platform for "
            "training and inference clusters."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings

    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    application.include_router(
        api_router,
        prefix=runtime_settings.api_v1_prefix,
    )

    configure_metrics(
        application,
        metrics_path=runtime_settings.prometheus_metrics_path,
        version=runtime_settings.app_version,
        environment=runtime_settings.app_environment,
    )

    @application.get(
        "/",
        response_model=RootResponse,
        tags=["system"],
        summary="ClusterMind API metadata",
    )
    async def root() -> RootResponse:
        return RootResponse(
            service=runtime_settings.app_name,
            version=runtime_settings.app_version,
            environment=runtime_settings.app_environment,
            status="running",
            documentation_url="/docs",
            health_url=f"{runtime_settings.api_v1_prefix}/health",
            readiness_url=f"{runtime_settings.api_v1_prefix}/readiness",
            metrics_url=runtime_settings.prometheus_metrics_path,
        )

    return application


app = create_app()
