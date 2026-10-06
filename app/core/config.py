from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the ClusterMind application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ClusterMind"
    app_version: str = "0.1.0"
    app_environment: Literal["development", "test", "production"] = (
        "development"
    )
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    api_v1_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )

    data_directory: Path = PROJECT_ROOT / "data" / "generated"
    artifact_directory: Path = PROJECT_ROOT / "data" / "artifacts"
    model_artifact_directory: Path = PROJECT_ROOT / "app" / "ml" / "artifacts"
    runbook_directory: Path = PROJECT_ROOT / "runbooks"

    influxdb_url: str = "http://localhost:8086"
    influxdb_org: str = "clustermind"
    influxdb_bucket: str = "gpu_telemetry"
    influxdb_token: str = "clustermind-development-token"
    influxdb_timeout_ms: int = Field(default=10000, ge=1000)

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = Field(default=120, ge=1)

    prometheus_metrics_path: str = "/metrics"

    otel_enabled: bool = True
    otel_service_name: str = "clustermind-api"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_exporter_otlp_insecure: bool = True

    ntfy_enabled: bool = False
    ntfy_base_url: str = "http://localhost:8080"
    ntfy_topic: str = "clustermind-alerts"

    smtp_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_address: str = ""
    smtp_to_address: str = ""
    smtp_use_tls: bool = True

    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    @field_validator(
        "api_v1_prefix",
        "prometheus_metrics_path",
        mode="before",
    )
    @classmethod
    def ensure_leading_slash(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Path value cannot be empty")
        return text if text.startswith("/") else f"/{text}"

    @field_validator(
        "influxdb_url",
        "ollama_base_url",
        "ntfy_base_url",
        "otel_exporter_otlp_endpoint",
        mode="before",
    )
    @classmethod
    def strip_trailing_slash(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Service URL cannot be empty")
        return text.rstrip("/")

    @field_validator(
        "data_directory",
        "artifact_directory",
        "model_artifact_directory",
        "runbook_directory",
        mode="before",
    )
    @classmethod
    def resolve_project_path(cls, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @field_validator("cors_origins")
    @classmethod
    def normalize_cors_origins(cls, value: str) -> str:
        origins = [item.strip().rstrip("/") for item in value.split(",")]
        origins = [item for item in origins if item]
        if not origins:
            raise ValueError("At least one CORS origin is required")
        return ",".join(dict.fromkeys(origins))

    @model_validator(mode="after")
    def validate_notification_settings(self) -> "Settings":
        if self.ntfy_enabled and not self.ntfy_topic.strip():
            raise ValueError("ntfy_topic is required when ntfy is enabled")

        if self.smtp_enabled:
            required = {
                "smtp_host": self.smtp_host,
                "smtp_from_address": self.smtp_from_address,
                "smtp_to_address": self.smtp_to_address,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                raise ValueError(
                    "Missing SMTP settings: " + ", ".join(sorted(missing))
                )

        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return self.cors_origins.split(",")

    @property
    def is_development(self) -> bool:
        return self.app_environment == "development"

    def ensure_runtime_directories(self) -> None:
        for directory in (
            self.data_directory,
            self.artifact_directory,
            self.model_artifact_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated application settings singleton."""

    settings = Settings()
    settings.ensure_runtime_directories()
    return settings
