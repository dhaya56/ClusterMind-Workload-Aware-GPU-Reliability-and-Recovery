from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Mapping, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings


LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaServiceError(RuntimeError):
    """Base error for bounded Ollama service failures."""


class OllamaUnavailableError(OllamaServiceError):
    """Raised when the configured Ollama endpoint is unavailable."""


class OllamaResponseError(OllamaServiceError):
    """Raised when Ollama returns malformed, oversized, or invalid output."""


@dataclass(frozen=True)
class OllamaGeneration:
    """Validated metadata for one structured Ollama generation."""

    model: str
    content: str
    duration_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int


class OllamaService:
    """Bounded synchronous client for local structured Ollama inference."""

    MAX_PROMPT_CHARACTERS = 24_000
    MAX_RESPONSE_CHARACTERS = 16_000
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_SEED = 42

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        max_attempts: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        if isinstance(max_attempts, bool) or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if retry_delay_seconds < 0 or retry_delay_seconds > 5:
            raise ValueError("retry_delay_seconds must be between 0 and 5")

        self.settings = settings or get_settings()
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.settings.ollama_base_url,
            timeout=httpx.Timeout(
                connect=min(5.0, float(self.settings.ollama_timeout_seconds)),
                read=float(self.settings.ollama_timeout_seconds),
                write=10.0,
                pool=5.0,
            ),
            trust_env=False,
            limits=httpx.Limits(
                max_connections=5,
                max_keepalive_connections=2,
                keepalive_expiry=30.0,
            ),
            headers={"User-Agent": "ClusterMind/0.1.0"},
        )

    def __enter__(self) -> "OllamaService":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def health(self) -> dict[str, Any]:
        """Return endpoint and configured-model availability."""

        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            raise OllamaUnavailableError("Ollama health check failed") from exc

        models = payload.get("models", [])
        available = {
            str(item.get("name", "")).strip()
            for item in models
            if isinstance(item, Mapping)
        }
        configured = self.settings.ollama_model
        configured_base = configured.split(":", 1)[0]
        model_available = any(
            name == configured
            or name.split(":", 1)[0] == configured_base
            for name in available
        )
        return {
            "status": "healthy",
            "configured_model": configured,
            "model_available": model_available,
            "available_models": sorted(name for name in available if name),
        }

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        text = content.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return fenced.group(1).strip() if fenced else text

    @staticmethod
    def _schema_for(model_type: type[T]) -> dict[str, Any]:
        schema = model_type.model_json_schema()
        schema.setdefault("additionalProperties", False)
        return schema

    def _validate_prompt(self, system_prompt: str, user_prompt: str) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be blank")
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be blank")
        total = len(system_prompt) + len(user_prompt)
        if total > self.MAX_PROMPT_CHARACTERS:
            raise ValueError(
                f"prompt exceeds {self.MAX_PROMPT_CHARACTERS} characters"
            )

    def _request_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> dict[str, Any]:
        return {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": self._schema_for(response_model),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.DEFAULT_TEMPERATURE,
                "seed": self.DEFAULT_SEED,
                "num_predict": 500,
                "top_p": 0.9,
            },
            "keep_alive": "15m",
        }

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> tuple[T, OllamaGeneration]:
        """Generate and schema-validate one bounded JSON response."""

        self._validate_prompt(system_prompt, user_prompt)
        payload = self._request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )
        started = perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.post("/api/chat", json=payload)

                if response.is_error:
                    error_body = response.text.strip()
                    if len(error_body) > 2000:
                        error_body = error_body[:2000] + "...[truncated]"

                    raise OllamaResponseError(
                        f"Ollama returned HTTP {response.status_code}: "
                        f"{error_body or 'empty error response'}"
                    )

                try:
                    body = response.json()
                except json.JSONDecodeError as exc:
                    raise OllamaResponseError(
                        "Ollama returned a non-JSON HTTP response"
                    ) from exc
                message = body.get("message")
                if not isinstance(message, Mapping):
                    raise OllamaResponseError(
                        "Ollama response is missing the message object"
                    )
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise OllamaResponseError(
                        "Ollama response content is empty"
                    )
                if len(content) > self.MAX_RESPONSE_CHARACTERS:
                    raise OllamaResponseError(
                        "Ollama response exceeded the configured size limit"
                    )

                cleaned = self._strip_markdown_fence(content)
                try:
                    decoded = json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    raise OllamaResponseError(
                        "Ollama response is not valid JSON"
                    ) from exc
                if not isinstance(decoded, dict):
                    raise OllamaResponseError(
                        "Ollama response root must be a JSON object"
                    )

                try:
                    parsed = response_model.model_validate(decoded)
                except ValidationError as exc:
                    raise OllamaResponseError(
                        f"Ollama response failed schema validation: {exc}"
                    ) from exc

                generation = OllamaGeneration(
                    model=str(body.get("model") or self.settings.ollama_model),
                    content=cleaned,
                    duration_ms=round((perf_counter() - started) * 1000.0, 3),
                    prompt_tokens=(
                        int(body["prompt_eval_count"])
                        if body.get("prompt_eval_count") is not None
                        else None
                    ),
                    completion_tokens=(
                        int(body["eval_count"])
                        if body.get("eval_count") is not None
                        else None
                    ),
                    attempts=attempt,
                )
                return parsed, generation
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                json.JSONDecodeError,
                OllamaResponseError,
            ) as exc:
                last_error = exc
                LOGGER.warning(
                    "Structured Ollama generation attempt %s/%s failed: %s",
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if attempt < self.max_attempts and self.retry_delay_seconds:
                    sleep(self.retry_delay_seconds * attempt)

        if isinstance(last_error, (httpx.TimeoutException, httpx.NetworkError)):
            raise OllamaUnavailableError(
                "Ollama was unavailable after bounded retries"
            ) from last_error
        raise OllamaResponseError(
            "Ollama did not return a valid structured response after bounded retries"
        ) from last_error
