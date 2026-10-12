from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import httpx


class APIClientError(RuntimeError):
    """Safe frontend-facing API error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class APIClientConfig:
    base_url: str
    timeout_seconds: float = 180.0

    @classmethod
    def from_environment(cls) -> "APIClientConfig":
        raw = os.getenv("CLUSTERMIND_API_URL", "http://localhost:18000")
        return cls(base_url=raw.strip().rstrip("/"))


class ClusterMindAPI:
    """Bounded client used by Streamlit. Business logic remains in FastAPI."""

    def __init__(self, config: APIClientConfig | None = None) -> None:
        self.config = config or APIClientConfig.from_environment()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=self.config.timeout_seconds,
                    write=15.0,
                    pool=5.0,
                ),
                trust_env=False,
                headers={"User-Agent": "ClusterMind-Streamlit/0.1.0"},
            ) as client:
                response = client.request(
                    method,
                    path,
                    params=params,
                    json=json_body,
                )
        except httpx.TimeoutException as exc:
            raise APIClientError(
                "The operation timed out. The local LLM may still be processing; "
                "wait briefly and refresh the workflow list."
            ) from exc
        except httpx.NetworkError as exc:
            raise APIClientError(
                f"ClusterMind API is unavailable at {self.config.base_url}."
            ) from exc

        if response.is_error:
            try:
                payload = response.json()
                detail = payload.get("detail", response.text)
            except ValueError:
                detail = response.text or response.reason_phrase
            raise APIClientError(str(detail), status_code=response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            raise APIClientError("ClusterMind API returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise APIClientError("ClusterMind API returned an unexpected response.")
        return payload

    def overview(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/overview")

    def readiness(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/readiness")

    def platform_status(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/platform/status")

    def incidents(
        self,
        *,
        severity: str | None = None,
        incident_type: str | None = None,
        node_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if severity and severity != "All":
            params["severity"] = severity
        if incident_type and incident_type != "All":
            params["incident_type"] = incident_type
        if node_id and node_id != "All":
            params["node_id"] = node_id
        return self._request("GET", "/api/v1/incidents", params=params)

    def incident(self, incident_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/incidents/{quote(incident_id, safe='')}",
        )

    def analyze(
        self,
        incident_id: str,
        *,
        use_llm: bool,
        force: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/incidents/{quote(incident_id, safe='')}/analyze",
            json_body={"use_llm": use_llm, "force": force},
        )

    def workflows(self, *, limit: int = 100) -> dict[str, Any]:
        return self._request("GET", "/api/v1/workflows", params={"limit": limit})

    def workflow(self, workflow_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/workflows/{quote(workflow_id, safe='')}",
        )

    def decide(
        self,
        workflow_id: str,
        *,
        decision: str,
        actor: str,
        reason: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": decision, "actor": actor}
        if reason:
            body["reason"] = reason
        return self._request(
            "POST",
            f"/api/v1/workflows/{quote(workflow_id, safe='')}/decision",
            json_body=body,
        )
