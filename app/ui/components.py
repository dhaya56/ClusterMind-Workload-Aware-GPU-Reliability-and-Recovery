from __future__ import annotations

import html
import os
from typing import Any

import pandas as pd
import streamlit as st


GRAFANA_URL = os.getenv("CLUSTERMIND_GRAFANA_URL", "http://localhost:13000")
JAEGER_URL = os.getenv("CLUSTERMIND_JAEGER_URL", "http://localhost:16686")
PROMETHEUS_URL = os.getenv("CLUSTERMIND_PROMETHEUS_URL", "http://localhost:19090")
API_PUBLIC_URL = os.getenv("CLUSTERMIND_API_PUBLIC_URL", "http://localhost:18000")
NTFY_URL = os.getenv("CLUSTERMIND_NTFY_URL", "http://localhost:18080/clustermind-alerts")


def escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def hero(title: str, subtitle: str, eyebrow: str = "CLUSTERMIND OPERATIONS") -> None:
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{escape(eyebrow)}</div>
        <h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>""",
        unsafe_allow_html=True,
    )


def status_pill(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    return f'<span class="status-pill status-{escape(normalized)}">{escape(value.replace("_", " ").title())}</span>'


def card(label: str, value: Any, help_text: str = "", status: str | None = None) -> None:
    suffix = status_pill(status) if status else ""
    st.markdown(
        f"""<div class="glass-card"><div class="card-label">{escape(label)}</div>
        <div class="card-value">{escape(value)}</div>{suffix}
        <div class="card-help">{escape(help_text)}</div></div>""",
        unsafe_allow_html=True,
    )


def external_links() -> None:
    cols = st.columns(5)
    links = [
        ("Grafana", f"{GRAFANA_URL}/d/clustermind-overview/clustermind-command-center"),
        ("Jaeger", JAEGER_URL),
        ("Prometheus", PROMETHEUS_URL),
        ("API Docs", f"{API_PUBLIC_URL}/docs"),
        ("Notifications", NTFY_URL),
    ]
    for column, (label, url) in zip(cols, links):
        column.link_button(label, url, use_container_width=True)


def incident_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    frame = pd.DataFrame(items)
    preferred = [
        "incident_id",
        "severity",
        "incident_type",
        "node_id",
        "gpu_id",
        "affected_workload",
        "start_timestamp",
        "incident_status",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns]


def workflow_timeline(workflow: dict[str, Any]) -> None:
    stages = [
        ("GPU Telemetry Agent", workflow.get("telemetry")),
        ("Workload Impact Agent", workflow.get("workload_impact")),
        ("Root Cause Agent", workflow.get("root_cause")),
        ("Recovery Agent", workflow.get("recovery_plan")),
    ]
    for label, data in stages:
        if data:
            execution = data.get("execution", {})
            status = execution.get("status", "completed")
            duration = execution.get("duration_ms")
            confidence = data.get("confidence")
            details = []
            if confidence is not None:
                details.append(f"Confidence {confidence:.1%}")
            if duration is not None:
                details.append(f"{duration:.1f} ms")
            st.markdown(
                f'<div class="agent-step"><b>{escape(label)}</b> {status_pill(status)}<br><small>{escape(" · ".join(details))}</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="agent-step"><b>{escape(label)}</b> {status_pill("pending")}</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<div class="agent-step"><b>Human Approval Gate</b> {status_pill(workflow.get("status", "pending"))}</div>',
        unsafe_allow_html=True,
    )


def evidence_list(items: list[dict[str, Any]], *, limit: int = 12) -> None:
    for item in items[:limit]:
        observation = item.get("observation", "Evidence available")
        metric = item.get("metric", "signal")
        st.markdown(
            f'<div class="evidence"><b>{escape(metric)}</b><br>{escape(observation)}</div>',
            unsafe_allow_html=True,
        )
