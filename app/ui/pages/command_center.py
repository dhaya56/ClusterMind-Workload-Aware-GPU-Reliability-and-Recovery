from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from app.ui.api_client import APIClientError, ClusterMindAPI


PAGE_CSS = """
<style>
.cm-page-head { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin:.15rem 0 1.1rem; }
.cm-page-title { font-size:2.15rem; font-weight:850; letter-spacing:-.04em; color:#172033; margin:0; }
.cm-page-subtitle { color:#667085; font-size:.98rem; margin-top:.35rem; max-width:760px; }
.cm-live { display:flex; align-items:center; gap:.45rem; padding:.48rem .72rem; border-radius:999px; background:#ecfdf5; color:#047857; border:1px solid #a7f3d0; font-weight:750; font-size:.75rem; white-space:nowrap; }
.cm-live-dot { width:8px; height:8px; border-radius:99px; background:#10b981; box-shadow:0 0 0 5px rgba(16,185,129,.12); }
.cm-context { display:flex; gap:.5rem; flex-wrap:wrap; margin:.2rem 0 1.1rem; }
.cm-chip { padding:.38rem .62rem; border-radius:999px; font-size:.72rem; font-weight:750; border:1px solid transparent; }
.cm-chip-purple { color:#6d28d9; background:#f3e8ff; border-color:#ddd6fe; }
.cm-chip-cyan { color:#0e7490; background:#cffafe; border-color:#a5f3fc; }
.cm-chip-coral { color:#c2410c; background:#ffedd5; border-color:#fed7aa; }
.cm-kpi { min-height:142px; padding:1rem 1.05rem; border-radius:20px; color:white; box-shadow:0 14px 34px rgba(60,60,120,.16); position:relative; overflow:hidden; }
.cm-kpi:after { content:''; position:absolute; width:100px; height:100px; border-radius:50%; right:-26px; top:-34px; background:rgba(255,255,255,.15); }
.cm-kpi-label { font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; opacity:.88; }
.cm-kpi-value { font-size:1.72rem; font-weight:880; margin:.42rem 0 .28rem; letter-spacing:-.04em; }
.cm-kpi-help { font-size:.74rem; opacity:.86; line-height:1.35; max-width:185px; }
.cm-purple { background:linear-gradient(135deg,#7c3aed,#4f46e5); }
.cm-coral { background:linear-gradient(135deg,#f43f5e,#f97316); }
.cm-cyan { background:linear-gradient(135deg,#0891b2,#06b6d4); }
.cm-emerald { background:linear-gradient(135deg,#059669,#10b981); }
.cm-panel { border-radius:22px; background:rgba(255,255,255,.88); border:1px solid rgba(99,102,241,.13); box-shadow:0 18px 45px rgba(44,62,96,.10); padding:1.2rem; height:100%; }
.cm-panel-title { color:#172033; font-size:1.05rem; font-weight:850; margin-bottom:.22rem; }
.cm-panel-subtitle { color:#64748b; font-size:.78rem; margin-bottom:.9rem; }
.cm-priority-head { display:flex; justify-content:space-between; gap:.8rem; align-items:flex-start; }
.cm-severity { display:inline-flex; padding:.28rem .55rem; border-radius:999px; background:#fff1f2; color:#be123c; border:1px solid #fecdd3; font-size:.7rem; font-weight:850; }
.cm-incident-name { font-size:1.35rem; color:#172033; font-weight:870; margin:.5rem 0 .3rem; }
.cm-resource { color:#475569; font-size:.82rem; }
.cm-agent-track { display:grid; grid-template-columns:repeat(5,1fr); gap:.45rem; margin:1rem 0; }
.cm-stage { padding:.65rem .45rem; border-radius:13px; text-align:center; font-size:.67rem; font-weight:780; border:1px solid #e2e8f0; background:#f8fafc; color:#64748b; }
.cm-stage-done { color:#047857; background:#ecfdf5; border-color:#a7f3d0; }
.cm-stage-active { color:#6d28d9; background:#f5f3ff; border-color:#c4b5fd; box-shadow:0 8px 18px rgba(109,40,217,.12); }
.cm-decision { border-radius:16px; padding:.85rem .95rem; background:linear-gradient(120deg,#f5f3ff,#ecfeff); border:1px solid #c4b5fd; }
.cm-decision-label { color:#6d28d9; font-size:.68rem; font-weight:850; letter-spacing:.08em; text-transform:uppercase; }
.cm-decision-text { color:#27324a; font-size:.84rem; margin-top:.3rem; line-height:1.48; }
.cm-queue-item { display:flex; justify-content:space-between; align-items:center; padding:.72rem .1rem; border-bottom:1px solid #eef2f7; }
.cm-queue-item:last-child { border-bottom:0; }
.cm-queue-name { color:#27324a; font-weight:760; font-size:.79rem; }
.cm-queue-help { color:#7c8798; font-size:.68rem; }
.cm-count { width:31px; height:31px; display:grid; place-items:center; border-radius:10px; font-weight:850; font-size:.76rem; }
.cm-count-red { background:#fff1f2; color:#be123c; }
.cm-count-amber { background:#fffbeb; color:#b45309; }
.cm-count-purple { background:#f5f3ff; color:#6d28d9; }
.cm-activity { display:flex; gap:.75rem; padding:.68rem 0; border-bottom:1px solid #eef2f7; }
.cm-activity:last-child { border-bottom:0; }
.cm-time { color:#7c3aed; font-size:.68rem; font-weight:850; min-width:54px; }
.cm-activity-text { color:#475569; font-size:.76rem; line-height:1.35; }
.cm-action-row { display:flex; gap:.65rem; flex-wrap:wrap; margin-top:.9rem; }
.cm-action { display:inline-flex; text-decoration:none !important; padding:.68rem .9rem; border-radius:12px; color:white !important; font-weight:800; font-size:.76rem; background:linear-gradient(100deg,#7c3aed,#4f46e5,#06b6d4); box-shadow:0 9px 20px rgba(79,70,229,.2); transition:.15s ease; }
.cm-action:hover { transform:translateY(-2px); box-shadow:0 13px 26px rgba(79,70,229,.3); }
.cm-action-secondary { background:linear-gradient(100deg,#f97360,#f59e0b); }
@media(max-width:900px) { .cm-agent-track{grid-template-columns:1fr 1fr;} .cm-page-head{display:block;} .cm-live{margin-top:.7rem;width:max-content;} }
</style>
"""


def _safe(callable_: Any, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return callable_(*args, **kwargs)
    except APIClientError as exc:
        st.error(str(exc), icon="🚨")
        return None


def _priority(incidents: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    workflow_by_incident = {item["incident_id"]: item for item in workflows}
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    workflow_rank = {"awaiting_approval": 4, "failed": 3, "more_evidence_requested": 2}

    def score(item: dict[str, Any]) -> tuple[int, int, str]:
        workflow = workflow_by_incident.get(item["incident_id"])
        return (
            severity_rank.get(item.get("severity", "low"), 0),
            workflow_rank.get(workflow.get("status", "") if workflow else "", 0),
            item.get("start_timestamp", ""),
        )

    if not incidents:
        return None, None
    selected = max(incidents, key=score)
    return selected, workflow_by_incident.get(selected["incident_id"])


def _stage_class(data: Any, workflow_status: str, is_approval: bool = False) -> str:
    if is_approval:
        return "cm-stage-active" if workflow_status == "awaiting_approval" else "cm-stage-done" if workflow_status in {"approved", "rejected"} else ""
    return "cm-stage-done" if data else ""


def _decision_summary(workflow: dict[str, Any] | None) -> str:
    if not workflow:
        return "No agent workflow exists for the priority incident. Open Incident Center to start a bounded analysis."
    recovery = workflow.get("recovery_plan") or {}
    return recovery.get("summary") or "Agent analysis is available for review."


def _activity_rows(workflows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in workflows[:4]:
        timestamp = item.get("updated_at", "")
        try:
            label = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone().strftime("%H:%M")
        except ValueError:
            label = "Recent"
        status = item.get("status", "unknown").replace("_", " ")
        rows.append((label, f"{item['incident_id']} moved to {status}."))
    return rows or [("Now", "No workflow activity has been recorded yet.")]


def _render_snapshot(api: ClusterMindAPI) -> None:
    overview = _safe(api.overview)
    incidents_payload = _safe(api.incidents, limit=100)
    workflows_payload = _safe(api.workflows, limit=100)
    platform = _safe(api.platform_status)
    if not overview or not incidents_payload or not workflows_payload:
        return

    incidents = incidents_payload.get("items", [])
    workflows = workflows_payload.get("items", [])
    priority, workflow = _priority(incidents, workflows)
    platform_status = platform.get("status", "unavailable") if platform else "unavailable"
    critical = overview.get("severity_counts", {}).get("critical", 0)
    high = overview.get("severity_counts", {}).get("high", 0)
    last_refresh = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")

    st.markdown(
        f"""<div class="cm-page-head"><div><h1 class="cm-page-title">Command Center</h1>
        <div class="cm-page-subtitle">A live operational overview of platform readiness, priority incidents, agent decisions, and human control points.</div></div>
        <div class="cm-live"><i class="cm-live-dot"></i> Live platform · updated {last_refresh}</div></div>
        <div class="cm-context"><span class="cm-chip cm-chip-purple">Development environment</span>
        <span class="cm-chip cm-chip-cyan">Synthetic telemetry snapshot</span>
        <span class="cm-chip cm-chip-coral">Recovery execution disabled</span></div>""",
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    cards = [
        ("ClusterMind services", platform_status.title(), "API, evidence repository, workflow store, and local model.", "cm-emerald" if platform_status == "healthy" else "cm-coral"),
        ("Critical attention", critical, f"{high} additional high-severity incidents in the dataset.", "cm-coral"),
        ("Awaiting decisions", overview.get("pending_approval_count", 0), "Agent recommendations requiring a human response.", "cm-purple"),
        ("GPU estate", overview.get("gpu_count", 0), f"Observed across {overview.get('node_count', 0)} simulated nodes.", "cm-cyan"),
    ]
    for column, (label, value, help_text, color) in zip(columns, cards):
        column.markdown(
            f'<div class="cm-kpi {color}"><div class="cm-kpi-label">{label}</div><div class="cm-kpi-value">{value}</div><div class="cm-kpi-help">{help_text}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        if priority:
            workflow_status = workflow.get("status", "not_started") if workflow else "not_started"
            stages = [
                ("Telemetry", workflow.get("telemetry") if workflow else None, False),
                ("Impact", workflow.get("workload_impact") if workflow else None, False),
                ("Root cause", workflow.get("root_cause") if workflow else None, False),
                ("Recovery", workflow.get("recovery_plan") if workflow else None, False),
                ("Approval", workflow, True),
            ]
            stage_html = "".join(
                f'<div class="cm-stage {_stage_class(data, workflow_status, approval)}">{"✓ " if data and not approval else ""}{label}</div>'
                for label, data, approval in stages
            )
            action_path = "/recovery-approvals" if workflow_status == "awaiting_approval" else "/agent-intelligence" if workflow else "/incidents"
            action_label = "Review approval" if workflow_status == "awaiting_approval" else "Open analysis" if workflow else "Analyze incident"
            severity = priority.get("severity", "unknown")
            st.markdown(
                f"""<div class="cm-panel"><div class="cm-priority-head"><div><div class="cm-panel-title">Priority decision intelligence</div>
                <div class="cm-panel-subtitle">The highest-ranked item based on severity and workflow state.</div></div><span class="cm-severity">{severity.title()}</span></div>
                <div class="cm-incident-name">{priority['incident_type'].replace('_',' ').title()}</div>
                <div class="cm-resource">{priority['node_id']} / {priority['gpu_id']} · Workload {priority['affected_workload']}</div>
                <div class="cm-agent-track">{stage_html}</div>
                <div class="cm-decision"><div class="cm-decision-label">Recommended next step</div><div class="cm-decision-text">{_decision_summary(workflow)}</div></div>
                <div class="cm-action-row"><a class="cm-action" href="{action_path}" target="_self">{action_label}</a>
                <a class="cm-action cm-action-secondary" href="/incidents" target="_self">Open incident workspace</a></div></div>""",
                unsafe_allow_html=True,
            )
    with right:
        failed = sum(item.get("status") == "failed" for item in workflows)
        more_evidence = sum(item.get("status") == "more_evidence_requested" for item in workflows)
        activity = "".join(
            f'<div class="cm-activity"><div class="cm-time">{time}</div><div class="cm-activity-text">{text}</div></div>'
            for time, text in _activity_rows(workflows)
        )
        st.markdown(
            f"""<div class="cm-panel"><div class="cm-panel-title">Action queue</div><div class="cm-panel-subtitle">Live workflow states requiring operator awareness.</div>
            <div class="cm-queue-item"><div><div class="cm-queue-name">Awaiting approval</div><div class="cm-queue-help">Human decision required</div></div><div class="cm-count cm-count-red">{overview.get('pending_approval_count',0)}</div></div>
            <div class="cm-queue-item"><div><div class="cm-queue-name">More evidence</div><div class="cm-queue-help">Analysis needs additional signals</div></div><div class="cm-count cm-count-amber">{more_evidence}</div></div>
            <div class="cm-queue-item"><div><div class="cm-queue-name">Failed workflows</div><div class="cm-queue-help">Safe failures requiring inspection</div></div><div class="cm-count cm-count-purple">{failed}</div></div>
            <div style="height:.6rem"></div><div class="cm-panel-title">Recent activity</div>{activity}</div>""",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=30)
def live_command_center() -> None:
    _render_snapshot(ClusterMindAPI())


def _operational_command_center() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    live_command_center()

# STEP_3_1_MONITORING_ONBOARDING
ONBOARDING_CSS = """
<style>
.cm-welcome { padding:1.4rem 1.5rem; border-radius:26px; color:white; background:linear-gradient(120deg,#7c3aed 0%,#4f46e5 43%,#06b6d4 100%); box-shadow:0 20px 50px rgba(79,70,229,.24); margin:.35rem 0 1.15rem; position:relative; overflow:hidden; }
.cm-welcome:after { content:''; position:absolute; width:230px; height:230px; border-radius:50%; right:-75px; top:-110px; background:rgba(255,255,255,.14); }
.cm-welcome-kicker { font-size:.7rem; font-weight:850; letter-spacing:.15em; text-transform:uppercase; opacity:.85; }
.cm-welcome-title { font-size:2.05rem; font-weight:880; letter-spacing:-.04em; margin:.35rem 0; }
.cm-welcome-copy { max-width:780px; line-height:1.55; font-size:.9rem; opacity:.92; }
.cm-source-card { min-height:282px; padding:1.15rem; border-radius:22px; background:rgba(255,255,255,.90); border:1px solid rgba(99,102,241,.14); box-shadow:0 16px 38px rgba(44,62,96,.10); }
.cm-source-live { border:2px solid #8b5cf6; box-shadow:0 18px 42px rgba(124,58,237,.16); }
.cm-source-icon { width:44px; height:44px; display:grid; place-items:center; border-radius:14px; color:white; font-size:1.25rem; background:linear-gradient(135deg,#7c3aed,#06b6d4); margin-bottom:.8rem; }
.cm-source-title { color:#172033; font-size:1.08rem; font-weight:850; }
.cm-source-status { display:inline-flex; margin:.45rem 0 .7rem; padding:.25rem .5rem; border-radius:999px; font-size:.68rem; font-weight:850; }
.cm-source-ready { color:#047857; background:#ecfdf5; border:1px solid #a7f3d0; }
.cm-source-future { color:#64748b; background:#f1f5f9; border:1px solid #cbd5e1; }
.cm-source-copy { color:#667085; font-size:.78rem; line-height:1.5; min-height:76px; }
.cm-source-facts { margin-top:.7rem; color:#334155; font-size:.75rem; line-height:1.65; }
.cm-flow { display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; padding:1rem; border-radius:18px; background:linear-gradient(100deg,#fff7ed,#f5f3ff,#ecfeff); border:1px solid #ddd6fe; margin:.9rem 0; }
.cm-flow-step { padding:.42rem .58rem; border-radius:10px; background:white; color:#334155; font-size:.72rem; font-weight:780; box-shadow:0 5px 13px rgba(71,85,105,.09); }
.cm-flow-arrow { color:#7c3aed; font-weight:900; }
.cm-provenance { color:#475569; font-size:.78rem; line-height:1.65; }
</style>
"""


def _monitoring_onboarding() -> None:
    api = ClusterMindAPI()
    overview = _safe(api.overview)
    readiness = _safe(api.readiness)
    if not overview or not readiness:
        st.error("The staged telemetry source could not be validated. Check Platform Health before starting monitoring.")
        return

    ready = bool(readiness.get("ready"))
    st.markdown(ONBOARDING_CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="cm-welcome"><div class="cm-welcome-kicker">Explainable GPU Assurance</div>
        <div class="cm-welcome-title">Choose what ClusterMind should monitor</div>
        <div class="cm-welcome-copy">Begin with the reproducible synthetic cluster replay prepared for this review. ClusterMind validates GPU, node, workload, incident, detector, and runbook evidence before any agent analysis is shown.</div></div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            f"""<div class="cm-source-card cm-source-live"><div class="cm-source-icon">▶</div><div class="cm-source-title">Synthetic Cluster Replay</div>
            <span class="cm-source-status cm-source-ready">{'Ready' if ready else 'Unavailable'}</span>
            <div class="cm-source-copy">Replays the validated January 2026 development dataset through the production-shaped evidence and agent contracts.</div>
            <div class="cm-source-facts"><b>{overview['node_count']} nodes</b> · <b>{overview['gpu_count']} GPUs</b><br><b>{overview['workload_count']} workloads</b> · <b>{overview['incident_count']} labelled incidents</b><br>{overview['sampling_interval_minutes']}-minute sampling interval</div></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Start demo monitoring", type="primary", use_container_width=True, disabled=not ready, icon=":material/play_arrow:"):
            st.session_state.monitoring_started = True
            st.session_state.telemetry_source = "synthetic_cluster_replay"
            st.session_state.last_operation = "monitoring_started"
            st.session_state.last_refresh_at = datetime.now(timezone.utc)
            st.rerun()
    with c2:
        st.markdown(
            """<div class="cm-source-card"><div class="cm-source-icon" style="background:linear-gradient(135deg,#f97360,#f59e0b)">⇧</div><div class="cm-source-title">Uploaded Evidence</div>
            <span class="cm-source-status cm-source-future">Not configured</span><div class="cm-source-copy">Future adapter for validated GPU, node, workload, incident, and detector evidence files.</div>
            <div class="cm-source-facts">Schema validation required<br>Immutable evidence package<br>No direct browser-side analysis</div></div>""",
            unsafe_allow_html=True,
        )
        with st.expander("View required evidence"):
            st.code("gpu_telemetry.csv\nnode_telemetry.csv\nworkloads.csv\nincidents.csv\ndetections.csv\nrunbooks/*.yaml", language="text")
    with c3:
        st.markdown(
            """<div class="cm-source-card"><div class="cm-source-icon" style="background:linear-gradient(135deg,#059669,#06b6d4)">⌁</div><div class="cm-source-title">Live GPU Connector</div>
            <span class="cm-source-status cm-source-future">Not connected</span><div class="cm-source-copy">Production integration path for NVIDIA DCGM Exporter, Prometheus, or InfluxDB telemetry.</div>
            <div class="cm-source-facts">Streaming telemetry adapter<br>Same evidence contract<br>Physical GPU feed not claimed</div></div>""",
            unsafe_allow_html=True,
        )
        with st.expander("View integration architecture"):
            st.markdown("DCGM Exporter → Prometheus or InfluxDB → Incident Repository → Agent workflow")

    st.markdown("### How input reaches the agents")
    st.markdown(
        """<div class="cm-flow"><span class="cm-flow-step">Cluster simulator</span><span class="cm-flow-arrow">→</span><span class="cm-flow-step">Generated telemetry</span><span class="cm-flow-arrow">→</span><span class="cm-flow-step">Isolation Forest detections</span><span class="cm-flow-arrow">→</span><span class="cm-flow-step">Incident Repository</span><span class="cm-flow-arrow">→</span><span class="cm-flow-step">Four-agent LangGraph</span><span class="cm-flow-arrow">→</span><span class="cm-flow-step">Human approval</span></div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Data provenance and exact ingestion boundary"):
        st.markdown(
            """<div class="cm-provenance"><b>Telemetry source:</b> Synthetic GPU cluster simulator<br>
            <b>GPU input:</b> data/generated/gpu_telemetry.csv<br>
            <b>Node input:</b> data/generated/node_telemetry.csv<br>
            <b>Workload input:</b> data/generated/workloads.csv<br>
            <b>Incident input:</b> data/generated/incidents.csv<br>
            <b>Detector evidence:</b> data/artifacts/detections.csv<br>
            <b>Operational knowledge:</b> runbooks/*.yaml<br>
            <b>Ingestion service:</b> app/services/incidents.py<br>
            <b>Agent boundary:</b> IncidentRepository.build_evidence(incident_id)</div>""",
            unsafe_allow_html=True,
        )


def command_center() -> None:
    if not st.session_state.get("monitoring_started", False):
        _monitoring_onboarding()
        return
    _operational_command_center()

