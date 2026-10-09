from __future__ import annotations

import json
import os
from typing import Any, Callable

import pandas as pd
import streamlit as st

from app.ui.api_client import APIClientError, ClusterMindAPI
from app.ui.components import (
    GRAFANA_URL,
    card,
    evidence_list,
    external_links,
    hero,
    incident_frame,
    status_pill,
    workflow_timeline,
)
from app.ui.styles import apply_styles


st.set_page_config(
    page_title="ClusterMind Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "http://localhost:18000/docs",
        "Report a bug": "http://localhost:18000/docs",
        "About": "ClusterMind multi-agent GPU workload assurance platform.",
    },
)
apply_styles()


@st.cache_resource
def api() -> ClusterMindAPI:
    return ClusterMindAPI()


def call(operation: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return operation(*args, **kwargs)
    except APIClientError as exc:
        st.error(str(exc), icon="🚨")
        return None


def load_incidents() -> list[dict[str, Any]]:
    payload = call(api().incidents, limit=100)
    return payload.get("items", []) if payload else []


def load_workflows() -> list[dict[str, Any]]:
    payload = call(api().workflows, limit=100)
    return payload.get("items", []) if payload else []


def sidebar() -> str:
    st.sidebar.markdown(
        '<div class="nav-brand"><div class="eyebrow">GPU ASSURANCE</div><h2>⚡ ClusterMind</h2><p>Explainable incident intelligence and human-approved recovery.</p></div>',
        unsafe_allow_html=True,
    )
    pages = [
        "Command Center",
        "Incident Center",
        "Agent Intelligence",
        "Recovery Approvals",
        "GPU Fleet",
        "Platform Health",
    ]
    selection = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("Refresh application data", use_container_width=True, icon="🔄"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("Recovery execution remains disabled. Every recommendation requires a human decision.")
    return selection


def command_center() -> None:
    hero(
        "ClusterMind Command Center",
        "From correlated GPU telemetry to an explainable diagnosis and a human-controlled recovery decision.",
    )
    overview = call(api().overview)
    incidents = load_incidents()
    workflows = load_workflows()
    if not overview:
        return

    critical = overview.get("severity_counts", {}).get("critical", 0)
    high = overview.get("severity_counts", {}).get("high", 0)
    cols = st.columns(5)
    with cols[0]: card("Cluster status", "Degraded" if critical or high else "Healthy", "Based on labelled development incidents", "critical" if critical else "high" if high else "healthy")
    with cols[1]: card("Observed GPUs", overview["gpu_count"], f"Across {overview['node_count']} nodes")
    with cols[2]: card("Incidents", overview["incident_count"], f"{critical} critical · {high} high")
    with cols[3]: card("Pending approvals", overview["pending_approval_count"], "Human decision required")
    with cols[4]: card("Workflows", overview["workflow_count"], "Persisted agent analyses")

    st.subheader("Operational brief")
    priority = next((item for item in incidents if item["severity"] == "critical"), incidents[0] if incidents else None)
    if priority:
        detail = call(api().incident, priority["incident_id"])
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown("### Highest-priority incident")
            st.markdown(status_pill(priority["severity"]), unsafe_allow_html=True)
            st.write(f"**{priority['incident_type'].replace('_', ' ').title()}** on `{priority['node_id']} / {priority['gpu_id']}`")
            st.write(f"Affected workload: `{priority['affected_workload']}`")
            if detail:
                st.info(f"Validated runbook: {detail['runbook_title']}")
        with c2:
            st.markdown("### Recommended next step")
            existing = next((item for item in workflows if item["incident_id"] == priority["incident_id"]), None)
            if existing:
                st.markdown(status_pill(existing["status"]), unsafe_allow_html=True)
                st.write(existing.get("recovery_plan", {}).get("summary", "Agent analysis is available."))
            else:
                st.write("Run the multi-agent analysis to correlate hardware state, workload impact, root cause, and safe recovery.")
                if st.button("Run priority analysis", type="primary", use_container_width=True):
                    with st.status("Running four-agent analysis…", expanded=True) as status_box:
                        st.write("Analyzing GPU telemetry")
                        st.write("Correlating workload impact")
                        st.write("Grounding root cause with runbooks and Ollama")
                        st.write("Building human-approved recovery plan")
                        result = call(api().analyze, priority["incident_id"], use_llm=True)
                        if result:
                            st.session_state["selected_workflow"] = result["workflow_id"]
                            status_box.update(label="Analysis completed", state="complete")
                            st.rerun()
                        status_box.update(label="Analysis failed safely", state="error")
    external_links()


def incident_center() -> None:
    hero("Incident Center", "Filter the incident inventory, inspect one event, and launch a bounded agent analysis.")
    all_items = load_incidents()
    if not all_items:
        st.info("No incidents are available.")
        return
    c1, c2, c3 = st.columns(3)
    severities = ["All"] + sorted({item["severity"] for item in all_items})
    types = ["All"] + sorted({item["incident_type"] for item in all_items})
    nodes = ["All"] + sorted({item["node_id"] for item in all_items})
    severity = c1.selectbox("Severity", severities)
    incident_type = c2.selectbox("Incident type", types, format_func=lambda value: value.replace("_", " ").title())
    node = c3.selectbox("Node", nodes)
    payload = call(api().incidents, severity=severity, incident_type=incident_type, node_id=node, limit=100)
    items = payload.get("items", []) if payload else []
    st.dataframe(incident_frame(items), use_container_width=True, hide_index=True)
    if not items:
        return
    selected_id = st.selectbox("Select incident", [item["incident_id"] for item in items])
    detail = call(api().incident, selected_id)
    if not detail:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Severity", detail["severity"].title())
    c2.metric("Node / GPU", f"{detail['node_id']} / {detail['gpu_id']}")
    c3.metric("Workload", detail["affected_workload"])
    c4.metric("GPU hours lost", f"{detail['gpu_hours_lost']:.2f}")
    st.markdown(f"**Runbook:** {detail['runbook_title']}  \n**Recorded recovery:** {detail['recovery_action']}")
    use_llm = st.toggle("Use Ollama root-cause reasoning", value=True, help="If unavailable, the workflow falls back to deterministic runbook diagnosis.")
    if st.button("Run four-agent analysis", type="primary", icon="🧠"):
        with st.spinner("ClusterMind agents are analyzing the incident…"):
            result = call(api().analyze, selected_id, use_llm=use_llm)
        if result:
            st.session_state["selected_workflow"] = result["workflow_id"]
            st.success("Analysis reached the human approval gate.")
            st.rerun()


def selected_workflow_selector(workflows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not workflows:
        st.info("No agent workflows have been created yet.")
        return None
    ids = [item["workflow_id"] for item in workflows]
    preferred = st.session_state.get("selected_workflow")
    index = ids.index(preferred) if preferred in ids else 0
    selected = st.selectbox("Workflow", ids, index=index)
    st.session_state["selected_workflow"] = selected
    return call(api().workflow, selected)


def agent_intelligence() -> None:
    hero("Agent Intelligence", "Inspect each specialized agent, the evidence used, confidence, fallbacks, and the approval boundary.")
    workflow = selected_workflow_selector(load_workflows())
    if not workflow:
        return
    workflow_timeline(workflow)
    tabs = st.tabs(["Telemetry", "Workload Impact", "Root Cause", "Recovery Plan", "Audit JSON"])
    telemetry = workflow.get("telemetry") or {}
    workload = workflow.get("workload_impact") or {}
    root = workflow.get("root_cause") or {}
    recovery = workflow.get("recovery_plan") or {}
    with tabs[0]:
        st.markdown(f"### {telemetry.get('condition', 'Pending').replace('_', ' ').title()}")
        st.write(telemetry.get("summary", "No telemetry result."))
        evidence_list(telemetry.get("evidence", []))
    with tabs[1]:
        st.write(workload.get("summary", "No workload result."))
        cols = st.columns(4)
        cols[0].metric("Impact", workload.get("impact_level", "Pending").title())
        cols[1].metric("Throughput", f"{workload.get('throughput_change_percent', 0):.1f}%")
        cols[2].metric("Latency", f"{workload.get('latency_change_percent', 0):.1f}%")
        cols[3].metric("Checkpoint", "Available" if workload.get("checkpoint_available") else "Unavailable")
        evidence_list(workload.get("evidence", []))
    with tabs[2]:
        st.markdown(f"### {root.get('probable_cause', 'Pending').replace('_', ' ').title()}")
        st.write(root.get("explanation", "No diagnosis."))
        st.write(f"LLM used: **{root.get('used_llm', False)}** · Fallback used: **{root.get('fallback_used', False)}**")
        for candidate in root.get("candidates", []):
            with st.expander(f"{candidate['cause'].replace('_', ' ').title()} · {candidate['confidence']:.1%}"):
                for item in candidate.get("supporting_evidence", []):
                    st.write("•", item)
    with tabs[3]:
        st.write(recovery.get("summary", "No recovery plan."))
        st.markdown(status_pill(recovery.get("risk_level", "pending")), unsafe_allow_html=True)
        for step in recovery.get("steps", []):
            st.markdown(f"**{step['order']}. {step['action'].replace('_', ' ').title()}** → `{step['target']}`")
            st.caption(step["description"])
        st.warning("No infrastructure action is executed by this console. Human approval only changes the workflow to queued.")
    with tabs[4]:
        st.json(workflow, expanded=False)


def recovery_approvals() -> None:
    hero("Recovery Approvals", "Review the exact allowlisted plan, safety checks, rollback, and record an auditable decision.")
    workflows = [item for item in load_workflows() if item.get("recovery_plan")]
    workflow = selected_workflow_selector(workflows)
    if not workflow:
        return
    plan = workflow["recovery_plan"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Workflow", workflow["status"].replace("_", " ").title())
    c2.metric("Risk", plan["risk_level"].title())
    c3.metric("Confidence", f"{plan['confidence']:.1%}")
    c4.metric("Expected downtime", f"{plan['expected_downtime_minutes']} min")
    st.subheader(plan["title"])
    st.write(plan["summary"])
    with st.expander("Recovery steps", expanded=True):
        for step in plan["steps"]:
            st.markdown(f"**{step['order']}. {step['action'].replace('_', ' ').title()}** → `{step['target']}`")
            st.write(step["description"])
            if step.get("rollback"):
                st.caption(f"Rollback: {step['rollback']}")
    with st.expander("Safety checks", expanded=True):
        for item in plan["safety_checks"]:
            st.write("•", item)
    st.info(f"Rollback plan: {plan['rollback_plan']}")
    if workflow.get("approval"):
        st.success(f"Decision already recorded: {workflow['approval']['decision']} by {workflow['approval']['actor']}")
        return
    actor = st.text_input("Operator identity", value="review-operator", max_chars=150)
    reason = st.text_area("Decision notes", max_chars=1000, placeholder="Required for rejection or requesting more evidence.")
    approve, reject, evidence_col = st.columns(3)
    if approve.button("Approve and queue", type="primary", use_container_width=True, icon="✅"):
        result = call(api().decide, workflow["workflow_id"], decision="approved", actor=actor, reason=reason or None)
        if result:
            st.success("Recommendation approved and queued. No recovery action was executed.")
            st.rerun()
    if reject.button("Reject recommendation", use_container_width=True, icon="⛔"):
        if not reason.strip():
            st.error("A rejection reason is required.")
        else:
            result = call(api().decide, workflow["workflow_id"], decision="rejected", actor=actor, reason=reason)
            if result:
                st.warning("Recommendation rejected and cancelled.")
                st.rerun()
    if evidence_col.button("Request more evidence", use_container_width=True, icon="🔎"):
        if not reason.strip():
            st.error("Explain what additional evidence is required.")
        else:
            result = call(api().decide, workflow["workflow_id"], decision="more_evidence_requested", actor=actor, reason=reason)
            if result:
                st.info("The workflow is marked for additional evidence.")
                st.rerun()


def gpu_fleet() -> None:
    hero("GPU Fleet", "A clear fleet summary with deep links to Grafana for full time-series investigation.")
    overview = call(api().overview)
    incidents = load_incidents()
    if not overview:
        return
    cols = st.columns(4)
    cols[0].metric("GPU nodes", overview["node_count"])
    cols[1].metric("GPUs", overview["gpu_count"])
    cols[2].metric("Workloads", overview["workload_count"])
    cols[3].metric("Sampling interval", f"{overview['sampling_interval_minutes']} min")
    st.subheader("Incident distribution")
    left, right = st.columns(2)
    with left:
        severity_df = pd.DataFrame({"Severity": list(overview["severity_counts"]), "Count": list(overview["severity_counts"].values())}).set_index("Severity")
        st.bar_chart(severity_df, color="#f97316")
    with right:
        type_df = pd.DataFrame({"Scenario": [key.replace("_", " ").title() for key in overview["incident_type_counts"]], "Count": list(overview["incident_type_counts"].values())}).set_index("Scenario")
        st.bar_chart(type_df, color="#8b5cf6")
    st.dataframe(incident_frame(incidents), use_container_width=True, hide_index=True)
    st.link_button("Open full GPU telemetry in Grafana", f"{GRAFANA_URL}/d/clustermind-overview/clustermind-command-center", type="primary", use_container_width=True)


def platform_health() -> None:
    hero("Platform Health", "Live readiness, service capabilities, and direct links to the observability stack.")
    status = call(api().platform_status)
    readiness = call(api().readiness)
    if status:
        st.markdown(status_pill(status["status"]), unsafe_allow_html=True)
        columns = st.columns(max(1, len(status["components"])))
        for column, component in zip(columns, status["components"]):
            with column:
                card(component["name"], component["status"].title(), component["detail"], component["status"])
        st.subheader("Runtime capabilities")
        for name, enabled in status["capabilities"].items():
            st.toggle(name.replace("_", " ").title(), value=bool(enabled), disabled=True)
    if readiness:
        st.subheader("Dependency readiness")
        st.dataframe(pd.DataFrame(readiness["components"]), use_container_width=True, hide_index=True)
    external_links()


page = sidebar()
PAGES = {
    "Command Center": command_center,
    "Incident Center": incident_center,
    "Agent Intelligence": agent_intelligence,
    "Recovery Approvals": recovery_approvals,
    "GPU Fleet": gpu_fleet,
    "Platform Health": platform_health,
}
PAGES[page]()
st.markdown('<div class="footer">ClusterMind · Explainable GPU assurance · Human approval required</div>', unsafe_allow_html=True)
