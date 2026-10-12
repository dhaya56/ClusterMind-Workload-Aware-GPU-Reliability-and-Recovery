from __future__ import annotations

import streamlit as st

from app.ui.api_client import ClusterMindAPI
from app.ui.legacy_pages import (
    agent_intelligence,
    gpu_fleet,
    incident_center,
    platform_health,
    recovery_approvals,
)
from app.ui.shell import render_shell
from app.ui.pages.command_center import command_center
from app.ui.state import initialize_state


st.set_page_config(
    page_title="ClusterMind",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_state()
api = ClusterMindAPI()
render_shell(api)

navigation = st.navigation(
    {
        "OPERATIONS": [
            st.Page(
                            command_center,
                title="Command Center",
                icon=":material/dashboard:",
                url_path="command-center",
                default=True,
            ),
            st.Page(
                incident_center,
                title="Incident Center",
                icon=":material/warning:",
                url_path="incidents",
            ),
            st.Page(
                agent_intelligence,
                title="Agent Intelligence",
                icon=":material/network_intelligence:",
                url_path="agent-intelligence",
            ),
        ],
        "CONTROL": [
            st.Page(
                recovery_approvals,
                title="Recovery Approvals",
                icon=":material/approval:",
                url_path="recovery-approvals",
            ),
        ],
        "OBSERVABILITY": [
            st.Page(
                gpu_fleet,
                title="GPU Fleet",
                icon=":material/memory:",
                url_path="gpu-fleet",
            ),
            st.Page(
                platform_health,
                title="Platform Health",
                icon=":material/monitor_heart:",
                url_path="platform-health",
            ),
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
