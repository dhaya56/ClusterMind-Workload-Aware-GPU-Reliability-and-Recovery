from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st


DEFAULTS: dict[str, Any] = {
    "monitoring_started": False,
    "telemetry_source": None,
    "selected_incident_id": None,
    "selected_workflow": None,
    "operator_identity": "review-operator",
    "live_refresh_enabled": True,
    "live_refresh_seconds": 5,
    "last_refresh_at": None,
    "last_operation": None,
}


def initialize_state() -> None:
    """Create stable cross-page session keys without overwriting user choices."""

    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def mark_refresh() -> None:
    st.session_state.last_refresh_at = datetime.now(timezone.utc)


def reset_transient_cache() -> None:
    """Clear cached API data while preserving navigation and operator context."""

    st.cache_data.clear()
    mark_refresh()
