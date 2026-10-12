from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st

from app.ui.api_client import APIClientError, ClusterMindAPI
from app.ui.state import reset_transient_cache


ASSET_ROOT = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSET_ROOT / "clustermind-logo.png"


SHELL_CSS = """
<style>
:root {
  --cm-ink: #172033;
  --cm-muted: #667085;
  --cm-violet: #7c3aed;
  --cm-indigo: #4f46e5;
  --cm-cyan: #06b6d4;
  --cm-coral: #f97360;
  --cm-amber: #f59e0b;
  --cm-emerald: #10b981;
}
.stApp {
  background:
    radial-gradient(circle at 8% 5%, rgba(6,182,212,.18), transparent 27%),
    radial-gradient(circle at 91% 14%, rgba(249,115,96,.18), transparent 29%),
    radial-gradient(circle at 76% 92%, rgba(124,58,237,.15), transparent 31%),
    linear-gradient(135deg, #fffdfb 0%, #f5f7ff 42%, #effbff 74%, #fff7ed 100%);
  color: var(--cm-ink);
}
[data-testid="stHeader"] { background: rgba(255,255,255,.78); backdrop-filter: blur(18px); }
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 15% 12%, rgba(6,182,212,.15), transparent 23%),
    linear-gradient(180deg, #14152d 0%, #211447 52%, #102b42 100%);
  border-right: 1px solid rgba(255,255,255,.12);
}
[data-testid="stSidebarContent"] { position: relative; padding-top: 0 !important; overflow-x: hidden !important; }
[data-testid="stSidebarNav"] { padding-top: 104px !important; }
[data-testid="stSidebarNav"] span, [data-testid="stSidebarNav"] a { font-weight: 680; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] a span, [data-testid="stSidebarNav"] a svg, [data-testid="stSidebarNav"] p { color: #dce8f7 !important; fill: #dce8f7 !important; }
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] header, [data-testid="stSidebarNav"] div[role="heading"] { color: #9fb5d1 !important; opacity: 1 !important; }
[data-testid="stSidebarNav"] a {
  border-radius: 13px; margin: 4px 12px; padding: .7rem .82rem;
  color: #d9e5f4; transition: transform .18s ease, background .18s ease, box-shadow .18s ease;
}
[data-testid="stSidebarNav"] a:hover {
  background: linear-gradient(100deg, rgba(6,182,212,.21), rgba(124,58,237,.23));
  transform: translateX(4px); color: white;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(100deg, #7c3aed 0%, #4f46e5 47%, #06b6d4 100%);
  box-shadow: 0 12px 28px rgba(79,70,229,.38); color: white;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: white !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] small { color:#a9bdd4 !important; }
.block-container { max-width: 1500px; padding-top: 1.15rem; padding-bottom: 2.5rem; }
.cm-brand {
  position: absolute; top: 14px; left: 14px; right: 14px; z-index: 20;
  width: auto; max-width: none; height: 72px;
  display:flex; align-items:center; gap:.72rem; padding:.55rem .78rem;
  border-radius:17px; box-sizing:border-box;
  background:linear-gradient(120deg, rgba(255,255,255,.13), rgba(255,255,255,.06));
  border:1px solid rgba(255,255,255,.14); backdrop-filter:blur(14px);
  box-shadow:0 12px 32px rgba(3,7,18,.22);
}
.cm-brand img { width:50px; height:50px; object-fit:contain; filter:drop-shadow(0 8px 11px rgba(34,211,238,.32)); }
.cm-brand-title { color:#fff; font-size:1.02rem; font-weight:850; line-height:1.05; }
.cm-brand-subtitle { color:#a5c9e9; font-size:.60rem; margin-top:.26rem; letter-spacing:.12em; text-transform:uppercase; }
.cm-runtime { margin:.55rem .78rem .8rem; padding:.8rem .85rem; border-radius:15px; background:rgba(7,15,32,.34); border:1px solid rgba(255,255,255,.10); }
.cm-runtime-row { display:flex; align-items:center; justify-content:space-between; color:#d5e4f3; font-size:.75rem; margin:.28rem 0; }
.cm-status { display:inline-flex; align-items:center; gap:.35rem; font-weight:750; }
.cm-dot { width:8px; height:8px; border-radius:999px; display:inline-block; }
.cm-dot-ok { background:#34d399; box-shadow:0 0 10px rgba(52,211,153,.9); }
.cm-dot-warn { background:#fbbf24; box-shadow:0 0 10px rgba(251,191,36,.9); }
.cm-dot-bad { background:#fb7185; box-shadow:0 0 10px rgba(251,113,133,.9); }
.stButton > button, .stLinkButton > a, [data-testid="stLinkButton"] a {
  min-height: 2.85rem; border-radius: 13px !important; font-weight: 750 !important;
  color: #fff !important; border: 0 !important;
  background: linear-gradient(100deg, #7c3aed 0%, #4f46e5 48%, #06b6d4 100%) !important;
  box-shadow: 0 10px 23px rgba(79,70,229,.23) !important;
  transition: transform .16s ease, box-shadow .16s ease !important;
}
.stButton > button:hover, .stLinkButton > a:hover, [data-testid="stLinkButton"] a:hover {
  color:#fff !important; transform: translateY(-2px); box-shadow:0 14px 30px rgba(79,70,229,.34) !important;
}
.stButton > button[kind="secondary"] { background:linear-gradient(100deg,#f97360,#f59e0b) !important; }
[data-testid="stMetric"], [data-testid="stDataFrame"] {
  background:rgba(255,255,255,.82); border:1px solid rgba(99,102,241,.14);
  border-radius:16px; box-shadow:0 10px 30px rgba(44,62,96,.08);
}
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stSidebar"] .stButton > button {
  background:linear-gradient(100deg,rgba(6,182,212,.8),rgba(79,70,229,.9)) !important;
  border:1px solid rgba(255,255,255,.15) !important;
}
@media (max-width: 900px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
}
</style>
"""


def _logo_data_uri() -> str:
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _safe_platform_status(api: ClusterMindAPI) -> dict[str, Any] | None:
    try:
        return api.platform_status()
    except APIClientError:
        return None


def render_shell(api: ClusterMindAPI) -> None:
    st.markdown(SHELL_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="cm-brand">
          <img src="{_logo_data_uri()}" alt="ClusterMind logo">
          <div><div class="cm-brand-title">ClusterMind</div><div class="cm-brand-subtitle">GPU Assurance Platform</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status = _safe_platform_status(api)
    overall = status.get("status", "unavailable") if status else "unavailable"
    dot_class = "cm-dot-ok" if overall == "healthy" else "cm-dot-warn" if overall == "degraded" else "cm-dot-bad"
    ollama_component = next((item for item in status.get("components", []) if item.get("name") == "ollama"), None) if status else None
    ollama = ollama_component.get("status", "unknown") if ollama_component else "unknown"
    st.sidebar.markdown(
        f"""
        <div class="cm-runtime">
          <div class="cm-runtime-row"><span>Backend</span><span class="cm-status"><i class="cm-dot {dot_class}"></i>{overall.title()}</span></div>
          <div class="cm-runtime-row"><span>Ollama</span><span>{ollama.title()}</span></div>
          <div class="cm-runtime-row"><span>Environment</span><span>Development</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Refresh live data", icon=":material/refresh:", use_container_width=True):
        reset_transient_cache()
        st.rerun()
    st.sidebar.caption("Synthetic telemetry · Human approval required")
