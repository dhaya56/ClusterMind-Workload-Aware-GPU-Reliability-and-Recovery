from __future__ import annotations

import streamlit as st


CSS = r"""
<style>
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background: radial-gradient(circle at 5% 5%, #10294c 0, #07111f 30%, #050a12 70%); color: #edf6ff; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #09192b 0%, #07111f 60%, #10082b 100%); border-right: 1px solid rgba(118,180,255,.18); }
[data-testid="stSidebar"] * { color: #e8f3ff; }
[data-testid="stHeader"] { background: rgba(6,14,25,.76); backdrop-filter: blur(14px); }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
.hero { padding: 1.4rem 1.6rem; border-radius: 22px; border: 1px solid rgba(120,190,255,.25); background: linear-gradient(120deg, rgba(13,57,96,.96), rgba(61,32,130,.92)); box-shadow: 0 18px 60px rgba(0,0,0,.3); margin-bottom: 1rem; }
.eyebrow { color:#7dd3fc; font-size:.72rem; letter-spacing:.18em; font-weight:800; }
.hero h1 { margin:.3rem 0 .35rem; font-size:2rem; line-height:1.15; }
.hero p { margin:0; color:#dbeafe; max-width:900px; }
.glass-card { border:1px solid rgba(148,199,255,.18); border-radius:18px; padding:1rem 1.1rem; background:linear-gradient(145deg,rgba(16,36,61,.88),rgba(9,20,35,.92)); box-shadow:0 12px 35px rgba(0,0,0,.22); min-height:130px; }
.glass-card:hover { transform:translateY(-2px); border-color:rgba(96,165,250,.55); transition:.18s ease; }
.card-label { color:#9db6d0; font-size:.76rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.card-value { font-size:1.75rem; font-weight:800; margin:.28rem 0; color:#f8fbff; }
.card-help { color:#91a9c2; font-size:.76rem; }
.status-pill { display:inline-block; padding:.26rem .58rem; border-radius:999px; font-size:.72rem; font-weight:800; letter-spacing:.04em; }
.status-healthy,.status-completed,.status-approved { background:rgba(16,185,129,.18); color:#6ee7b7; border:1px solid rgba(16,185,129,.35); }
.status-high,.status-degraded,.status-awaiting_approval,.status-queued { background:rgba(245,158,11,.17); color:#fcd34d; border:1px solid rgba(245,158,11,.35); }
.status-critical,.status-failed,.status-rejected,.status-unavailable { background:rgba(239,68,68,.17); color:#fca5a5; border:1px solid rgba(239,68,68,.35); }
.status-medium,.status-running,.status-more_evidence_requested { background:rgba(99,102,241,.18); color:#c7d2fe; border:1px solid rgba(99,102,241,.35); }
.status-low,.status-suppressed,.status-not_started { background:rgba(14,165,233,.18); color:#7dd3fc; border:1px solid rgba(14,165,233,.35); }
.agent-step { border-left:4px solid #38bdf8; padding:.7rem 1rem; margin:.55rem 0; border-radius:0 12px 12px 0; background:rgba(15,38,65,.7); }
.evidence { padding:.65rem .8rem; border-radius:10px; background:rgba(10,25,44,.75); margin:.35rem 0; border:1px solid rgba(148,199,255,.12); }
.nav-brand { padding:.8rem .75rem 1.1rem; }
.nav-brand h2 { margin:0; font-size:1.25rem; }
.nav-brand p { color:#87a3bf; font-size:.72rem; margin:.25rem 0 0; }
[data-testid="stMetric"] { border:1px solid rgba(148,199,255,.18); padding:.8rem 1rem; border-radius:16px; background:rgba(11,28,48,.76); }
.stButton>button, .stDownloadButton>button { border-radius:11px; font-weight:700; border:1px solid rgba(125,190,255,.3); }
.stButton>button:hover { border-color:#38bdf8; color:#e0f2fe; transform:translateY(-1px); }
div[data-testid="stDataFrame"] { border:1px solid rgba(148,199,255,.16); border-radius:14px; overflow:hidden; }
.footer { text-align:center; color:#69839e; font-size:.72rem; padding:2rem 0 .5rem; }
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
