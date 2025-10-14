from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chatbot import Chatbot

st.set_page_config(page_title="Atlas — Building Assistant", page_icon="🏢", layout="wide")

@st.cache_resource(show_spinner=False)
def get_bot(data_dir: str) -> Chatbot:
    return Chatbot(Path(data_dir))

# ---- Render helper ----
def render_answer(payload: Dict[str, Any]):
    title: Optional[str] = payload.get("title")
    text: Optional[str] = payload.get("text")
    df: Optional[pd.DataFrame] = payload.get("df")

    if title:
        st.subheader(title)
    if text:
        st.markdown(text)

    # Evidence (SQL + sample rows)
    ev = payload.get("evidence")
    if ev:
        with st.expander("Evidence", expanded=False):
            if "sql" in ev:
                st.code(ev["sql"], language="sql")
            if "sample" in ev and ev["sample"]:
                st.json(ev["sample"])

    # Route trace (always shown so you can verify auto-routing)
    if "route" in payload and isinstance(payload["route"], dict):
        r = payload["route"]
        st.caption(f"Route: {r.get('route')} • {r.get('trace','')}")

    if isinstance(df, pd.DataFrame):
        if df.empty:
            st.info("No rows to display.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

# ---- Sidebar  ----
st.sidebar.title("⚙️ Settings")
data_dir = st.sidebar.text_input("Data directory", value=".", help="Folder with atlas.duckdb and the CSVs")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick queries")
for q in [
    "list floors",
    "rooms on floor 3",
    "free meeting rooms now on floor 2",
    "utilization by floor last 30 days",
    "busiest rooms on floor 3 last 14 days",
    "underused rooms last month",
    "Best 2 coffee machine spots on floor 3 now (avoid quiet areas)",
]:
    if st.sidebar.button(q, key=f"qq_{q}", use_container_width=True):
        st.session_state["prefill"] = q

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()

# ---- Header ----
st.title("Atlas — Building Assistant")

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and isinstance(m.get("payload"), dict):
            render_answer(m["payload"])
        else:
            st.markdown(m.get("content", ""))

# ---- Input ----
prefill = st.session_state.pop("prefill", "")
prompt = st.chat_input("Ask anything about the building…")
if prefill and not prompt:
    prompt = prefill

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        bot = get_bot(data_dir)
        # strict=True (always) to prevent ungrounded answers; show_route=True for trace
        out: Dict[str, Any] = bot.ask(prompt, strict=True, show_route=True)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"⚠️ {e}")
        st.session_state["messages"].append({"role": "assistant", "content": f"⚠️ {e}"})
    else:
        st.session_state["messages"].append({"role": "assistant", "payload": out})
        with st.chat_message("assistant"):
            render_answer(out)

    st.rerun()
