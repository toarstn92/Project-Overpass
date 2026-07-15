"""
Streamlit Frontend — Project Overpass

Two pages:
  Chat        — user-facing chat interface with workflow visualisation
  Admin       — routing analytics, conversation history, system health

Run with: streamlit run frontend/app.py
Requires the FastAPI backend to be running on API_URL
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.environ.get("API_URL", "http://localhost:8000")

# Page config

st.set_page_config(
    page_title="Project Overpass",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling

st.markdown("""
<style>
    .path-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .badge-safety    { background: #fee2e2; color: #991b1b; }
    .badge-structured { background: #dbeafe; color: #1e40af; }
    .badge-reflective { background: #d1fae5; color: #065f46; }
    .badge-none      { background: #f3f4f6; color: #4b5563; }
    .badge-error     { background: #fef3c7; color: #92400e; }
    .workflow-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
    .source-chip {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

#  Session state ─

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, meta}

if "history" not in st.session_state:
    st.session_state.history = []    # full response objects for admin view

#  Helper functions 

PATH_COLOURS = {
    "safety": "badge-safety",
    "structured": "badge-structured",
    "reflective": "badge-reflective",
    "none": "badge-none",
    "error": "badge-error",
}

PATH_ICONS = {
    "safety": "🆘",
    "structured": "📋",
    "reflective": "🌿",
    "none": "🤷",
    "error": "⚠️",
}

PATH_LABELS = {
    "safety": "Safety Route",
    "structured": "Structured Guidance",
    "reflective": "Wisdom Retrieval",
    "none": "Out of Scope",
    "error": "Error",
}


def call_api(message: str) -> Dict[str, Any] | None:
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"message": message},
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "connection_failed", "message": "⚠️ Could not reach the assistant backend. Is the API running?", "path": "error"}
    except requests.exceptions.Timeout:
        return {"error": "timeout", "message": "⚠️ The assistant took too long to respond. Please try again.", "path": "error"}
    except requests.exceptions.HTTPError as e:
        return {"error": str(e), "message": f"⚠️ API error: {e}", "path": "error"}


def health_check() -> Dict[str, Any] | None:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def render_path_badge(path: str):
    colour = PATH_COLOURS.get(path, "badge-none")
    icon   = PATH_ICONS.get(path, "")
    label  = PATH_LABELS.get(path, path)
    st.markdown(
        f'<span class="path-badge {colour}">{icon} {label}</span>',
        unsafe_allow_html=True
    )


def render_workflow(meta: Dict[str, Any]):
    """Show the agentic workflow trace in an expandable panel."""
    with st.expander("🔍 Workflow trace", expanded=False):
        st.markdown('<div class="workflow-box">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            render_path_badge(meta.get("path", "none"))
            st.metric("Safety level", meta.get("safety_level", "none").upper())
        with col2:
            st.metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms")
            st.metric("Confidence", f"{meta.get('router_confidence', 0):.2f}")

        if meta.get("router_reasoning"):
            st.caption(f"**Router reasoning:** {meta['router_reasoning']}")

        if meta.get("state_label"):
            st.caption(f"**State identified:** {meta['state_id']} — _{meta['state_label']}_")

        if meta.get("framework"):
            st.caption(f"**Framework:** {meta['framework']}")

        if meta.get("practice"):
            p = meta["practice"]
            if isinstance(p, dict):
                st.caption(f"**Practice:** {p.get('name', '')} ({p.get('duration_min', '')} min)")

        if meta.get("sources"):
            st.markdown("**Sources drawn on:**")
            chips = ""
            for s in meta["sources"]:
                label = s.get("label") or s.get("label_name") or "Source"
                chips += f'<span class="source-chip">{label}</span> '
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


#  Sidebar ─

with st.sidebar:
    st.image("https://via.placeholder.com/140x40?text=OVERPASS", width=140)
    st.markdown("---")

    page = st.radio("Navigate", ["💬 Chat", "📊 Admin"], label_visibility="collapsed")

    st.markdown("---")
    health = health_check()
    if health:
        st.success(f"API online · {health.get('corpus_chunks', 0)} corpus chunks")
        st.caption(f"Model: {health.get('model', '—')}")
    else:
        st.error("API offline")

    st.markdown("---")
    st.caption("Project Overpass — an emotional assistant. All conversations are private and ephemeral.")


#  Chat page ─

if page == "💬 Chat":
    st.title("How are you feeling?")
    st.caption("Share what's on your mind. Everything you say stays private.")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🌿"):
            if msg["role"] == "assistant" and msg.get("meta"):
                st.markdown(
                    f'<div style="color:#e8e8e8; line-height:1.75; padding: 4px 0 8px 0;">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            if msg["role"] == "assistant" and msg.get("meta"):
                render_workflow(msg["meta"])

    # Input
    if prompt := st.chat_input("Tell me how you're feeling..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("Thinking..."):
                result = call_api(prompt)

            if result:
                st.session_state.history.append(
                    {"timestamp": datetime.now().isoformat(), "message": prompt, **result}
                )
                st.markdown(
                    f'<div style="color:#e8e8e8; line-height:1.75; padding: 4px 0 8px 0;">{result.get("message", "")}</div>',
                    unsafe_allow_html=True
                )
                render_workflow(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("message", ""),
                    "meta": result,
                })


#  Admin page 

elif page == "📊 Admin":
    st.title("📊 System Dashboard")

    history = st.session_state.history
    if not history:
        st.info("No conversations yet. Start chatting to see analytics here.")
        st.stop()

    #  Summary metrics ─
    total = len(history)
    by_path = {}
    for h in history:
        p = h.get("path", "unknown")
        by_path[p] = by_path.get(p, 0) + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total messages", total)
    col2.metric("Safety escalations", by_path.get("safety", 0))
    col3.metric("Structured", by_path.get("structured", 0))
    col4.metric("Reflective", by_path.get("reflective", 0))

    st.markdown("---")

    #  Path distribution 
    st.subheader("Path distribution")
    path_data = {PATH_LABELS.get(k, k): v for k, v in by_path.items()}
    st.bar_chart(path_data)

    #  Latency ─
    latencies = [h.get("latency_ms", 0) for h in history if h.get("latency_ms")]
    if latencies:
        st.subheader("Latency (ms)")
        col1, col2 = st.columns(2)
        col1.metric("Avg", f"{sum(latencies)/len(latencies):.0f} ms")
        col2.metric("Max", f"{max(latencies):.0f} ms")

    st.markdown("---")

    #  Conversation log ─
    st.subheader("Conversation log")
    for h in reversed(history):
        with st.expander(f"{h['timestamp'][:19]} — {h.get('path', '?').upper()} — \"{h['message'][:60]}...\""):
            st.markdown(f"**User:** {h['message']}")
            st.markdown(f"**Assistant:** {h.get('message', '')}")
            render_path_badge(h.get("path", "none"))
            if h.get("state_label"):
                st.caption(f"State: {h['state_id']} — {h['state_label']}")
            if h.get("router_reasoning"):
                st.caption(f"Routing: {h['router_reasoning']}")
            st.caption(f"Latency: {h.get('latency_ms', 0):.0f} ms · Confidence: {h.get('router_confidence', 0):.2f}")
