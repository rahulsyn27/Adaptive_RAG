"""Streamlit chat UI for AdaptiveRAG."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components.chat import render_sources
from frontend.components.sidebar import render_sidebar
from frontend.services.api import BackendClient

st.set_page_config(page_title="AdaptiveRAG", page_icon="🧭", layout="wide")

BACKEND_URL = os.environ.get("ADAPTIVE_RAG_API_URL", "http://localhost:8000")
client = BackendClient(BACKEND_URL)

session_id = render_sidebar(client)

st.title("AdaptiveRAG")
st.caption("The router chooses INDEX, GENERAL, or SEARCH before answering.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            st.badge(message["route"]) if hasattr(st, "badge") else st.caption(f"Route: {message['route']}")
        if message.get("sources"):
            render_sources(message["sources"])
        meta = message.get("metadata") or {}
        if meta:
            bits = [f"{meta.get('latency_ms', 0)} ms"]
            if meta.get("retrieval_attempts"):
                bits.append(f"{meta['retrieval_attempts']} retrieval attempt(s)")
            if meta.get("rewritten_query"):
                bits.append("query rewritten")
            st.caption(" · ".join(bits))

prompt = st.chat_input("Ask about your documents, a concept, or current events")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Routing and answering..."):
            try:
                payload = client.chat(prompt, session_id)
            except Exception as exc:
                st.error("The backend could not complete this request.")
                st.caption(str(exc))
                st.stop()
        answer = payload.get("answer", "")
        route = payload.get("route", "")
        sources = payload.get("sources") or []
        metadata = payload.get("metadata") or {}
        st.markdown(answer)
        st.markdown(f"`{route}`")
        render_sources(sources)
        bits = [f"{metadata.get('latency_ms', 0)} ms"]
        if metadata.get("retrieval_attempts"):
            bits.append(f"{metadata['retrieval_attempts']} retrieval attempt(s)")
        st.caption(" · ".join(bits))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "route": route,
                "sources": sources,
                "metadata": metadata,
            }
        )
