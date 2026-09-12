"""Sidebar: session, uploads, document list, backend status."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from frontend.services.api import BackendClient


def render_sidebar(client: BackendClient) -> str:
    st.sidebar.title("AdaptiveRAG")
    st.sidebar.caption("Agentic routing across documents, general knowledge, and the web.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex[:10]

    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id

    if st.sidebar.button("New session"):
        st.session_state.session_id = uuid4().hex[:10]
        st.session_state.messages = []
        st.rerun()

    st.sidebar.subheader("Backend")
    try:
        health = client.health()
        st.sidebar.success(f"API {health.get('status', 'ok')}")
        st.sidebar.caption(f"Qdrant: {health.get('qdrant')} · SQLite: {health.get('database')}")
    except Exception:
        st.sidebar.error("Backend unreachable")
        health = {}

    st.sidebar.subheader("Upload documents")
    uploaded = st.sidebar.file_uploader(
        "PDF, TXT, Markdown, or DOCX",
        type=["pdf", "txt", "md", "markdown", "docx"],
    )
    description = st.sidebar.text_input("Description", placeholder="Optional context for this file")
    if uploaded is not None and st.sidebar.button("Index file"):
        tmp = Path("/tmp") / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        try:
            result = client.upload(tmp, description=description)
            st.sidebar.success(f"Indexed {result['document']['filename']} ({result['chunk_count']} chunks)")
        except Exception as exc:
            st.sidebar.error(f"Upload failed: {exc}")

    st.sidebar.subheader("Indexed documents")
    try:
        listing: dict[str, Any] = client.list_documents()
        docs = listing.get("documents") or []
        if not docs:
            st.sidebar.caption("No documents yet.")
        for item in docs:
            st.sidebar.markdown(f"- `{item['filename']}`")
    except Exception:
        st.sidebar.caption("Could not load document list.")

    return session_id
