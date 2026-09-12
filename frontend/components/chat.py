"""Source citation display."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for item in sources:
        source_type = item.get("source_type")
        if source_type == "document":
            filename = item.get("filename") or "document"
            page = item.get("page")
            label = f"{filename} — page {page}" if page is not None else filename
            st.markdown(f"- {label}")
        elif source_type == "web":
            title = item.get("title") or item.get("url")
            url = item.get("url")
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")
