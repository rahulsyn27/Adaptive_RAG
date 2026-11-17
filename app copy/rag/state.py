"""Typed LangGraph state for AdaptiveRAG."""

from typing import Any, TypedDict

from app.core.constants import Route
from app.schemas.response import RetrievedChunk, Source, WebResult


class HistoryTurn(TypedDict):
    role: str
    content: str


class GraphState(TypedDict, total=False):
    query: str
    session_id: str
    route: Route | str
    route_reason: str
    conversation_history: list[HistoryTurn]
    documents: list[RetrievedChunk]
    web_results: list[WebResult]
    rewritten_query: str
    answer: str
    sources: list[Source]
    retrieval_attempts: int
    relevance_score: float
    relevant: bool
    grade_reason: str
    retrieval_latency_ms: int
    llm_latency_ms: int
    metadata: dict[str, Any]
    error: str | None
