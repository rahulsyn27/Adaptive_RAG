"""Simplified state using dataclass instead of TypedDict."""

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import Route
from app.schemas.response import RetrievedChunk, Source, WebResult


@dataclass
class HistoryTurn:
    role: str
    content: str


@dataclass
class GraphState:
    query: str = ""
    session_id: str = ""
    route: Route = Route.GENERAL
    route_reason: str = ""
    conversation_history: list[HistoryTurn] = field(default_factory=list)
    documents: list[RetrievedChunk] = field(default_factory=list)
    web_results: list[WebResult] = field(default_factory=list)
    rewritten_query: str = ""
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    retrieval_attempts: int = 0
    relevance_score: float = 0.0
    relevant: bool = False
    grade_reason: str = ""
    retrieval_latency_ms: int = 0
    llm_latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for LangGraph compatibility."""
        history = []
        for h in self.conversation_history:
            if isinstance(h, dict):
                history.append(h)
            else:
                history.append({"role": h.role, "content": h.content})
        return {
            "query": self.query,
            "session_id": self.session_id,
            "route": self.route,
            "route_reason": self.route_reason,
            "conversation_history": history,
            "documents": self.documents,
            "web_results": self.web_results,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_attempts": self.retrieval_attempts,
            "relevance_score": self.relevance_score,
            "relevant": self.relevant,
            "grade_reason": self.grade_reason,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "metadata": self.metadata,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphState":
        """Create from LangGraph dict."""
        return cls(
            query=data.get("query", ""),
            session_id=data.get("session_id", ""),
            route=data.get("route", Route.GENERAL),
            route_reason=data.get("route_reason", ""),
            conversation_history=[HistoryTurn(**h) for h in data.get("conversation_history", [])],
            documents=data.get("documents", []),
            web_results=data.get("web_results", []),
            rewritten_query=data.get("rewritten_query", ""),
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
            retrieval_attempts=data.get("retrieval_attempts", 0),
            relevance_score=data.get("relevance_score", 0.0),
            relevant=data.get("relevant", False),
            grade_reason=data.get("grade_reason", ""),
            retrieval_latency_ms=data.get("retrieval_latency_ms", 0),
            llm_latency_ms=data.get("llm_latency_ms", 0),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )
