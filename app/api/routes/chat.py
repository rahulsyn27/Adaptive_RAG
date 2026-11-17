"""Simplified chat endpoint."""

import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.api.dependencies import db_session_dep
from app.core.constants import MessageRole
from app.core.exceptions import AdaptiveRAGError, GraphExecutionError, to_http_exception
from app.core.logging import bind_session, get_logger
from app.database.repositories import add_message, create_session
from app.rag.graph import build_simple_graph
from app.rag.state import GraphState
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse
from app.schemas.response import Source

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


def _extract_text(value: Any) -> str:
    """Extract plain text from AIMessage, str, or other types."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # Handle LangChain AIMessage
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    # Fallback to string representation
    return str(value)


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_simple_graph()


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, request: Request, db: Session = Depends(db_session_dep)) -> ChatResponse:
    bind_session(payload.session_id)
    started = time.perf_counter()

    create_session(db, payload.session_id)
    add_message(db, payload.session_id, MessageRole.USER.value, payload.query)
    db.commit()

    graph = get_compiled_graph()
    initial = GraphState(
        query=payload.query,
        session_id=payload.session_id,
        retrieval_attempts=0,
        llm_latency_ms=0,
        retrieval_latency_ms=0,
        metadata={"request_id": getattr(request.state, "request_id", "-")},
    ).to_dict()

    try:
        result = await graph.ainvoke(initial)
    except AdaptiveRAGError as exc:
        logger.error("chat_failed", extra={"error": exc.message})
        raise to_http_exception(exc) from exc
    except Exception as exc:
        logger.error("chat_failed", extra={"error": str(exc)})
        raise to_http_exception(GraphExecutionError("The adaptive workflow failed to complete.")) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    meta = result.get("metadata", {})

    raw_sources = result.get("sources") or []
    sources = [s if isinstance(s, Source) else Source.model_validate(s) for s in raw_sources]

    logger.info("chat_complete", extra={
        "route": result.get("route"),
        "retrieval_attempts": result.get("retrieval_attempts", 0),
        "retrieved_documents": meta.get("retrieved_documents", 0),
        "relevant_documents": meta.get("relevant_documents", 0),
        "llm_latency_ms": result.get("llm_latency_ms", 0),
        "retrieval_latency_ms": result.get("retrieval_latency_ms", 0),
        "latency_ms": latency_ms,
    })

    answer_text = _extract_text(result.get("answer")) or "I was unable to produce an answer."
    return ChatResponse(
        answer=answer_text,
        route=result.get("route", "GENERAL"),
        sources=sources,
        session_id=payload.session_id,
        metadata=ChatMetadata(
            retrieval_attempts=result.get("retrieval_attempts", 0),
            retrieved_documents=meta.get("retrieved_documents", 0),
            relevant_documents=meta.get("relevant_documents", 0),
            latency_ms=latency_ms,
            retrieval_latency_ms=result.get("retrieval_latency_ms", 0),
            llm_latency_ms=result.get("llm_latency_ms", 0),
            rewritten_query=result.get("rewritten_query"),
        ),
    )
