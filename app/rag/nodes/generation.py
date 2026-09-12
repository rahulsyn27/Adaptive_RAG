"""Answer generation for INDEX, GENERAL, and SEARCH paths."""

import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.constants import Route, SourceType
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.groq import get_llm
from app.rag.nodes.grading import _format_chunks
from app.rag.nodes.router import _format_history
from app.rag.policies import normalize_route
from app.rag.state import GraphState
from app.schemas.response import Source, WebResult

logger = get_logger(__name__)


def _text(message: BaseMessage | str) -> str:
    if isinstance(message, str):
        return message.strip()
    content = message.content
    return content.strip() if isinstance(content, str) else str(content).strip()


def _format_web(results: list[WebResult] | None) -> str:
    if not results:
        return "(none)"
    lines = []
    for index, item in enumerate(results, start=1):
        lines.append(f"[{index}] {item.title}\nURL: {item.url}\n{item.content}")
    return "\n\n".join(lines)


def sources_from_state(state: GraphState) -> list[Source]:
    route = normalize_route(state.get("route"))
    if route is Route.INDEX:
        sources: list[Source] = []
        seen: set[tuple[str | None, int | None]] = set()
        for chunk in state.get("documents") or []:
            key = (chunk.filename, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                Source(
                    title=chunk.filename,
                    source_type=SourceType.DOCUMENT,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    relevance_score=chunk.score,
                    chunk_id=chunk.chunk_id,
                )
            )
        if not state.get("relevant"):
            return sources[:3]
        return sources
    if route is Route.SEARCH:
        return [
            Source(
                title=item.title,
                source_type=SourceType.WEB,
                url=item.url,
                relevance_score=item.score,
            )
            for item in state.get("web_results") or []
            if item.url
        ]
    return []


def make_general_answer_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("general", "system")),
            ("human", get_prompt("general", "human")),
        ]
    )
    chain = prompt | model

    def general_answer(state: GraphState) -> GraphState:
        started = time.perf_counter()
        try:
            raw = chain.invoke(
                {"query": state["query"], "history": _format_history(state.get("conversation_history"))}
            )
            answer = _text(raw)
        except Exception as exc:
            raise LLMError("The language model failed while answering a general question.") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "answer": answer,
            "sources": [],
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
        }

    return general_answer


def make_generation_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    """Unified generator for document and web evidence (and INDEX retries)."""
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("generation", "system")),
            ("human", get_prompt("generation", "human")),
        ]
    )
    chain = prompt | model

    def generate(state: GraphState) -> GraphState:
        if state.get("answer") and normalize_route(state.get("route")) is Route.GENERAL:
            return {"sources": []}
        started = time.perf_counter()
        documents = state.get("documents") or []
        relevant = bool(state.get("relevant"))
        document_context = _format_chunks(documents if relevant else documents[:2])
        if normalize_route(state.get("route")) is Route.INDEX and not documents:
            document_context = (
                "(no documents retrieved; tell the user that indexed context is unavailable)"
            )
        try:
            raw = chain.invoke(
                {
                    "query": state["query"],
                    "history": _format_history(state.get("conversation_history")),
                    "document_context": document_context,
                    "web_context": _format_web(state.get("web_results")),
                }
            )
            answer = _text(raw)
        except Exception as exc:
            raise LLMError("The language model failed while generating an answer.") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        sources = sources_from_state(state)
        logger.info(
            "generation_complete",
            extra={"source_count": len(sources), "llm_latency_ms": elapsed_ms, "route": state.get("route")},
        )
        return {
            "answer": answer,
            "sources": sources,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
        }

    return generate
