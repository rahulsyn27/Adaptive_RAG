"""Simplified generation nodes."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.constants import Route, SourceType
from app.llm.groq import get_llm
from app.rag.grading import _format_chunks
from app.rag.nodes import make_node
from app.schemas.response import Source, WebResult


def _format_history(turns: list[dict]) -> str:
    if not turns:
        return "(none)"
    return "\n".join(f"{h['role']}: {h['content']}" for h in turns[-8:])


def _format_web(results: list[WebResult] | None) -> str:
    if not results:
        return "(none)"
    return "\n\n".join(f"[{i}] {r.title}\nURL: {r.url}\n{r.content}" for i, r in enumerate(results, 1))


def _normalize_route(route) -> Route:
    try:
        return Route(str(route))
    except (ValueError, TypeError):
        return Route.GENERAL


def _make_sources(state: dict[str, Any]) -> list[Source]:
    route = _normalize_route(state.get("route"))
    if route is Route.INDEX:
        sources = []
        seen = set()
        for chunk in state.get("documents", []):
            key = (chunk.filename, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(Source(
                title=chunk.filename,
                source_type=SourceType.DOCUMENT,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                relevance_score=chunk.score,
                chunk_id=chunk.chunk_id,
            ))
        if not state.get("relevant"):
            return sources[:3]
        return sources
    if route is Route.SEARCH:
        return [
            Source(title=r.title, source_type=SourceType.WEB, url=r.url, relevance_score=r.score)
            for r in state.get("web_results", []) if r.url
        ]
    return []


def make_general_answer(llm: BaseChatModel | None = None) -> make_node:
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt("general", "system")),
        ("human", get_prompt("general", "human")),
    ])
    chain = prompt | model

    def run(state: dict[str, Any]) -> dict[str, Any]:
        raw = chain.invoke({"query": state["query"], "history": _format_history(state.get("conversation_history", []))})
        answer = str(raw).strip()
        return {"answer": answer, "sources": []}

    return make_node(run, "general_answer")


def make_generator(llm: BaseChatModel | None = None) -> make_node:
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt("generation", "system")),
        ("human", get_prompt("generation", "human")),
    ])
    chain = prompt | model

    def run(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("answer") and _normalize_route(state.get("route")) is Route.GENERAL:
            return {"sources": []}

        documents = state.get("documents", [])
        relevant = state.get("relevant", False)
        doc_context = _format_chunks(documents if relevant else documents[:2])

        if _normalize_route(state.get("route")) is Route.INDEX and not documents:
            doc_context = "(no documents retrieved; tell the user that indexed context is unavailable)"

        raw = chain.invoke({
            "query": state["query"],
            "history": _format_history(state.get("conversation_history", [])),
            "document_context": doc_context,
            "web_context": _format_web(state.get("web_results")),
        })
        answer = str(raw).strip()
        sources = _make_sources(state)

        return {"answer": answer, "sources": sources}

    return make_node(run, "generator")
