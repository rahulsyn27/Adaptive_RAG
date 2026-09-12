"""Relevance grading of retrieved chunks."""

import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.logging import get_logger
from app.llm.groq import get_llm
from app.rag.state import GraphState
from app.schemas.chat import GradeResult
from app.schemas.response import RetrievedChunk

logger = get_logger(__name__)


def _format_chunks(chunks: list[RetrievedChunk] | None) -> str:
    if not chunks:
        return "(no documents retrieved)"
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        locator = chunk.filename or "document"
        if chunk.page is not None:
            locator = f"{locator} p.{chunk.page}"
        parts.append(f"[{index}] {locator}\n{chunk.content}")
    return "\n\n".join(parts)


def make_grading_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    model = llm or get_llm()
    structured = model.with_structured_output(GradeResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("grader", "system")),
            ("human", get_prompt("grader", "human")),
        ]
    )
    chain = prompt | structured

    def grade(state: GraphState) -> GraphState:
        started = time.perf_counter()
        documents = state.get("documents") or []
        if not documents:
            result = GradeResult(relevant=False, score=0.0, reason="No documents were retrieved.")
        else:
            try:
                parsed = chain.invoke({"query": state["query"], "context": _format_chunks(documents)})
                result = parsed if isinstance(parsed, GradeResult) else GradeResult.model_validate(parsed)
            except Exception as exc:
                logger.warning("grader_parse_failed", extra={"error": str(exc)})
                result = GradeResult(
                    relevant=False,
                    score=0.0,
                    reason="Grader output could not be parsed; treating context as insufficient.",
                )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        relevant_count = len(documents) if result.relevant else 0
        logger.info(
            "grade_complete",
            extra={
                "relevant": result.relevant,
                "relevance_score": result.score,
                "relevant_documents": relevant_count,
                "llm_latency_ms": elapsed_ms,
                "session_id": state.get("session_id"),
            },
        )
        return {
            "relevant": result.relevant,
            "relevance_score": result.score,
            "grade_reason": result.reason,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
            "metadata": {
                **(state.get("metadata") or {}),
                "relevant_documents": relevant_count,
                "grade_score": result.score,
            },
        }

    return grade
