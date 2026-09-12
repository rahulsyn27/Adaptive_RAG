"""Query rewriting after insufficient retrieval."""

import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.logging import get_logger
from app.llm.groq import get_llm
from app.rag.nodes.grading import _format_chunks
from app.rag.state import GraphState

logger = get_logger(__name__)


def _message_text(message: BaseMessage | str) -> str:
    if isinstance(message, str):
        return message.strip()
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def make_rewriting_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("rewriter", "system")),
            ("human", get_prompt("rewriter", "human")),
        ]
    )
    chain = prompt | model

    def rewrite(state: GraphState) -> GraphState:
        started = time.perf_counter()
        original = state["query"]
        try:
            raw = chain.invoke({"query": original, "context": _format_chunks(state.get("documents"))})
            rewritten = _message_text(raw) or original
        except Exception as exc:
            logger.warning("rewrite_failed", extra={"error": str(exc)})
            rewritten = original
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "query_rewritten",
            extra={"retrieval_attempts": state.get("retrieval_attempts"), "llm_latency_ms": elapsed_ms},
        )
        return {
            "rewritten_query": rewritten,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
            "metadata": {**(state.get("metadata") or {}), "rewritten_query": rewritten},
        }

    return rewrite
