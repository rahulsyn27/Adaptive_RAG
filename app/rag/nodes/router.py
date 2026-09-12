"""Query analyzer that selects INDEX, GENERAL, or SEARCH."""

import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.constants import Route
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.groq import get_llm
from app.rag.state import GraphState, HistoryTurn
from app.schemas.chat import RouteDecision

logger = get_logger(__name__)


def _format_history(turns: list[HistoryTurn] | None) -> str:
    if not turns:
        return "(none)"
    return "\n".join(f"{item['role']}: {item['content']}" for item in turns[-8:])


def make_router_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    """Build a testable router node using structured LLM output."""

    model = llm or get_llm()
    structured = model.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("router", "system")),
            ("human", get_prompt("router", "human")),
        ]
    )
    chain = prompt | structured

    def analyze_query(state: GraphState) -> GraphState:
        started = time.perf_counter()
        query = state["query"]
        try:
            decision = chain.invoke({"query": query, "history": _format_history(state.get("conversation_history"))})
        except Exception as exc:
            logger.warning("router_structured_output_failed", extra={"error": str(exc)})
            decision = RouteDecision(
                route=Route.GENERAL,
                reason="Router output could not be parsed; defaulting to general knowledge.",
            )
        if not isinstance(decision, RouteDecision):
            try:
                decision = RouteDecision.model_validate(decision)
            except Exception as exc:
                raise LLMError("Router returned malformed structured output.") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "route_selected",
            extra={"route": decision.route.value, "llm_latency_ms": elapsed_ms, "session_id": state.get("session_id")},
        )
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
            "metadata": {**(state.get("metadata") or {}), "route_reason": decision.reason},
        }

    return analyze_query
