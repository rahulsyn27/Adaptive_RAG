"""Simplified router node."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.constants import Route
from app.llm.groq import get_llm
from app.schemas.chat import RouteDecision
from app.rag.nodes import make_node, _to_dict


def make_router(llm: BaseChatModel | None = None) -> make_node:
    model = llm or get_llm()
    structured = model.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt("router", "system")),
        ("human", get_prompt("router", "human")),
    ])
    chain = prompt | structured

    def run(state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        history = state.get("conversation_history", [])
        history_str = "\n".join(f"{h['role']}: {h['content']}" for h in history[-8:]) or "(none)"

        try:
            decision = chain.invoke({"query": query, "history": history_str})
        except Exception:
            decision = RouteDecision(
                route=Route.GENERAL,
                reason="Router output could not be parsed; defaulting to general knowledge.",
            )

        if not isinstance(decision, RouteDecision):
            decision = RouteDecision.model_validate(decision)

        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "metadata": {"route_reason": decision.reason},
        }

    return make_node(run, "router")


def route_after_analysis(state) -> str:
    """Map router output to branch."""
    state_dict = _to_dict(state)
    route = state_dict.get("route", Route.GENERAL)
    if hasattr(route, "value"):
        return route.value
    return str(route)
