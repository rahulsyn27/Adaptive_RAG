"""Simplified rewriting node."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.llm.groq import get_llm
from app.rag.grading import _format_chunks
from app.rag.nodes import make_node


def make_rewriter(llm: BaseChatModel | None = None) -> make_node:
    model = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt("rewriter", "system")),
        ("human", get_prompt("rewriter", "human")),
    ])
    chain = prompt | model

    def run(state: dict[str, Any]) -> dict[str, Any]:
        original = state["query"]
        try:
            raw = chain.invoke({"query": original, "context": _format_chunks(state.get("documents"))})
            rewritten = str(raw).strip() or original
        except Exception:
            rewritten = original

        return {
            "rewritten_query": rewritten,
            "metadata": {"rewritten_query": rewritten},
        }

    return make_node(run, "rewriter")
