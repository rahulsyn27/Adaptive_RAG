"""Simplified grading node."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.config.settings import Settings
from app.llm.groq import get_llm
from app.schemas.chat import GradeResult
from app.schemas.response import RetrievedChunk
from app.rag.nodes import make_node, _to_dict


def _format_chunks(chunks: list[RetrievedChunk] | None) -> str:
    if not chunks:
        return "(no documents retrieved)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        locator = chunk.filename or "document"
        if chunk.page is not None:
            locator = f"{locator} p.{chunk.page}"
        parts.append(f"[{i}] {locator}\n{chunk.content}")
    return "\n\n".join(parts)


def make_grader(llm: BaseChatModel | None = None) -> make_node:
    model = llm or get_llm()
    structured = model.with_structured_output(GradeResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt("grader", "system")),
        ("human", get_prompt("grader", "human")),
    ])
    chain = prompt | structured

    def run(state: dict[str, Any]) -> dict[str, Any]:
        documents = state.get("documents", [])

        if not documents:
            result = GradeResult(relevant=False, score=0.0, reason="No documents were retrieved.")
        else:
            try:
                parsed = chain.invoke({"query": state["query"], "context": _format_chunks(documents)})
                result = parsed if isinstance(parsed, GradeResult) else GradeResult.model_validate(parsed)
            except Exception:
                result = GradeResult(
                    relevant=False,
                    score=0.0,
                    reason="Grader output could not be parsed; treating context as insufficient.",
                )

        relevant_count = len(documents) if result.relevant else 0
        return {
            "relevant": result.relevant,
            "relevance_score": result.score,
            "grade_reason": result.reason,
            "metadata": {"relevant_documents": relevant_count, "grade_score": result.score},
        }

    return make_node(run, "grader")


def after_grade(state, max_attempts: int | Settings = 3) -> str:
    """Decide generate vs rewrite after grading."""
    state_dict = _to_dict(state)
    if state_dict.get("relevant"):
        return "generate"
    if isinstance(max_attempts, Settings):
        max_attempts = max_attempts.max_retrieval_attempts
    if state_dict.get("retrieval_attempts", 0) < max_attempts:
        return "rewrite"
    return "generate"
