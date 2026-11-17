"""Simplified graph builder."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config.settings import get_settings
from app.core.constants import MessageRole
from app.core.exceptions import GraphExecutionError
from app.core.logging import get_logger
from app.database.database import get_default_session_factory
from app.database.repositories import add_message, get_recent_messages
from app.rag.router import make_router, route_after_analysis
from app.rag.retrieval import make_retrieval
from app.rag.grading import make_grader, after_grade
from app.rag.rewriting import make_rewriter
from app.rag.web_search import make_web_search
from app.rag.generation import make_general_answer, make_generator
from app.rag.state import GraphState
from app.tools.search import TavilySearchTool
from app.vectorstore.retriever import DocumentRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session, sessionmaker

logger = get_logger(__name__)


def _to_dict(state) -> dict[str, Any]:
    """Convert GraphState or dict to dict."""
    if hasattr(state, 'to_dict'):
        return state.to_dict()
    return state


def make_load_history(session_factory: sessionmaker[Session] | None = None):
    factory = session_factory or get_default_session_factory()

    def run(state) -> dict[str, Any]:
        state_dict = _to_dict(state)
        session_id = state_dict["session_id"]
        db = factory()
        try:
            rows = get_recent_messages(db, session_id)
            history = [{"role": row.role, "content": row.content} for row in rows]
            return {"conversation_history": history}
        finally:
            db.close()

    return run


def make_save_response(session_factory: sessionmaker[Session] | None = None):
    factory = session_factory or get_default_session_factory()

    def run(state) -> dict[str, Any]:
        state_dict = _to_dict(state)
        session_id = state_dict.get("session_id")
        answer = state_dict.get("answer", "")
        db = factory()
        try:
            add_message(db, session_id, MessageRole.ASSISTANT.value, answer)
            db.commit()
        except Exception as exc:
            db.rollback()
            raise GraphExecutionError("Failed to persist the assistant response.") from exc
        finally:
            db.close()
        return {}

    return run


def build_simple_graph(
    *,
    llm: BaseChatModel | None = None,
    retriever: DocumentRetriever | None = None,
    search_tool: TavilySearchTool | None = None,
    session_factory: sessionmaker[Session] | None = None,
    max_retrieval_attempts: int | None = None,
):
    """Build the simplified adaptive RAG graph."""
    if max_retrieval_attempts is None:
        max_retrieval_attempts = get_settings().max_retrieval_attempts
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("load_history", make_load_history(session_factory))
    graph.add_node("analyze_query", make_router(llm))
    graph.add_node("retrieve", make_retrieval(retriever))
    graph.add_node("grade", make_grader(llm))
    graph.add_node("rewrite", make_rewriter(llm))
    graph.add_node("web_search", make_web_search(search_tool))
    graph.add_node("general_answer", make_general_answer(llm))
    graph.add_node("generate", make_generator(llm))
    graph.add_node("save_response", make_save_response(session_factory))

    # Edges
    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "analyze_query")
    graph.add_conditional_edges("analyze_query", route_after_analysis, {
        "INDEX": "retrieve",
        "GENERAL": "general_answer",
        "SEARCH": "web_search",
    })
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", lambda s: after_grade(s, max_retrieval_attempts), {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("general_answer", "generate")
    graph.add_edge("generate", "save_response")
    graph.add_edge("save_response", END)

    return graph.compile()
