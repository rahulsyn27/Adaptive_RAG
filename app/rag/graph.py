"""LangGraph workflow: route, retrieve, grade, rewrite, search, generate."""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

from app.core.constants import MessageRole
from app.core.exceptions import GraphExecutionError
from app.core.logging import get_logger
from app.database.database import get_default_session_factory
from app.database.repositories import add_message, get_recent_messages
from app.rag.nodes.generation import make_general_answer_node, make_generation_node
from app.rag.nodes.grading import make_grading_node
from app.rag.nodes.retrieval import make_retrieval_node
from app.rag.nodes.rewriting import make_rewriting_node
from app.rag.nodes.router import make_router_node
from app.rag.nodes.web_search import make_web_search_node
from app.rag.policies import after_grade, route_after_analysis
from app.rag.state import GraphState, HistoryTurn
from app.tools.search import TavilySearchTool
from app.vectorstore.retriever import DocumentRetriever

logger = get_logger(__name__)


def make_load_history_node(
    session_factory: sessionmaker[Session] | None = None,
) -> Callable[[GraphState], GraphState]:
    factory = session_factory or get_default_session_factory()

    def load_history(state: GraphState) -> GraphState:
        session_id = state["session_id"]
        db = factory()
        try:
            rows = get_recent_messages(db, session_id)
            history: list[HistoryTurn] = [
                {"role": row.role, "content": row.content} for row in rows
            ]
            return {"conversation_history": history}
        finally:
            db.close()

    return load_history


def make_save_response_node(
    session_factory: sessionmaker[Session] | None = None,
) -> Callable[[GraphState], GraphState]:
    factory = session_factory or get_default_session_factory()

    def save_response(state: GraphState) -> GraphState:
        logger.debug(
            "save_response_start",
            extra={"session_id": state.get("session_id"), "has_answer": bool(state.get("answer"))},
        )
        db = factory()
        try:
            add_message(db, state["session_id"], MessageRole.ASSISTANT.value, state.get("answer") or "")
            db.commit()
            logger.debug("save_response_success", extra={"session_id": state.get("session_id")})
        except Exception as exc:
            logger.error("save_response_failed", extra={"session_id": state.get("session_id"), "error": str(exc)})
            db.rollback()
            raise GraphExecutionError("Failed to persist the assistant response.") from exc
        finally:
            db.close()
        return {}

    return save_response


def build_graph(
    *,
    llm: BaseChatModel | None = None,
    retriever: DocumentRetriever | None = None,
    search_tool: TavilySearchTool | None = None,
    session_factory: sessionmaker[Session] | None = None,
):
    """Compile the adaptive RAG state machine with explicit conditional edges."""
    graph = StateGraph(GraphState)
    graph.add_node("load_history", make_load_history_node(session_factory))
    graph.add_node("analyze_query", make_router_node(llm))
    graph.add_node("retrieve", make_retrieval_node(retriever))
    graph.add_node("grade", make_grading_node(llm))
    graph.add_node("rewrite", make_rewriting_node(llm))
    graph.add_node("web_search", make_web_search_node(search_tool))
    graph.add_node("general_answer", make_general_answer_node(llm))
    graph.add_node("generate", make_generation_node(llm))
    graph.add_node("save_response", make_save_response_node(session_factory))

    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "INDEX": "retrieve",
            "GENERAL": "general_answer",
            "SEARCH": "web_search",
        },
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("general_answer", "generate")
    graph.add_edge("generate", "save_response")
    graph.add_edge("save_response", END)
    return graph.compile()
