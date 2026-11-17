"""SEARCH-path web lookup."""

from collections.abc import Callable

from app.core.exceptions import SearchToolError
from app.core.logging import get_logger
from app.rag.state import GraphState
from app.tools.search import TavilySearchTool, get_search_tool

logger = get_logger(__name__)


def make_web_search_node(search_tool: TavilySearchTool | None = None) -> Callable[[GraphState], GraphState]:
    tool = search_tool or get_search_tool()

    def web_search(state: GraphState) -> GraphState:
        query = state.get("rewritten_query") or state["query"]
        try:
            results = tool.search(query)
        except SearchToolError:
            raise
        logger.info(
            "web_search_complete",
            extra={"result_count": len(results), "session_id": state.get("session_id")},
        )
        return {
            "web_results": results,
            "metadata": {**(state.get("metadata") or {}), "web_results": len(results)},
        }

    return web_search
