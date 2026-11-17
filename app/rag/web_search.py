"""Simplified web search node."""

from typing import Any

from app.tools.search import TavilySearchTool, get_search_tool
from app.rag.nodes import make_node


def make_web_search(search_tool: TavilySearchTool | None = None) -> make_node:
    tool = search_tool or get_search_tool()

    def run(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        results = tool.search(query)
        return {
            "web_results": results,
            "metadata": {"web_results": len(results)},
        }

    return make_node(run, "web_search")
