"""Search tool adapter tests."""

from app.schemas.response import WebResult
from app.tools.search import TavilySearchTool


class _Backend:
    def invoke(self, query: str) -> list[dict[str, object]]:
        assert "LangGraph" in query
        return [{"title": "Docs", "url": "https://example.com", "content": "release notes", "score": 0.7}]


def test_tavily_adapter_maps_results() -> None:
    tool = TavilySearchTool(backend=_Backend())
    results = tool.search("LangGraph latest")
    assert results == [
        WebResult(title="Docs", url="https://example.com", content="release notes", score=0.7)
    ]
