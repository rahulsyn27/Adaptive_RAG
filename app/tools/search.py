"""Tavily web search through a LangChain-compatible wrapper."""

from typing import Any, Protocol

from app.config.settings import Settings, get_settings
from app.core.exceptions import ConfigurationError, SearchToolError
from app.schemas.response import WebResult


class SearchBackend(Protocol):
    def invoke(self, query: str) -> list[dict[str, Any]]: ...


class TavilySearchTool:
    """Thin adapter over Tavily that returns typed web results."""

    def __init__(self, settings: Settings | None = None, backend: SearchBackend | None = None) -> None:
        self.settings = settings or get_settings()
        self._backend = backend

    def _client(self) -> SearchBackend:
        if self._backend is not None:
            return self._backend
        if not self.settings.tavily_api_key:
            raise ConfigurationError("TAVILY_API_KEY is not configured.")
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            try:
                return TavilySearchResults(max_results=5, tavily_api_key=self.settings.tavily_api_key)
            except TypeError:
                return TavilySearchResults(max_results=5, api_key=self.settings.tavily_api_key)
        except Exception as exc:
            raise SearchToolError("Unable to initialize the Tavily search tool.") from exc

    def search(self, query: str) -> list[WebResult]:
        try:
            raw = self._client().invoke(query)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise SearchToolError("Web search failed.") from exc
        results: list[WebResult] = []
        if isinstance(raw, str):
            return results
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            title = str(item.get("title") or item.get("url") or "Untitled")
            content = str(item.get("content") or item.get("snippet") or "")
            if not url:
                continue
            score = item.get("score")
            results.append(
                WebResult(
                    title=title,
                    url=url,
                    content=content,
                    score=float(score) if score is not None else None,
                )
            )
        return results


def get_search_tool(settings: Settings | None = None, backend: SearchBackend | None = None) -> TavilySearchTool:
    return TavilySearchTool(settings=settings, backend=backend)
