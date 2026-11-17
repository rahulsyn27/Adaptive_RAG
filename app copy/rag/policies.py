"""Routing and retry policies used by graph conditional edges."""

from app.config.settings import Settings, get_settings
from app.core.constants import Route
from app.rag.state import GraphState


def normalize_route(value: object) -> Route:
    try:
        return Route(str(value))
    except ValueError:
        return Route.GENERAL


def route_after_analysis(state: GraphState) -> str:
    """Map router output onto INDEX / GENERAL / SEARCH branches."""
    return normalize_route(state.get("route")).value


def after_grade(state: GraphState, settings: Settings | None = None) -> str:
    """Decide generate vs rewrite after relevance grading."""
    cfg = settings or get_settings()
    relevant = bool(state.get("relevant"))
    attempts = int(state.get("retrieval_attempts") or 0)
    if relevant:
        return "generate"
    if attempts < cfg.max_retrieval_attempts:
        return "rewrite"
    return "generate"
