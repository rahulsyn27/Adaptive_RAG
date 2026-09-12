"""Router node unit tests."""

from app.core.constants import Route
from app.rag.nodes.router import make_router_node
from tests.conftest import FakeChatModel


def test_router_selects_index() -> None:
    node = make_router_node(FakeChatModel(route=Route.INDEX))
    result = node({"query": "What does the uploaded PDF say about transformers?", "session_id": "s1"})
    assert result["route"] == Route.INDEX
    assert "rationale" in result["route_reason"].lower() or result["route_reason"]


def test_router_selects_general() -> None:
    node = make_router_node(FakeChatModel(route=Route.GENERAL))
    result = node({"query": "What is an embedding?", "session_id": "s1"})
    assert result["route"] == Route.GENERAL


def test_router_selects_search() -> None:
    node = make_router_node(FakeChatModel(route=Route.SEARCH))
    result = node({"query": "Latest developments in LangGraph?", "session_id": "s1"})
    assert result["route"] == Route.SEARCH
