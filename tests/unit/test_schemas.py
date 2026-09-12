"""API and Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.core.constants import Route, SourceType
from app.schemas.chat import ChatRequest, RouteDecision
from app.schemas.response import Source


def test_chat_request_requires_query() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(query="", session_id="abc")


def test_chat_request_ok() -> None:
    payload = ChatRequest(query="What is RAG?", session_id="abc")
    assert payload.session_id == "abc"


def test_route_decision_enum() -> None:
    decision = RouteDecision(route=Route.SEARCH, reason="Needs live data.")
    assert decision.route is Route.SEARCH


def test_source_document_shape() -> None:
    source = Source(source_type=SourceType.DOCUMENT, filename="paper.pdf", page=5)
    dumped = source.model_dump()
    assert dumped["source_type"] == "document"
    assert dumped["filename"] == "paper.pdf"
