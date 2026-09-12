"""Query rewriting tests."""

from app.core.constants import Route
from app.rag.nodes.rewriting import make_rewriting_node
from app.schemas.response import RetrievedChunk
from tests.conftest import FakeChatModel


def test_rewrite_preserves_callable_contract() -> None:
    node = make_rewriting_node(
        FakeChatModel(route=Route.INDEX, relevant=False, rewrite="transformer architecture efficiency")
    )
    result = node(
        {
            "query": "What does the paper say about transformer efficiency?",
            "documents": [
                RetrievedChunk(content="unrelated weather notes", filename="paper.pdf", page=1)
            ],
        }
    )
    assert "transformer" in result["rewritten_query"].lower()
    assert result["metadata"]["rewritten_query"]
