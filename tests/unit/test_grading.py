"""Grading node and retry policy tests."""

from app.config.settings import Settings
from app.rag.nodes.grading import make_grading_node
from app.rag.policies import after_grade
from app.schemas.response import RetrievedChunk
from tests.conftest import FakeChatModel


def test_grader_marks_relevant() -> None:
    node = make_grading_node(FakeChatModel(relevant=True))
    result = node(
        {
            "query": "What does the paper say?",
            "documents": [RetrievedChunk(content="Transformers use attention.", filename="p.pdf")],
        }
    )
    assert result["relevant"] is True
    assert result["relevance_score"] >= 0.5


def test_grader_empty_documents_not_relevant() -> None:
    node = make_grading_node(FakeChatModel(relevant=True))
    result = node({"query": "missing", "documents": []})
    assert result["relevant"] is False
    assert result["relevance_score"] == 0.0


def test_after_grade_rewrites_until_max_attempts() -> None:
    settings = Settings(max_retrieval_attempts=2)
    assert after_grade({"relevant": False, "retrieval_attempts": 1}, settings) == "rewrite"
    assert after_grade({"relevant": False, "retrieval_attempts": 2}, settings) == "generate"
    assert after_grade({"relevant": True, "retrieval_attempts": 1}, settings) == "generate"
