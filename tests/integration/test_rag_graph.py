"""LangGraph path integration tests with fakes (no live APIs)."""

from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.constants import Route
from app.database.models import Base
from app.database.repositories import add_message, create_session
from app.rag.graph import build_graph
from app.schemas.response import RetrievedChunk, WebResult
from tests.conftest import FakeChatModel


class FakeRetriever:
    def similarity_search(self, query: str, top_k: int | None = None, document_id: str | None = None):
        return [
            Document(
                page_content="The paper states that transformer efficiency improves with sparse attention.",
                metadata={
                    "filename": "paper.pdf",
                    "page": 7,
                    "document_id": "doc-1",
                    "chunk_id": "11111111-1111-1111-1111-111111111111",
                    "score": 0.88,
                    "source": "paper.pdf",
                    "description": "demo paper",
                },
            )
        ]

    def to_chunks(self, documents: list[Document]) -> list[RetrievedChunk]:
        from app.vectorstore.retriever import DocumentRetriever

        helper = DocumentRetriever.__new__(DocumentRetriever)
        return DocumentRetriever.to_chunks(helper, documents)


class FakeSearch:
    def search(self, query: str) -> list[WebResult]:
        return [
            WebResult(
                title="LangGraph release notes",
                url="https://example.com/langgraph",
                content="New checkpointing APIs were announced.",
                score=0.8,
            )
        ]


def _factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_general_query_skips_retrieval() -> None:
    factory = _factory()
    db = factory()
    create_session(db, "mem-1")
    add_message(db, "mem-1", "user", "hello")
    db.commit()
    db.close()
    graph = build_graph(
        llm=FakeChatModel(route=Route.GENERAL, answer="An embedding is a vector representation of text."),
        retriever=FakeRetriever(),
        search_tool=FakeSearch(),
        session_factory=factory,
    )
    result = graph.invoke({"query": "What is an embedding?", "session_id": "mem-1", "retrieval_attempts": 0})
    assert result["route"] == Route.GENERAL
    assert "vector" in result["answer"].lower()
    assert result.get("retrieval_attempts", 0) == 0
    assert result.get("web_results") in (None, [])


def test_index_query_retrieves_and_cites() -> None:
    factory = _factory()
    db = factory()
    create_session(db, "idx-1")
    db.commit()
    db.close()
    graph = build_graph(
        llm=FakeChatModel(route=Route.INDEX, relevant=True, answer="The paper discusses sparse attention."),
        retriever=FakeRetriever(),
        search_tool=FakeSearch(),
        session_factory=factory,
    )
    result = graph.invoke({"query": "What does the uploaded paper say about efficiency?", "session_id": "idx-1"})
    assert result["route"] == Route.INDEX
    assert result["retrieval_attempts"] == 1
    assert result["sources"]
    assert result["sources"][0].filename == "paper.pdf"


def test_search_query_uses_web_results() -> None:
    factory = _factory()
    db = factory()
    create_session(db, "web-1")
    db.commit()
    db.close()
    graph = build_graph(
        llm=FakeChatModel(route=Route.SEARCH, answer="LangGraph added checkpointing APIs."),
        retriever=FakeRetriever(),
        search_tool=FakeSearch(),
        session_factory=factory,
    )
    result = graph.invoke({"query": "What are the latest developments in LangGraph?", "session_id": "web-1"})
    assert result["route"] == Route.SEARCH
    assert result["web_results"]
    assert result["sources"][0].url == "https://example.com/langgraph"


def test_irrelevant_retrieval_rewrites_then_stops() -> None:
    factory = _factory()
    db = factory()
    create_session(db, "retry-1")
    db.commit()
    db.close()
    graph = build_graph(
        llm=FakeChatModel(route=Route.INDEX, relevant=False, answer="The indexed documents do not cover this."),
        retriever=FakeRetriever(),
        search_tool=FakeSearch(),
        session_factory=factory,
    )
    result = graph.invoke({"query": "Martian tax law in my report?", "session_id": "retry-1"})
    assert result["retrieval_attempts"] == 2
    assert result.get("rewritten_query")
