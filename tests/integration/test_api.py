"""FastAPI integration tests with mocked graph and ingestion."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import db_session_dep
from app.core.constants import Route
from app.database.models import Base
from app.main import create_app
from app.schemas.response import Source


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _override():
        session = factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[db_session_dep] = _override
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "ok"
    assert "version" in body


def test_chat_validation(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "", "session_id": "s"})
    assert response.status_code == 422


def test_general_chat(client: TestClient) -> None:
    class FakeGraph:
        async def ainvoke(self, state: dict) -> dict:
            return {
                "answer": "Retrieval augmented generation combines search with an LLM.",
                "route": Route.GENERAL,
                "sources": [],
                "retrieval_attempts": 0,
                "llm_latency_ms": 3,
                "retrieval_latency_ms": 0,
                "metadata": {},
            }

    with patch("app.api.routes.chat.get_compiled_graph", return_value=FakeGraph()):
        response = client.post(
            "/api/v1/chat",
            json={"query": "What is retrieval augmented generation?", "session_id": "s1"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "GENERAL"
    assert "retrieval" in body["answer"].lower()


def test_search_chat_returns_sources(client: TestClient) -> None:
    class FakeGraph:
        async def ainvoke(self, state: dict) -> dict:
            return {
                "answer": "Recent RAG work focuses on adaptive routing.",
                "route": Route.SEARCH,
                "sources": [Source(source_type="web", title="Blog", url="https://example.com/rag")],
                "retrieval_attempts": 0,
                "metadata": {},
            }

    with patch("app.api.routes.chat.get_compiled_graph", return_value=FakeGraph()):
        response = client.post(
            "/api/v1/chat",
            json={"query": "What are the latest developments in RAG?", "session_id": "s2"},
        )
    assert response.status_code == 200
    assert response.json()["sources"][0]["url"] == "https://example.com/rag"


def test_document_upload_and_list(client: TestClient, tmp_path: Path) -> None:
    def _fake_ingest(**kwargs):
        from app.database import repositories

        repositories.create_document_metadata(
            kwargs["db"],
            document_id="doc-99",
            filename=kwargs["filename"],
            description=kwargs["description"],
            collection_name="adaptive_rag",
        )
        return "doc-99", 3

    sample = tmp_path / "notes.txt"
    sample.write_text("hello rag", encoding="utf-8")
    with patch("app.api.routes.documents.ingest_file", side_effect=_fake_ingest):
        with sample.open("rb") as handle:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("notes.txt", handle, "text/plain")},
                data={"description": "test notes"},
            )
    assert response.status_code == 201
    assert response.json()["chunk_count"] == 3
    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["filename"] == "notes.txt"


def test_rejected_extension(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("payload.exe", b"xx", "application/octet-stream")},
    )
    assert response.status_code == 400
