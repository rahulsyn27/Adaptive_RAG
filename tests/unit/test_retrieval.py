"""Retrieval node tests with a fake vector store."""

from langchain_core.documents import Document

from app.rag.nodes.retrieval import make_retrieval_node
from app.schemas.response import RetrievedChunk


class _FakeStore:
    def similarity_search(self, query: str, top_k: int | None = None, document_id: str | None = None) -> list[Document]:
        assert query
        return [
            Document(
                page_content="Self-attention is the core of the Transformer.",
                metadata={
                    "filename": "paper.pdf",
                    "page": 4,
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "score": 0.91,
                    "source": "paper.pdf",
                    "description": "demo",
                },
            )
        ]

    def to_chunks(self, documents: list[Document]) -> list[RetrievedChunk]:
        from app.vectorstore.retriever import DocumentRetriever

        helper = DocumentRetriever.__new__(DocumentRetriever)
        return DocumentRetriever.to_chunks(helper, documents)


def test_retrieval_increments_attempts_and_stores_chunks() -> None:
    node = make_retrieval_node(_FakeStore())  # type: ignore[arg-type]
    result = node({"query": "transformers", "session_id": "s", "retrieval_attempts": 0})
    assert result["retrieval_attempts"] == 1
    assert len(result["documents"]) == 1
    assert result["documents"][0].filename == "paper.pdf"
    assert result["documents"][0].page == 4
