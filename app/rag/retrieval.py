"""Simplified retrieval node."""

from collections.abc import Callable
from typing import Any

from app.vectorstore.retriever import DocumentRetriever, get_retriever
from app.rag.nodes import make_node


def make_retrieval(retriever: DocumentRetriever | None = None) -> Callable:
    store = retriever or get_retriever()

    def run(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        attempts = state.get("retrieval_attempts", 0) + 1
        session_id = state.get("session_id")

        documents = store.similarity_search(query, session_id=session_id)
        chunks = store.to_chunks(documents)

        return {
            "documents": chunks,
            "retrieval_attempts": attempts,
            "retrieval_latency_ms": 0,  # timing handled by base
            "metadata": {"retrieved_documents": len(chunks), "active_query": query},
        }

    return make_node(run, "retrieval")
