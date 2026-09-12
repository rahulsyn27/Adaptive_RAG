"""INDEX-path retrieval from Qdrant."""

import time
from collections.abc import Callable

from app.core.logging import get_logger
from app.rag.state import GraphState
from app.vectorstore.retriever import DocumentRetriever, get_retriever

logger = get_logger(__name__)


def make_retrieval_node(
    retriever: DocumentRetriever | None = None,
) -> Callable[[GraphState], GraphState]:
    store = retriever or get_retriever()

    def retrieve(state: GraphState) -> GraphState:
        started = time.perf_counter()
        query = state.get("rewritten_query") or state["query"]
        attempts = int(state.get("retrieval_attempts") or 0) + 1
        documents = store.similarity_search(query)
        chunks = store.to_chunks(documents)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "retrieval_complete",
            extra={
                "retrieval_attempts": attempts,
                "retrieved_documents": len(chunks),
                "retrieval_latency_ms": elapsed_ms,
                "session_id": state.get("session_id"),
            },
        )
        return {
            "documents": chunks,
            "retrieval_attempts": attempts,
            "retrieval_latency_ms": int(state.get("retrieval_latency_ms") or 0) + elapsed_ms,
            "metadata": {
                **(state.get("metadata") or {}),
                "retrieved_documents": len(chunks),
                "active_query": query,
            },
        }

    return retrieve
