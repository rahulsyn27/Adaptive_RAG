"""Shared ChromaDB client and collection bootstrap.

Uses persistent on-disk ChromaDB storage.
"""


import threading

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)


_chroma_client_instance = None
_chroma_lock = threading.Lock()


def get_chroma_client() -> chromadb.ClientAPI:
    """Return the process-wide ChromaDB client (persistent)."""
    global _chroma_client_instance

    with _chroma_lock:
        if _chroma_client_instance is not None:
            return _chroma_client_instance

        cfg = get_settings()
        chroma_settings = ChromaSettings(
            persist_directory=str(cfg.chroma_path),
            anonymized_telemetry=False,
        )
        _chroma_client_instance = chromadb.Client(chroma_settings)
        logger.info("using_chroma", extra={"path": str(cfg.chroma_path)})
        return _chroma_client_instance


def ensure_collection(client: chromadb.ClientAPI | None = None, settings: Settings | None = None) -> None:
    """Create the configured collection when it is missing."""
    cfg = settings or get_settings()
    chroma = client or get_chroma_client()
    try:
        collections = chroma.list_collections()
        names = [c.name for c in collections]
        if cfg.chroma_collection in names:
            return

        from app.embeddings.provider import embedding_dimension, get_embeddings

        size = embedding_dimension(get_embeddings())
        chroma.create_collection(
            name=cfg.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("created_chroma_collection", extra={"collection": cfg.chroma_collection, "size": size})
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("chroma_collection_failed", extra={"error": str(exc)})
        raise VectorStoreError(
            f"Vector store is unavailable. Check CHROMA_PATH: {cfg.chroma_path}"
        ) from exc


def check_chroma(settings: Settings | None = None) -> str:
    """Return a short ChromaDB status string for health checks."""
    _ = settings or get_settings()
    try:
        client = get_chroma_client()
        client.list_collections()
        return "ok"
    except Exception:
        return "error"
