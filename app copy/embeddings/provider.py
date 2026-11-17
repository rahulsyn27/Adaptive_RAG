"""Embedding model factory."""

from functools import lru_cache
from typing import Any

from app.config.settings import get_settings


@lru_cache(maxsize=1)
def get_embeddings(model_name: str | None = None, device: str | None = None) -> Any:
    """Return a local HuggingFace embedding model (BGE by default)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = get_settings()
    name = model_name or cfg.embedding_model
    chosen_device = device or cfg.embedding_device
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": chosen_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embeddings: Any | None = None) -> int:
    """Probe embedding size so Qdrant collections are created correctly."""
    model = embeddings or get_embeddings()
    vector = model.embed_query("dimension probe")
    return len(vector)
