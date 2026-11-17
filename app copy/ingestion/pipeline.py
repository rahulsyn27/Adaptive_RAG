"""Upload → validate → load → split → embed → ChromaDB."""

from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.core.constants import SUPPORTED_DOCUMENT_EXTENSIONS
from app.core.exceptions import InvalidDocumentTypeError
from app.core.logging import get_logger
from app.database import repositories
from app.ingestion.loaders import load_file
from app.ingestion.metadata import attach_chunk_metadata
from app.ingestion.splitter import split_documents
from app.vectorstore.retriever import DocumentRetriever, get_retriever

logger = get_logger(__name__)


def validate_filename(filename: str) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise InvalidDocumentTypeError(filename)
    return Path(filename)


def ingest_file(
    *,
    path: Path,
    filename: str,
    description: str | None,
    db: Session,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
) -> tuple[str, int]:
    """Persist document metadata in SQLite and chunks in ChromaDB."""
    cfg = settings or get_settings()
    validate_filename(filename)
    document_id = str(uuid4())
    raw_docs = load_file(path)
    chunks = split_documents(raw_docs, cfg)
    enriched = attach_chunk_metadata(
        chunks,
        document_id=document_id,
        filename=filename,
        description=description,
    )
    store = retriever or get_retriever(cfg)
    count = store.upsert_documents(enriched)
    repositories.create_document_metadata(
        db,
        document_id=document_id,
        filename=filename,
        description=description,
        collection_name=cfg.chroma_collection,
    )
    logger.info(
        "ingested_document",
        extra={"document_id": document_id, "doc_filename": filename, "chunk_count": count},
    )
    return document_id, count
