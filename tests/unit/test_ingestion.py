"""Ingestion loader and splitter tests."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config.settings import Settings
from app.core.exceptions import EmptyDocumentError, InvalidDocumentTypeError
from app.ingestion.loaders import load_file
from app.ingestion.metadata import attach_chunk_metadata
from app.ingestion.pipeline import validate_filename
from app.ingestion.splitter import split_documents


def test_validate_filename_rejects_exe() -> None:
    with pytest.raises(InvalidDocumentTypeError):
        validate_filename("malware.exe")


def test_load_text_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Adaptive RAG routes queries before retrieval.", encoding="utf-8")
    docs = load_file(path)
    assert docs[0].page_content.startswith("Adaptive RAG")


def test_empty_text_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_file(path)


def test_split_and_metadata() -> None:
    settings = Settings(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content="alpha " * 80, metadata={"source": "a.txt", "page": 1})]
    chunks = split_documents(docs, settings)
    enriched = attach_chunk_metadata(chunks, document_id="d1", filename="a.txt", description="demo")
    assert enriched
    assert enriched[0].metadata["document_id"] == "d1"
    assert enriched[0].metadata["filename"] == "a.txt"
    assert enriched[0].metadata["chunk_id"]
