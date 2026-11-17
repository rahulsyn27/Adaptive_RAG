"""Shared response models including source citations."""

from pydantic import BaseModel

from app.core.constants import SourceType


class Source(BaseModel):
    title: str | None = None
    source_type: SourceType
    url: str | None = None
    document_id: str | None = None
    filename: str | None = None
    page: int | None = None
    relevance_score: float | None = None
    chunk_id: str | None = None


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    version: str
    chroma: str = "unknown"
    database: str = "unknown"


class WebResult(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = None


class RetrievedChunk(BaseModel):
    content: str
    document_id: str | None = None
    filename: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    source: str | None = None
    description: str | None = None
    score: float | None = None
