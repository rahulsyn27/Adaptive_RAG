"""Document upload and listing schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    description: str | None = None
    collection_name: str
    created_at: datetime
    chunk_count: int | None = None


class DocumentUploadResponse(BaseModel):
    document: DocumentMetadata
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata] = Field(default_factory=list)
