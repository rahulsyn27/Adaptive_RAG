"""Chat request and graph-facing schemas."""

from pydantic import BaseModel, Field

from app.core.constants import Route
from app.schemas.response import Source


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=128)


class ChatMetadata(BaseModel):
    retrieval_attempts: int = 0
    retrieved_documents: int = 0
    relevant_documents: int = 0
    latency_ms: int = 0
    retrieval_latency_ms: int = 0
    llm_latency_ms: int = 0
    rewritten_query: str | None = None


class ChatResponse(BaseModel):
    answer: str
    route: Route
    sources: list[Source] = Field(default_factory=list)
    metadata: ChatMetadata
    session_id: str


class RouteDecision(BaseModel):
    """Structured router output. `reason` is a short classification rationale."""

    route: Route
    reason: str = Field(..., max_length=280)


class GradeResult(BaseModel):
    relevant: bool
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., max_length=280)
