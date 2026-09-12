"""Liveness and dependency health checks."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.dependencies import db_session_dep, settings_dep
from app.config.settings import Settings
from app.core.constants import APP_VERSION
from app.schemas.response import HealthResponse
from app.vectorstore.client import check_chroma

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health(
    settings: Settings = Depends(settings_dep),
    db: Session = Depends(db_session_dep),
) -> HealthResponse:
    """Report API, database, and ChromaDB availability."""
    database_status = "ok"
    chroma_status = "unknown"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_status = "error"
    try:
        chroma_status = "ok" if check_chroma(settings) else "error"
    except Exception:
        chroma_status = "error"
    overall = "ok" if database_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        version=APP_VERSION,
        chroma=chroma_status,
        database=database_status,
    )
