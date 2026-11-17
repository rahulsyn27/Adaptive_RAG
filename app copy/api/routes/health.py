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


"""
==========================================
WHAT IS A 'ROUTER' IN FASTAPI?
==========================================
EXPLICIT BREAKDOWN:

In FastAPI, a 'router' (specifically an APIRouter instance) acts as a "mini-application" or a sub-module.

1. THE PROBLEM IT SOLVES:
   If you put every single endpoint (upload document, get document, chat, health check, etc.) 
   into your main entrypoint file, that file would quickly become thousands of lines long 
   and impossible to manage.

2. WHAT IT EXACTLY DOES:
   A router allows you to group related endpoints together and store them in separate files. 
   For example, you have separate files for:
   - health.py (contains health check routes)
   - documents.py (contains document ingestion routes)
   - chat.py (contains chat routes)

   Inside 'health.py', instead of attaching the route directly to the main app like `@app.get(...)`, 
   you attach it to the router using `@router.get(...)`. The router acts as a basket, collecting 
   all the endpoints defined in that specific file.

3. HOW IT CONNECTS BACK TO THE MAIN APP:
   Remember these lines from your main entrypoint file?
   
   application.include_router(health.router, prefix=API_V1_PREFIX)
   application.include_router(documents.router, prefix=API_V1_PREFIX)

   The main application takes the "basket" (the router) full of endpoints from your separate files 
   and plugs them into the main server. 

ANALOGY:
Think of the main `app` as a large office building, and the `router` as a specific department 
(like the Document Processing Department). The router organizes all the tasks for that department 
in one room, and the main application just provides the hallway sign (`prefix=API_V1_PREFIX`) 
to point users to that room.
"""