"""Document upload and listing."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import db_session_dep, settings_dep
from app.config.settings import Settings
from app.core.exceptions import AdaptiveRAGError, to_http_exception
from app.core.logging import get_logger
from app.database import repositories
from app.ingestion.pipeline import ingest_file, validate_filename
from app.schemas.document import DocumentListResponse, DocumentMetadata, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    session_id: str = Form(...),
    db: Session = Depends(db_session_dep),
    settings: Settings = Depends(settings_dep),
) -> DocumentUploadResponse:
    filename = file.filename or "upload.bin"
    try:
        validate_filename(filename)
    except AdaptiveRAGError as exc:
        raise to_http_exception(exc) from exc

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the upload limit.",
        )

    destination = settings.upload_dir / f"{uuid4().hex}_{Path(filename).name}"
    destination.write_bytes(data)
    try:
        document_id, chunk_count = ingest_file(
            path=destination,
            filename=filename,
            description=description,
            db=db,
            settings=settings,
            session_id=session_id,
        )
    except AdaptiveRAGError as exc:
        destination.unlink(missing_ok=True)
        raise to_http_exception(exc) from exc

    row = repositories.get_document(db, document_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Document metadata was not stored.")
    logger.info("document_uploaded", extra={"document_id": document_id, "chunk_count": chunk_count, "session_id": session_id})
    return DocumentUploadResponse(
        document=DocumentMetadata(
            id=row.id,
            filename=row.filename,
            description=row.description,
            collection_name=row.collection_name,
            created_at=row.created_at,
            chunk_count=chunk_count,
        ),
        chunk_count=chunk_count,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    session_id: str | None = None,
    db: Session = Depends(db_session_dep),
) -> DocumentListResponse:
    rows = repositories.list_documents(db, session_id=session_id)
    return DocumentListResponse(
        documents=[
            DocumentMetadata(
                id=row.id,
                filename=row.filename,
                description=row.description,
                collection_name=row.collection_name,
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


@router.get("/sessions", response_model=list[dict])
async def list_sessions_with_documents(db: Session = Depends(db_session_dep)) -> list[dict]:
    """List all sessions that have documents, for sidebar display."""
    return repositories.list_sessions_with_documents(db)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, db: Session = Depends(db_session_dep)):
    deleted = repositories.delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    db.commit()


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session_documents(session_id: str, db: Session = Depends(db_session_dep)):
    count = repositories.delete_session_documents(db, session_id)
    db.commit()
    return {"deleted_count": count}
