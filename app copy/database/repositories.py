"""Persistence helpers for conversations, messages, and documents."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DatabaseError
from app.database.models import Conversation, Document, Message


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def create_session(db: Session, session_id: str) -> Conversation:
    existing = get_session(db, session_id)
    if existing is not None:
        return existing
    try:
        conversation = Conversation(session_id=session_id)
        db.add(conversation)
        db.flush()
        return conversation
    except Exception as exc:
        raise DatabaseError("Unable to create conversation session.") from exc


def get_session(db: Session, session_id: str) -> Conversation | None:
    return db.scalar(select(Conversation).where(Conversation.session_id == session_id))


def add_message(db: Session, session_id: str, role: str, content: str) -> Message:
    try:
        conversation = create_session(db, session_id)
        conversation.updated_at = _utcnow()
        created_at = _utcnow()
        latest = db.scalar(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        if latest is not None and latest.created_at >= created_at:
            created_at = latest.created_at + timedelta(microseconds=1000)
        message = Message(session_id=session_id, role=role, content=content, created_at=created_at)
        db.add(message)
        db.flush()
        return message
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError("Unable to store chat message.") from exc


def get_recent_messages(db: Session, session_id: str, limit: int = 12) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(db.scalars(stmt))
    return list(reversed(rows))


def create_document_metadata(
    db: Session,
    *,
    document_id: str,
    filename: str,
    description: str | None,
    collection_name: str,
) -> Document:
    try:
        document = Document(
            id=document_id,
            filename=filename,
            description=description,
            collection_name=collection_name,
        )
        db.add(document)
        db.flush()
        return document
    except Exception as exc:
        raise DatabaseError("Unable to store document metadata.") from exc


def list_documents(db: Session) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc())
    return list(db.scalars(stmt))


def get_document(db: Session, document_id: str) -> Document | None:
    return db.get(Document, document_id)
