"""SQLite repository tests."""

from sqlalchemy.orm import Session

from app.database import repositories


def test_create_session_and_messages(db_session: Session) -> None:
    conversation = repositories.create_session(db_session, "sess-1")
    assert conversation.session_id == "sess-1"
    repositories.add_message(db_session, "sess-1", "user", "hello")
    repositories.add_message(db_session, "sess-1", "assistant", "hi")
    rows = repositories.get_recent_messages(db_session, "sess-1")
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[0].content == "hello"


def test_document_metadata_roundtrip(db_session: Session) -> None:
    repositories.create_document_metadata(
        db_session,
        document_id="doc-1",
        filename="paper.pdf",
        description="attention paper",
        collection_name="adaptive_rag",
    )
    listed = repositories.list_documents(db_session)
    assert listed[0].filename == "paper.pdf"
    assert repositories.get_document(db_session, "doc-1") is not None
    assert repositories.get_session(db_session, "missing") is None
