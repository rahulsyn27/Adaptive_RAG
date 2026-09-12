"""HTTP dependencies."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.database import get_session_factory


def settings_dep() -> Settings:
    return get_settings()


def db_session_dep() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
