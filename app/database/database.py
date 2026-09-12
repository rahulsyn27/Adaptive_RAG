"""SQLAlchemy engine and session factory."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config.settings import Settings, get_settings
from app.database.models import Base


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
    module = type(dbapi_connection).__module__
    if "sqlite3" not in module:
        return
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _database_url() -> str:
    return get_settings().database_url


@lru_cache(maxsize=8)
def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or _database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url.startswith("sqlite") and ":memory:" in url:
        return create_engine(
            url,
            future=True,
            connect_args=connect_args,
            poolclass=StaticPool,
        )
    return create_engine(url, future=True, connect_args=connect_args)


@lru_cache(maxsize=1)
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), autoflush=False, autocommit=False, future=True)


def get_default_session_factory() -> sessionmaker[Session]:
    """Get the default session factory using settings database URL."""
    return get_session_factory(_database_url())


def init_db(settings: Settings | None = None) -> None:
    """Create SQLite tables if they do not exist."""
    cfg = settings or get_settings()
    engine = get_engine(cfg.database_url)
    Base.metadata.create_all(bind=engine)


def session_scope() -> Generator[Session, None, None]:
    session = get_default_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
