"""CLI helper to ingest a local file into Qdrant + SQLite."""

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.core.logging import configure_logging
from app.database.database import get_session_factory, init_db
from app.ingestion.pipeline import ingest_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into AdaptiveRAG")
    parser.add_argument("path", type=Path)
    parser.add_argument("--description", default=None)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings)
    factory = get_session_factory()
    db = factory()
    try:
        document_id, count = ingest_file(
            path=args.path,
            filename=args.path.name,
            description=args.description,
            db=db,
            settings=settings,
        )
        db.commit()
        print(f"Ingested {args.path.name} as {document_id} ({count} chunks)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
