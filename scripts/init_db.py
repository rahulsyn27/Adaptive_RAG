"""Initialize SQLite tables."""

from app.config.settings import get_settings
from app.core.logging import configure_logging
from app.database.database import init_db


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings)
    print(f"Initialized database at {settings.database_url}")


if __name__ == "__main__":
    main()
