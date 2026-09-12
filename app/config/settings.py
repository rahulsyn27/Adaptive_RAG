"""Environment-driven application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")

    chroma_path: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="adaptive_rag", alias="CHROMA_COLLECTION")

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    database_url: str = Field(default="sqlite:///./adaptive_rag.db", alias="DATABASE_URL")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    top_k: int = Field(default=5, alias="TOP_K", ge=1, le=20)
    max_retrieval_attempts: int = Field(default=2, alias="MAX_RETRIEVAL_ATTEMPTS", ge=1, le=5)

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP", ge=0)

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads", alias="UPLOAD_DIR")
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    groq_temperature: float = Field(default=0.1, alias="GROQ_TEMPERATURE", ge=0.0, le=2.0)
    groq_timeout_seconds: int = Field(default=60, alias="GROQ_TIMEOUT_SECONDS", ge=5)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide settings singleton."""
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return settings
