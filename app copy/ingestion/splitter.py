"""Text splitting configuration."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import Settings, get_settings


def split_documents(documents: list[Document], settings: Settings | None = None) -> list[Document]:
    cfg = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
