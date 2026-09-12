"""Chunk metadata attached before Qdrant upsert."""

from uuid import uuid4

from langchain_core.documents import Document


def attach_chunk_metadata(
    chunks: list[Document],
    *,
    document_id: str,
    filename: str,
    description: str | None,
) -> list[Document]:
    enriched: list[Document] = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        page = metadata.get("page")
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None
        metadata.update(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_id": str(uuid4()),
                "chunk_index": index,
                "page": page,
                "source": metadata.get("source") or filename,
                "description": description or "",
            }
        )
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched
