"""ChromaDB similarity search returning LangChain documents."""

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.embeddings.provider import get_embeddings
from app.schemas.response import RetrievedChunk
from app.vectorstore.client import ensure_collection, get_chroma_client

logger = get_logger(__name__)


class DocumentRetriever:
    """Single configured retriever used by graph nodes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = get_chroma_client()
        self._embeddings = get_embeddings()
        self._collection = None

    def _get_collection(self):
        """Get or create the collection."""
        if self._collection is None:
            ensure_collection(self._client, self.settings)
            self._collection = self._client.get_collection(self.settings.chroma_collection)
        return self._collection

    def upsert_documents(self, documents: list[Document]) -> int:
        """Embed and upsert LangChain documents into ChromaDB."""
        if not documents:
            return 0
        ensure_collection(self._client, self.settings)
        collection = self._get_collection()
        try:
            vectors = self._embeddings.embed_documents([doc.page_content for doc in documents])

            ids = []
            metadatas = []
            contents = []

            for doc in documents:
                chunk_id = str(doc.metadata.get("chunk_id"))
                ids.append(chunk_id)
                metadatas.append({k: v for k, v in doc.metadata.items() if v is not None})
                contents.append(doc.page_content)

            collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=contents,
            )
            logger.info(
                "upserted_chunks",
                extra={"count": len(ids), "collection": self.settings.chroma_collection},
            )
            return len(ids)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Failed to upsert document chunks into ChromaDB.") from exc

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[Document]:
        """Return the nearest chunks as LangChain Documents."""
        ensure_collection(self._client, self.settings)
        collection = self._get_collection()
        k = top_k or self.settings.top_k

        where = {"document_id": document_id} if document_id else None

        try:
            vector = self._embeddings.embed_query(query)
            results = collection.query(
                query_embeddings=[vector],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("ChromaDB similarity search failed.") from exc

        documents: list[Document] = []
        if results["documents"]:
            for doc_content, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                metadata = dict(metadata)
                # Convert distance to score (cosine similarity = 1 - distance for normalized vectors)
                score = 1.0 - distance
                metadata["score"] = score
                documents.append(Document(page_content=doc_content, metadata=metadata))
        logger.info(
            "retrieved_documents",
            extra={"count": len(documents), "top_k": k},
        )
        return documents

    def to_chunks(self, documents: list[Document]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for doc in documents:
            meta = doc.metadata
            page = meta.get("page")
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    document_id=meta.get("document_id"),
                    filename=meta.get("filename"),
                    chunk_id=str(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
                    page=int(page) if page is not None else None,
                    source=meta.get("source"),
                    description=meta.get("description"),
                    score=float(meta["score"]) if meta.get("score") is not None else None,
                )
            )
        return chunks


def get_retriever(settings: Settings | None = None) -> DocumentRetriever:
    return DocumentRetriever(settings)
