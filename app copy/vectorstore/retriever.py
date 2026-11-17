"""
==========================================
DOCUMENT RETRIEVER (ChromaDB)
==========================================
This file manages all interactions with the Vector Database (ChromaDB). 
When you upload a document, this code saves it. 
When the LLM routes a query to "INDEX", this code searches it.
"""

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.embeddings.provider import get_embeddings
from app.schemas.response import RetrievedChunk
from app.vectorstore.client import ensure_collection, get_chroma_client

logger = get_logger(__name__)


class DocumentRetriever:
    """
    A helper class that bundles all database operations together.
    The graph nodes (like the retrieval node we saw earlier) use this class 
    so they don't have to write raw database queries themselves.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # Load configuration (like database name, default number of search results)
        self.settings = settings or get_settings()
        
        # Get the connection to the ChromaDB server
        self._client = get_chroma_client()
        
        # Get the Embedding Model (the AI that turns text into lists of numbers)
        self._embeddings = get_embeddings()
        
        # A placeholder for the specific "collection" (like a table in SQL) we will use
        self._collection = None

    def _get_collection(self):
        """
        Lazy-loading helper. It only connects to the specific collection 
        when it is actually needed, saving memory and startup time.
        """
        if self._collection is None:
            # Make sure the collection actually exists in the database
            ensure_collection(self._client, self.settings)
            # Fetch the collection using the name from settings
            self._collection = self._client.get_collection(self.settings.chroma_collection)
        return self._collection


    """
    ==========================================
    EXPLICIT BREAKDOWN: UPSERT DOCUMENTS
    ==========================================
    When you upload a document, it gets chopped into smaller paragraphs ("chunks").
    This function takes those chunks and saves them into the vector database.
    "Upsert" means "Update if it exists, Insert if it is new."
    """
    def upsert_documents(self, documents: list[Document]) -> int:
        
        # If the list is empty, do nothing and return 0
        if not documents:
            return 0
            
        ensure_collection(self._client, self.settings)
        collection = self._get_collection()
        
        try:
            # 1. EMBEDDING THE TEXT
            # This is the heavy lifting. It sends the text of every paragraph to an 
            # Embedding Model, which returns a "Vector" (a long array of floating-point numbers) 
            # representing the mathematical meaning of that text.
            vectors = self._embeddings.embed_documents([doc.page_content for doc in documents])

            # Prepare empty lists to organize the data for ChromaDB
            ids = []
            metadatas = []
            contents = []

            # 2. ORGANIZING THE DATA
            # Loop through the documents to separate their parts
            for doc in documents:
                # Extract a unique ID for this specific paragraph chunk
                chunk_id = str(doc.metadata.get("chunk_id"))
                ids.append(chunk_id)
                
                # Clean up metadata (like page number, filename) by removing any empty (None) values
                metadatas.append({k: v for k, v in doc.metadata.items() if v is not None})
                
                # Keep the actual text content
                contents.append(doc.page_content)

            # 3. SAVING TO THE DATABASE
            # Shove all four synchronized lists (IDs, Math Vectors, Metadata, Text) into ChromaDB.
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=contents,
            )
            
            # Log how many chunks we successfully saved
            logger.info(
                "upserted_chunks",
                extra={"count": len(ids), "collection": self.settings.chroma_collection},
            )
            return len(ids)
            
        except VectorStoreError:
            # If a custom vector store error happened, just pass it up the chain
            raise
        except Exception as exc:
            # If something completely unexpected crashed, wrap it in our custom error so the app doesn't die
            raise VectorStoreError("Failed to upsert document chunks into ChromaDB.") from exc


    """
    ==========================================
    EXPLICIT BREAKDOWN: SIMILARITY SEARCH
    ==========================================
    This is what runs when the Router chooses "INDEX". 
    It finds paragraphs in the database that answer the user's question.
    """
    def similarity_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[Document]:
        
        ensure_collection(self._client, self.settings)
        collection = self._get_collection()
        
        # Decide how many results to bring back (e.g., top 3 closest paragraphs)
        k = top_k or self.settings.top_k

        # If we only want to search inside one specific uploaded file, we set a filter ("where")
        where = {"document_id": document_id} if document_id else None

        try:
            # 1. EMBED THE QUESTION
            # Turn the user's text question into a math vector.
            vector = self._embeddings.embed_query(query)
            
            # 2. SEARCH THE DATABASE
            # Ask ChromaDB: "Compare my question vector to all the paragraph vectors you have. 
            # Give me the 'k' closest ones, filtering by 'where'."
            results = collection.query(
                query_embeddings=[vector],
                n_results=k,
                where=where,
                # Tell ChromaDB to return the text, the metadata, and the mathematical distance
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("ChromaDB similarity search failed.") from exc

        # 3. FORMAT THE RESULTS
        documents: list[Document] = []
        if results["documents"]:
            # ChromaDB returns lists inside of lists. `zip` lets us loop through the text, 
            # metadata, and distance at the same time.
            for doc_content, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                metadata = dict(metadata)
                
                # Convert mathematical "Distance" (lower is closer) 
                # into a "Score" (higher is better, out of 1.0)
                score = 1.0 - distance
                metadata["score"] = score
                
                # Package it back into a standard LangChain Document object
                documents.append(Document(page_content=doc_content, metadata=metadata))
                
        logger.info(
            "retrieved_documents",
            extra={"count": len(documents), "top_k": k},
        )
        return documents


    # ==========================================
    # DATA CONVERSION HELPER
    # ==========================================
    def to_chunks(self, documents: list[Document]) -> list[RetrievedChunk]:
        """
        The rest of our app doesn't use raw LangChain 'Document' objects.
        It uses a custom Pydantic schema called `RetrievedChunk`.
        This function just loops through the LangChain docs and translates them 
        into our app's specific format so they are easy to send to the frontend.
        """
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


# ==========================================
# FACTORY FUNCTION
# ==========================================
def get_retriever(settings: Settings | None = None) -> DocumentRetriever:
    """
    A simple helper function to create a new instance of the Retriever class.
    Used by the dependency injection system in FastAPI and LangGraph.
    """
    return DocumentRetriever(settings)