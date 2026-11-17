"""
==========================================
CHROMADB CLIENT MANAGER
==========================================
This file acts as the "Connection Manager" for the vector database. 
Instead of every file opening its own connection to ChromaDB (which would crash 
your computer and the database), this file ensures the entire application shares 
exactly one stable, persistent connection.
"""

import threading

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)


"""
==========================================
EXPLICIT BREAKDOWN: SINGLETON PATTERN & LOCKS
==========================================
FastAPI can handle multiple user requests at the exact same time using multiple threads.
If two users ask a question at the exact same millisecond, both requests might 
try to open a brand new database connection simultaneously.

To prevent this, we use the "Singleton Pattern" with a Thread Lock:
1. `_chroma_client_instance`: A global variable holding the single database connection.
2. `_chroma_lock`: A strict traffic light. It forces threads to line up one-by-one 
   when trying to access or create the database connection.
"""
_chroma_client_instance = None
_chroma_lock = threading.Lock()


# ==========================================
# 1. GET (OR CREATE) THE GLOBAL CLIENT
# ==========================================
def get_chroma_client() -> chromadb.ClientAPI:
    """Return the process-wide ChromaDB client (persistent)."""
    
    # Tell Python we want to modify the global variable, not create a local one.
    global _chroma_client_instance

    # Acquire the lock. If another request is currently inside this block, 
    # the current request pauses and waits its turn.
    with _chroma_lock:
        
        # If the connection was already created by a previous request, 
        # just return it instantly.
        if _chroma_client_instance is not None:
            return _chroma_client_instance

        # If we are the very first request (app just started), load the settings.
        cfg = get_settings()
        
        # Configure ChromaDB to save data to a physical folder on your hard drive 
        # (persist_directory) so data survives when the server restarts.
        chroma_settings = ChromaSettings(
            persist_directory=str(cfg.chroma_path),
            anonymized_telemetry=False, # Turns off sending usage data to ChromaDB creators
        )
        
        # Actually boot up the ChromaDB engine and save it to the global variable.
        _chroma_client_instance = chromadb.Client(chroma_settings)
        logger.info("using_chroma", extra={"path": str(cfg.chroma_path)})
        
        return _chroma_client_instance


# ==========================================
# 2. COLLECTION BOOTSTRAP (TABLE CREATION)
# ==========================================
def ensure_collection(client: chromadb.ClientAPI | None = None, settings: Settings | None = None) -> None:
    """
    In SQL, you have "Tables". In Vector Databases, you have "Collections".
    This function checks if your specific collection exists. If it doesn't, it creates it.
    It runs right before every search or upload to guarantee the database is ready.
    """
    cfg = settings or get_settings()
    chroma = client or get_chroma_client()
    
    try:
        # Get a list of all existing collections in the database
        collections = chroma.list_collections()
        names = [c.name for c in collections]
        
        # If our collection is already in the list, stop here. We are good to go.
        if cfg.chroma_collection in names:
            return

        # If it DOES NOT exist, we need to create it.
        # First, we need to know how big the math vectors are going to be.
        from app.embeddings.provider import embedding_dimension, get_embeddings
        size = embedding_dimension(get_embeddings())
        
        """
        ==========================================
        EXPLICIT BREAKDOWN: HNSW & COSINE SIMILARITY
        ==========================================
        `chroma.create_collection(...)` creates the empty bucket for your documents.
        
        `"hnsw:space": "cosine"` tells the database HOW to measure distance between vectors.
        HNSW (Hierarchical Navigable Small World) is the extremely fast search algorithm.
        "Cosine" means it will measure the angle between two vectors to see how similar 
        their meanings are, rather than the straight-line physical distance.
        """
        chroma.create_collection(
            name=cfg.chroma_collection,
            metadata={"hnsw:space": "cosine"}, 
        )
        logger.info("created_chroma_collection", extra={"collection": cfg.chroma_collection, "size": size})
        
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("chroma_collection_failed", extra={"error": str(exc)})
        raise VectorStoreError(
            f"Vector store is unavailable. Check CHROMA_PATH: {cfg.chroma_path}"
        ) from exc


# ==========================================
# 3. HEALTH CHECK
# ==========================================
def check_chroma(settings: Settings | None = None) -> str:
    """
    Remember the `/health` endpoint in main.py? 
    That endpoint calls this function. It simply tries to ask the database 
    for its list of collections. If it replies, the database is healthy ("ok"). 
    If it crashes, the database is down ("error").
    """
    _ = settings or get_settings()
    try:
        client = get_chroma_client()
        client.list_collections()
        return "ok"
    except Exception:
        return "error"