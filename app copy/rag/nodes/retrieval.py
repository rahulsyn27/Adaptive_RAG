"""
==========================================
THE RETRIEVAL NODE
==========================================
This file contains the logic for fetching relevant document chunks from the 
Vector Database (Qdrant in this case) when the LLM router decides the 
user's question requires the "INDEX" path.
"""

import time
from collections.abc import Callable

# Custom imports for logging, state management, and vector database connection
from app.core.logging import get_logger
from app.rag.state import GraphState
from app.vectorstore.retriever import DocumentRetriever, get_retriever

# Setup a logger specifically for this file
logger = get_logger(__name__)


# ==========================================
# THE FACTORY FUNCTION
# ==========================================
def make_retrieval_node(
    retriever: DocumentRetriever | None = None,
) -> Callable[[GraphState], GraphState]:
    """
    Builds and returns the 'retrieve' node function for LangGraph.
    Like the router, it uses the factory pattern to inject the database connection.
    """
    
    # Grab the connection to the vector database (Qdrant). 
    # If one isn't passed in (usually for testing), it gets the default one.
    store = retriever or get_retriever()


    # ==========================================
    # THE ACTUAL NODE LOGIC (Runs during the graph execution)
    # ==========================================
    def retrieve(state: GraphState) -> GraphState:
        # Start a high-precision timer to track how long the database search takes.
        started = time.perf_counter()
        
        """
        ==========================================
        EXPLICIT BREAKDOWN: WHICH QUERY TO USE?
        ==========================================
        `query = state.get("rewritten_query") or state["query"]`
        
        Why are we checking for a "rewritten_query"?
        Remember the loop in the LangGraph blueprint (from a few files ago):
        Retrieve -> Grade -> Rewrite -> Retrieve (Loop)
        
        1. On the FIRST pass, "rewritten_query" doesn't exist yet, so it falls 
           back to `state["query"]` (the user's exact original words).
        2. If the documents found were bad, the graph hits the "Rewrite" node.
           The LLM rewrites the question to be better for searching and saves it 
           as "rewritten_query".
        3. On the SECOND pass (the loop), this code grabs the "rewritten_query" 
           instead of the original one to get better search results.
        """
        query = state.get("rewritten_query") or state["query"]
        
        # Keep track of how many times we've tried to search the database during this request.
        # It defaults to 0 on the first try, adding 1 each time it runs.
        attempts = int(state.get("retrieval_attempts") or 0) + 1
        
        # --- THE CORE SEARCH ENGINE ---
        # This asks the vector database to find documents that are mathematically 
        # similar to the meaning of the user's query.
        documents = store.similarity_search(query)
        
        # Convert the raw database response into clean, standardized "Chunks" 
        # (pieces of text) that the rest of our app understands.
        chunks = store.to_chunks(documents)
        
        # Calculate exactly how many milliseconds the search took.
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        
        # Log the results of the search for observability and debugging.
        logger.info(
            "retrieval_complete",
            extra={
                "retrieval_attempts": attempts,          # Was this try #1? Try #2?
                "retrieved_documents": len(chunks),      # How many pieces of text did we find?
                "retrieval_latency_ms": elapsed_ms,      # How fast was the database?
                "session_id": state.get("session_id"),
            },
        )
        
        # ==========================================
        # UPDATE THE STATE
        # ==========================================
        # Return a dictionary containing the new information. 
        # LangGraph will automatically update the shared `GraphState` with these values.
        return {
            "documents": chunks, # The actual text chunks we found (passed to the Grader next)
            "retrieval_attempts": attempts,
            # Add the time taken for this search to the total retrieval time 
            # (important if we looped and searched multiple times).
            "retrieval_latency_ms": int(state.get("retrieval_latency_ms") or 0) + elapsed_ms,
            "metadata": {
                # Copy any existing metadata
                **(state.get("metadata") or {}),
                "retrieved_documents": len(chunks),
                # Record exactly which query (original or rewritten) was used to find these docs
                "active_query": query, 
            },
        }

    # Return the inner function so LangGraph can plug it into the flowchart.
    return retrieve