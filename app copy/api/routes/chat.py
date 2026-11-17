"""
==========================================
CHAT ENDPOINT FILE
==========================================
This file handles incoming user chat messages, saves them to the database, 
and passes them to an AI workflow (LangGraph) to generate a response.
"""

import time
from functools import lru_cache

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

# Importing dependencies, database functions, schemas, and LangGraph components
from app.api.dependencies import db_session_dep
from app.core.constants import MessageRole
from app.core.exceptions import AdaptiveRAGError, GraphExecutionError, to_http_exception
from app.core.logging import bind_session, get_logger
from app.database.repositories import add_message, create_session
from app.rag.graph import build_graph
from app.rag.policies import normalize_route
from app.rag.state import GraphState
from app.schemas.chat import ChatMetadata, ChatRequest, ChatResponse
from app.schemas.response import Source

# Create a specific router for chat operations with the prefix `/chat`
router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


"""
==========================================
EXPLICIT BREAKDOWN: @lru_cache(maxsize=1)
==========================================
This is a standard Python decorator for caching (Least Recently Used cache).
Building a LangGraph workflow (the AI decision tree) is a heavy, slow process. 
If we rebuilt it for every single user request, the app would be very slow.

By using `@lru_cache(maxsize=1)`, we tell Python:
1. Run `build_graph()` the very first time this function is called.
2. Save (cache) the compiled result in memory.
3. For all future calls, do not run `build_graph()` again. Just return the saved copy instantly.
`maxsize=1` means we only need to store exactly one copy of the graph.
"""
@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()


"""
==========================================
EXPLICIT BREAKDOWN: DEPENDENCY INJECTION
==========================================
`db: Session = Depends(db_session_dep)`

This tells FastAPI: "Before you run this `chat` function, run `db_session_dep` first."
As we saw in the previous file, `db_session_dep` yields an open database connection.
FastAPI takes that open connection and hands it to this function as the `db` variable.
"""
@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(payload: ChatRequest, request: Request, db: Session = Depends(db_session_dep)) -> ChatResponse:
    
    # 1. Attach the session ID to the logger so all logs for this request are grouped together.
    bind_session(payload.session_id)
    
    # 2. Start a high-precision timer to measure exactly how long this entire request takes.
    started = time.perf_counter()
    
    # 3. Database Operations:
    # Ensure a chat session exists in the DB for this user, then save the user's prompt (message).
    create_session(db, payload.session_id)
    add_message(db, payload.session_id, MessageRole.USER.value, payload.query)
    
    # Save (commit) these changes to the database immediately.
    db.commit()
    
    # 4. Fetch the pre-compiled AI workflow (instantly retrieved from cache).
    graph = get_compiled_graph()
    
    # 5. Setup the initial state (the starting variables) that will be fed into the LangGraph workflow.
    initial: GraphState = {
        "query": payload.query,
        "session_id": payload.session_id,
        "retrieval_attempts": 0,
        "llm_latency_ms": 0,
        "retrieval_latency_ms": 0,
        # Fetch the unique request_id we generated back in the middleware (main.py)
        "metadata": {"request_id": getattr(request.state, "request_id", "-")},
    }
    
    """
    ==========================================
    EXPLICIT BREAKDOWN: GRAPH EXECUTION
    ==========================================
    `await graph.ainvoke(initial)`
    
    This is the core of the endpoint. It passes the initial data (the user's query) into the 
    LangGraph workflow. The graph will autonomously decide what to do next (e.g., search a vector 
    database, re-write the query, or answer directly) based on its built-in logic.
    
    This runs asynchronously (`await`), meaning the server can handle other users' requests 
    while waiting for the LLM to finish thinking.
    """
    try:
        result = await graph.ainvoke(initial)
    except AdaptiveRAGError as exc:
        # If our custom logic fails, log it and convert it to a standard HTTP error.
        logger.error("chat_failed", extra={"error": exc.message})
        raise to_http_exception(exc) from exc
    except Exception as exc:
        # If a completely unexpected error occurs (like the database crashing), catch it safely.
        logger.error("chat_failed", extra={"error": str(exc)})
        raise to_http_exception(GraphExecutionError("The adaptive workflow failed to complete.")) from exc

    # 6. Calculate total execution time in milliseconds.
    latency_ms = int((time.perf_counter() - started) * 1000)
    
    # 7. Extract data from the graph's final result.
    route = normalize_route(result.get("route"))
    
    # Validate and format the sources (the documents the AI used to find the answer).
    raw_sources = result.get("sources") or []
    sources = [item if isinstance(item, Source) else Source.model_validate(item) for item in raw_sources]
    
    meta = result.get("metadata") or {}
    
    # 8. Log the performance metrics and decision paths taken by the AI workflow.
    logger.info(
        "chat_complete",
        extra={
            "route": route.value,
            "retrieval_attempts": result.get("retrieval_attempts") or 0,
            "retrieved_documents": meta.get("retrieved_documents", 0),
            "relevant_documents": meta.get("relevant_documents", 0),
            "llm_latency_ms": result.get("llm_latency_ms") or 0,
            "retrieval_latency_ms": result.get("retrieval_latency_ms") or 0,
            "latency_ms": latency_ms,
        },
    )
    
    # 9. Format the final output matching the ChatResponse schema and send it back to the user.
    return ChatResponse(
        answer=result.get("answer") or "I was unable to produce an answer.",
        route=route,
        sources=sources,
        session_id=payload.session_id,
        metadata=ChatMetadata(
            retrieval_attempts=int(result.get("retrieval_attempts") or 0),
            retrieved_documents=int(meta.get("retrieved_documents") or 0),
            relevant_documents=int(meta.get("relevant_documents") or 0),
            latency_ms=latency_ms,
            retrieval_latency_ms=int(result.get("retrieval_latency_ms") or 0),
            llm_latency_ms=int(result.get("llm_latency_ms") or 0),
            rewritten_query=result.get("rewritten_query"),
        ),
    )