"""
==========================================
THE GENERATION NODES (Answer Builders)
==========================================
This file contains the logic that actually writes the final answer back to the user.
You pasted the Generation node instead of the Grader node! That's perfectly fine—we will 
break this down now, and you can share the Grader next.

This file handles all three paths:
1. GENERAL: Small talk, no database or web search needed.
2. INDEX: Answering based on your uploaded documents.
3. SEARCH: Answering based on internet search results.
"""

import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.constants import Route, SourceType
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.groq import get_llm
from app.rag.nodes.grading import _format_chunks
from app.rag.nodes.router import _format_history
from app.rag.policies import normalize_route
from app.rag.state import GraphState
from app.schemas.response import Source, WebResult

logger = get_logger(__name__)


# ==========================================
# HELPER 1: TEXT EXTRACTOR
# ==========================================
def _text(message: BaseMessage | str) -> str:
    """
    LangChain sometimes returns a simple string, and sometimes it returns a complex 
    Message object depending on the LLM used. This helper safely extracts just the 
    raw text string no matter what LangChain gives us, and strips extra whitespace.
    """
    if isinstance(message, str):
        return message.strip()
    content = message.content
    return content.strip() if isinstance(content, str) else str(content).strip()


# ==========================================
# HELPER 2: WEB RESULT FORMATTER
# ==========================================
def _format_web(results: list[WebResult] | None) -> str:
    """
    Takes the raw search results from the internet and formats them into a clean 
    string like: "[1] Title \n URL \n Content", so the LLM can easily read it in the prompt.
    """
    if not results:
        return "(none)"
    lines = []
    for index, item in enumerate(results, start=1):
        lines.append(f"[{index}] {item.title}\nURL: {item.url}\n{item.content}")
    return "\n\n".join(lines)


# ==========================================
# HELPER 3: SOURCE EXTRACTOR (EXPLICIT BREAKDOWN)
# ==========================================
def sources_from_state(state: GraphState) -> list[Source]:
    """
    EXPLICIT BREAKDOWN: 
    When the app replies, it needs to tell the frontend WHERE it got the information 
    (so the UI can display "Source: financial_report.pdf, Page 3").
    This function looks at the route taken and extracts the correct sources.
    """
    route = normalize_route(state.get("route"))
    
    if route is Route.INDEX:
        sources: list[Source] = []
        # We use a 'set' called 'seen' to prevent duplicates.
        # If the database returns 5 chunks all from Page 3 of the same file, 
        # we only want to tell the user "Page 3" once, not 5 times.
        seen: set[tuple[str | None, int | None]] = set()
        
        for chunk in state.get("documents") or []:
            key = (chunk.filename, chunk.page)
            if key in seen:
                continue # Skip if we already recorded this page
                
            seen.add(key)
            # Create a clean Source object to send to the frontend
            sources.append(
                Source(
                    title=chunk.filename,
                    source_type=SourceType.DOCUMENT,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    relevance_score=chunk.score,
                    chunk_id=chunk.chunk_id,
                )
            )
            
        # If the Grader (which runs before this) decided the documents were NOT relevant,
        # we still return the top 3 documents just to show the user what we looked at.
        if not state.get("relevant"):
            return sources[:3]
        return sources
        
    if route is Route.SEARCH:
        # If it was a web search, just convert the URLs into Source objects.
        return [
            Source(
                title=item.title,
                source_type=SourceType.WEB,
                url=item.url,
                relevance_score=item.score,
            )
            for item in state.get("web_results") or []
            if item.url
        ]
        
    # If it was the GENERAL route, there are no sources.
    return []


# ==========================================
# NODE: GENERAL ANSWER
# ==========================================
def make_general_answer_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    """
    This is triggered ONLY if the router picked Route.GENERAL (e.g., User says "Hi").
    It does not look at the database or the web.
    """
    model = llm or get_llm()
    # It gets a specific "general" prompt (e.g., "You are a helpful assistant. Chat with the user.")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("general", "system")),
            ("human", get_prompt("general", "human")),
        ]
    )
    chain = prompt | model

    def general_answer(state: GraphState) -> GraphState:
        started = time.perf_counter()
        try:
            # Just pass the user's query and chat history to the LLM.
            raw = chain.invoke(
                {"query": state["query"], "history": _format_history(state.get("conversation_history"))}
            )
            answer = _text(raw)
        except Exception as exc:
            raise LLMError("The language model failed while answering a general question.") from exc
            
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        
        return {
            "answer": answer,
            "sources": [], # No sources for small talk
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
        }

    return general_answer


# ==========================================
# NODE: MAIN GENERATOR (INDEX & SEARCH)
# ==========================================
def make_generation_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    """
    This is the heavy lifter. It takes the retrieved documents (or web results) 
    and uses them to answer the user's question.
    """
    model = llm or get_llm()
    # Uses a specific prompt (e.g., "Answer the question using ONLY the context provided below.")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("generation", "system")),
            ("human", get_prompt("generation", "human")),
        ]
    )
    chain = prompt | model

    def generate(state: GraphState) -> GraphState:
        # SHORTCIRCUIT: If the 'general_answer' node already ran and generated an answer,
        # we don't want to overwrite it. We just pass through.
        if state.get("answer") and normalize_route(state.get("route")) is Route.GENERAL:
            return {"sources": []}
            
        started = time.perf_counter()
        
        # --- EXPLICIT BREAKDOWN: PREPARING THE CONTEXT ---
        documents = state.get("documents") or []
        relevant = bool(state.get("relevant")) # Did the Grader approve these docs?
        
        # If the docs are relevant, format all of them into a string.
        # If they are NOT relevant (but we ended up here anyway as a fallback), 
        # only send the top 2 chunks so we don't confuse the LLM with too much bad data.
        document_context = _format_chunks(documents if relevant else documents[:2])
        
        # Edge case: The vector database was completely empty or returned 0 results.
        if normalize_route(state.get("route")) is Route.INDEX and not documents:
            document_context = (
                "(no documents retrieved; tell the user that indexed context is unavailable)"
            )
            
        try:
            # 4. RUN THE LLM
            # We shove EVERYTHING into the prompt: the question, history, documents, and web results.
            # The prompt instructions will tell the LLM which context to prioritize based on the route.
            raw = chain.invoke(
                {
                    "query": state["query"],
                    "history": _format_history(state.get("conversation_history")),
                    "document_context": document_context,
                    "web_context": _format_web(state.get("web_results")),
                }
            )
            answer = _text(raw)
        except Exception as exc:
            raise LLMError("The language model failed while generating an answer.") from exc
            
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        
        # Grab the sources to show the user
        sources = sources_from_state(state)
        
        logger.info(
            "generation_complete",
            extra={"source_count": len(sources), "llm_latency_ms": elapsed_ms, "route": state.get("route")},
        )
        
        # Return the final answer! The next node (save_response) will take this and save it to the DB.
        return {
            "answer": answer,
            "sources": sources,
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms,
        }

    return generate