"""
==========================================
THE ROUTER NODE (Query Analyzer)
==========================================
This file is the exact implementation of the "Router" we just discussed. 
It shows how we force the LLM to choose a path (INDEX, GENERAL, SEARCH) 
and how it handles errors if the LLM gets confused.
"""

import time
from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config.prompts import get_prompt
from app.core.constants import Route
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.groq import get_llm
from app.rag.state import GraphState, HistoryTurn
from app.schemas.chat import RouteDecision

logger = get_logger(__name__)


# ==========================================
# HELPER: FORMAT CONVERSATION HISTORY
# ==========================================
def _format_history(turns: list[HistoryTurn] | None) -> str:
    """
    If the user says "What does it mean?", the LLM needs to know what "it" is.
    This function takes the previous chat messages and mashes them into a single 
    text string so the LLM has context. It only takes the last 8 messages (`turns[-8:]`) 
    so we don't send a massive history that wastes tokens or confuses the LLM.
    """
    if not turns:
        return "(none)"
    return "\n".join(f"{item['role']}: {item['content']}" for item in turns[-8:])


# ==========================================
# THE FACTORY FUNCTION
# ==========================================
def make_router_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], GraphState]:
    """
    Builds and returns the 'analyze_query' node function for LangGraph.
    """
    
    # 1. Initialize the AI Model (e.g., Groq, OpenAI).
    model = llm or get_llm()
    
    # --- EXPLICIT BREAKDOWN: STRUCTURED OUTPUT ---
    # This is the magic trick. LLMs naturally want to talk in full sentences.
    # `with_structured_output` forces the LLM to output a strict JSON object that 
    # perfectly matches your Pydantic model (`RouteDecision`). 
    # Instead of "I think the route is INDEX", it outputs `{"route": "INDEX", "reason": "..."}`.
    structured = model.with_structured_output(RouteDecision)
    
    # 2. Build the Hidden Prompt.
    # This fetches the instructions (like the ones I described in the previous message) 
    # from a configuration file.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("router", "system")),
            ("human", get_prompt("router", "human")),
        ]
    )
    
    # 3. Create the LangChain "Chain".
    # This connects the prompt directly to the structured LLM. Data flows from left to right.
    chain = prompt | structured


    # ==========================================
    # THE ACTUAL NODE LOGIC (Runs during the graph execution)
    # ==========================================
    def analyze_query(state: GraphState) -> GraphState:
        # Start a timer to track how long the LLM takes to think.
        started = time.perf_counter()
        
        # Grab the user's current question from the shared state.
        query = state["query"]
        
        try:
            # 4. RUN THE AI
            # We pass the user's question and the formatted history into the chain.
            # `decision` will be a nice, clean `RouteDecision` python object.
            decision = chain.invoke({"query": query, "history": _format_history(state.get("conversation_history"))})
            
        except Exception as exc:
            # SAFETY NET: What if the LLM hallucinated, output bad JSON, or the API crashed?
            logger.warning("router_structured_output_failed", extra={"error": str(exc)})
            
            # Instead of crashing the whole app, we default to the "GENERAL" route 
            # so the user at least gets a response.
            decision = RouteDecision(
                route=Route.GENERAL,
                reason="Router output could not be parsed; defaulting to general knowledge.",
            )
            
        # 5. Strict Type Validation
        # Double check that Langchain actually gave us the exact object we expect.
        if not isinstance(decision, RouteDecision):
            try:
                decision = RouteDecision.model_validate(decision)
            except Exception as exc:
                raise LLMError("Router returned malformed structured output.") from exc
                
        # Calculate how long this took in milliseconds.
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        
        # Log the decision for observability.
        logger.info(
            "route_selected",
            extra={"route": decision.route.value, "llm_latency_ms": elapsed_ms, "session_id": state.get("session_id")},
        )
        
        # 6. UPDATE THE STATE
        # This returns a dictionary that LangGraph will merge into the main GraphState.
        # This is how the next node (and the conditional edge) knows where to go.
        return {
            "route": decision.route,                          # e.g., "INDEX"
            "route_reason": decision.reason,                  # The LLM's explanation for its choice
            "llm_latency_ms": int(state.get("llm_latency_ms") or 0) + elapsed_ms, # Add to total LLM time
            "metadata": {**(state.get("metadata") or {}), "route_reason": decision.reason},
        }

    # Return the inner function so LangGraph can use it.
    return analyze_query