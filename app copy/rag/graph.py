"""
==========================================
WHAT IS LANGGRAPH? (A BEGINNER'S GUIDE)
==========================================
EXPLICIT BREAKDOWN:

LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs). 
Instead of a simple linear chain (Prompt -> LLM -> Output), LangGraph lets you build complex, 
looping workflows that act like a flowchart or a state machine.

Core Concepts:
1. THE STATE (`GraphState`): 
   Think of this as a shared clipboard or a dictionary that gets passed from step to step. 
   Every step reads from this state, does some work, and updates the state with new information.

2. NODES (`add_node`): 
   These are the actual "doers" or workers in your flowchart (e.g., a python function that searches the web, 
   a function that talks to an LLM, a function that queries a database). A node takes the current State, 
   performs an action, and returns an updated State.

3. EDGES (`add_edge`): 
   These are the hardcoded arrows connecting the nodes. It tells the graph: 
   "When Node A is done, ALWAYS go to Node B."

4. CONDITIONAL EDGES (`add_conditional_edges`): 
   These are the decision diamonds in your flowchart. Instead of a hardcoded path, 
   an LLM or a Python function looks at the current State and decides where to go next. 
   (e.g., "Did we find good documents? If YES -> Generate Answer. If NO -> Rewrite Query and Search Again.")

By using LangGraph, your AI doesn't just guess an answer; it follows a strict, repeatable business process.
"""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session, sessionmaker

# Import constants, errors, and logging
from app.core.constants import MessageRole
from app.core.exceptions import GraphExecutionError
from app.core.logging import get_logger
from app.database.database import get_default_session_factory
from app.database.repositories import add_message, get_recent_messages

# Import the actual functions (nodes) that do the work for each step in the graph
from app.rag.nodes.generation import make_general_answer_node, make_generation_node
from app.rag.nodes.grading import make_grading_node
from app.rag.nodes.retrieval import make_retrieval_node
from app.rag.nodes.rewriting import make_rewriting_node
from app.rag.nodes.router import make_router_node
from app.rag.nodes.web_search import make_web_search_node

# Import the decision-making functions (conditional edges)
from app.rag.policies import after_grade, route_after_analysis
from app.rag.state import GraphState, HistoryTurn
from app.tools.search import TavilySearchTool
from app.vectorstore.retriever import DocumentRetriever

logger = get_logger(__name__)


"""
==========================================
EXPLICIT BREAKDOWN: FACTORY FUNCTIONS FOR NODES
==========================================
Notice functions like `make_load_history_node`. Why the `make_` prefix?
LangGraph requires every node to be a function that takes exactly ONE argument: `state`.
But our node needs access to the database (`session_factory`)! 

To solve this, we use a "closure" (a factory pattern). 
The outer function (`make_load_history_node`) takes the database factory.
It then creates and returns the inner function (`load_history`), which only takes `state`.
The inner function "remembers" the database factory from the outer function.
"""

# ==========================================
# NODE: LOAD HISTORY
# ==========================================
def make_load_history_node(
    session_factory: sessionmaker[Session] | None = None,
) -> Callable[[GraphState], GraphState]:
    # Setup the DB factory once
    factory = session_factory or get_default_session_factory()

    # This is the actual function LangGraph will execute
    def load_history(state: GraphState) -> GraphState:
        session_id = state["session_id"]
        db = factory() # Open a database connection
        try:
            # Fetch previous chat messages to give the LLM context of the conversation
            rows = get_recent_messages(db, session_id)
            # Format them into a list of dictionaries
            history: list[HistoryTurn] = [
                {"role": row.role, "content": row.content} for row in rows
            ]
            # Return the updated state. LangGraph will merge this into the main state dictionary.
            return {"conversation_history": history}
        finally:
            db.close() # Always close the connection

    return load_history


# ==========================================
# NODE: SAVE RESPONSE
# ==========================================
def make_save_response_node(
    session_factory: sessionmaker[Session] | None = None,
) -> Callable[[GraphState], GraphState]:
    factory = session_factory or get_default_session_factory()

    def save_response(state: GraphState) -> GraphState:
        logger.debug(
            "save_response_start",
            extra={"session_id": state.get("session_id"), "has_answer": bool(state.get("answer"))},
        )
        db = factory()
        try:
            # Grab the final answer that the LLM generated and save it to the DB as the "ASSISTANT" role.
            add_message(db, state["session_id"], MessageRole.ASSISTANT.value, state.get("answer") or "")
            db.commit() # Save changes
            logger.debug("save_response_success", extra={"session_id": state.get("session_id")})
        except Exception as exc:
            logger.error("save_response_failed", extra={"session_id": state.get("session_id"), "error": str(exc)})
            db.rollback() # Undo if it crashed
            raise GraphExecutionError("Failed to persist the assistant response.") from exc
        finally:
            db.close()
            
        # We don't need to add anything new to the state here, so we return an empty dictionary.
        return {}

    return save_response


# ==========================================
# THE GRAPH BUILDER (THE FLOWCHART BLUEPRINT)
# ==========================================
def build_graph(
    *,
    llm: BaseChatModel | None = None,
    retriever: DocumentRetriever | None = None,
    search_tool: TavilySearchTool | None = None,
    session_factory: sessionmaker[Session] | None = None,
):
    """
    EXPLICIT BREAKDOWN: BUILDING THE STATE MACHINE
    This function wires all the nodes together into a functioning graph.
    """
    
    # 1. Initialize the graph, telling it what schema to use for the shared state.
    graph = StateGraph(GraphState)
    
    # 2. Add all the "Workers" (Nodes) to the graph.
    # We name the node (e.g., "load_history") and assign the Python function that executes it.
    graph.add_node("load_history", make_load_history_node(session_factory))
    graph.add_node("analyze_query", make_router_node(llm))
    graph.add_node("retrieve", make_retrieval_node(retriever))
    graph.add_node("grade", make_grading_node(llm))
    graph.add_node("rewrite", make_rewriting_node(llm))
    graph.add_node("web_search", make_web_search_node(search_tool))
    graph.add_node("general_answer", make_general_answer_node(llm))
    graph.add_node("generate", make_generation_node(llm))
    graph.add_node("save_response", make_save_response_node(session_factory))

    # 3. Define the flow (Edges).
    
    # START is a special built-in node. The graph ALWAYS starts by loading history.
    graph.add_edge(START, "load_history")
    
    # After loading history, ALWAYS go to analyze the query.
    graph.add_edge("load_history", "analyze_query")
    
    # --- CONDITIONAL EDGE (Decision Point 1) ---
    # The graph pauses here and runs the `route_after_analysis` function.
    # That function looks at what `analyze_query` figured out and returns a string.
    # If it returns "INDEX", the graph moves to the "retrieve" node.
    # If "GENERAL", it goes to "general_answer". If "SEARCH", it goes to "web_search".
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "INDEX": "retrieve",
            "GENERAL": "general_answer",
            "SEARCH": "web_search",
        },
    )
    
    # If it went to retrieve, ALWAYS grade the retrieved documents next.
    graph.add_edge("retrieve", "grade")
    
    # --- CONDITIONAL EDGE (Decision Point 2) ---
    # After grading, did we find good documents?
    # `after_grade` function looks at the state. 
    # If the docs were good -> go to "generate" to write the final answer.
    # If the docs were bad -> go to "rewrite" to change the search query and try again.
    graph.add_conditional_edges(
        "grade",
        after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )
    
    # If we rewrote the query, ALWAYS go back to retrieve new documents (this forms a loop!).
    graph.add_edge("rewrite", "retrieve")
    
    # If we did a web search, ALWAYS generate an answer based on it.
    graph.add_edge("web_search", "generate")
    
    # If it was just a general greeting ("Hi"), general_answer handles it, then ALWAYS goes to generate.
    graph.add_edge("general_answer", "generate")
    
    # After generating the final answer, ALWAYS save it to the DB.
    graph.add_edge("generate", "save_response")
    
    # END is a special built-in node. After saving, the graph execution is completely finished.
    graph.add_edge("save_response", END)
    
    # .compile() locks the graph blueprint and returns an executable object that the chat endpoint can `ainvoke`.
    return graph.compile()


"""
==========================================
WHAT DOES `Callable[[GraphState], GraphState]` MEAN?
==========================================
EXPLICIT BREAKDOWN:

This is a Python "Type Hint". It doesn't change how the code runs, but it acts as a strict blueprint. 
It tells developers (and code editors like VSCode) exactly what kind of data is going in and coming out.

Let's break it down piece by piece:

1. `Callable`:
   This simply means "a function" (or anything in Python that can be executed using parentheses `()`).
   When you see a factory function returning a `Callable`, it means it is generating and returning 
   ANOTHER function, not a simple variable like a string or a number.

2. `[[GraphState], ...]` (The Input Argument):
   The first part inside the square brackets defines what arguments this returned function must accept.
   Here, it means the function accepts exactly one argument, and that argument must be of type `GraphState`.

3. `..., GraphState]` (The Return Value):
   The second part (after the comma) defines what this function will give back when it finishes running.
   Here, it promises to return a new or updated `GraphState`.

==========================================
WHY IS IT USED IN THIS PROJECT?
==========================================
LangGraph is very strict. It requires EVERY node in its flowchart to follow one golden rule:
"You must be a function that takes the current State as your only input, and you must return a State update."

Because we are using Factory Functions (like `make_router_node`) to dynamically build our nodes, 
the Type Hint `-> Callable[[GraphState], GraphState]:` acts as a guarantee. 

It tells LangGraph: 
"I am a factory. When you call me, I will construct and hand you back a custom function that perfectly 
matches your strict 'State IN -> State OUT' rule."
"""