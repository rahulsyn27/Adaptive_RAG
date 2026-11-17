"""
HTTP Dependencies File

This file contains "dependencies" for FastAPI endpoints. 
In FastAPI, a dependency is a reusable function that gets executed BEFORE your actual endpoint logic runs. 
It is used to safely "inject" required tools (like configuration settings or database connections) 
into your routes, ensuring you don't have to write setup/teardown code in every single route file.
"""

from collections.abc import Generator
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.database import get_session_factory


# ==========================================
# 1. SETTINGS DEPENDENCY
# ==========================================
def settings_dep() -> Settings:
    """
    A simple dependency function. 
    When an endpoint route needs access to the app's configuration (like environment variables), 
    it will declare this dependency. FastAPI calls this function and hands the route the settings.
    """
    return get_settings()


# ==========================================
# 2. DATABASE SESSION DEPENDENCY (EXPLICIT BREAKDOWN)
# ==========================================
def db_session_dep() -> Generator[Session, None, None]:
    """
    EXPLICIT EXPLANATION:
    This function manages the lifecycle of a database transaction safely. 
    It uses a Python Generator (indicated by the 'yield' keyword).

    Instead of just returning a connection, it acts as a wrapper around your route's execution:
    1. It opens the database connection.
    2. It PAUSES and gives the connection to your route.
    3. When your route is done, it RESUMES to save the data or undo errors.
    """
    
    # Step 1: Retrieve the factory object that knows how to create new database connections.
    factory = get_session_factory()
    
    # Step 2: Open a fresh, empty database session specifically for the current HTTP request.
    session = factory()
    
    try:
        # Step 3: YIELD (The Pause Point)
        # The 'yield' keyword temporarily pauses this function. 
        # FastAPI takes this 'session' variable and hands it to your route (e.g., the document upload route).
        # Your route uses it to run queries (like session.add(new_document)).
        yield session
        
        # Step 4: COMMIT (The Success Path)
        # If your route finishes successfully without any errors, FastAPI comes back here.
        # session.commit() takes all the pending changes your route made and permanently saves them to the DB.
        session.commit()
        
    except Exception:
        # Step 5: ROLLBACK (The Failure Path)
        # If your route crashes (e.g., bad data format, logic error), execution immediately jumps here.
        # session.rollback() acts as an "Undo" button. It wipes out any partial changes made during the request
        # so you don't end up with corrupted or half-saved document records in your database.
        session.rollback()
        
        # Re-raise the exact error so it can be caught by the global error handler (from the main.py file).
        raise
        
    finally:
        # Step 6: CLOSE (The Cleanup Path)
        # The 'finally' block is guaranteed to run, regardless of whether the request succeeded or crashed.
        # It closes the database connection, freeing up resources. 
        # Without this, your application would hold connections hostage and eventually crash from connection limits.
        session.close()