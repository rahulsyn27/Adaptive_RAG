# """FastAPI application entrypoint."""

# from collections.abc import AsyncIterator
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# # Custom modules for routing, configuration, and core utilities
# from app.api.routes import chat, documents, health
# from app.config.settings import get_settings
# from app.core.constants import API_V1_PREFIX, APP_NAME, APP_VERSION
# from app.core.exceptions import AdaptiveRAGError
# from app.core.logging import configure_logging, get_logger, new_request_id
# from app.database.database import init_db

# # Initialize a logger specific to this file path
# logger = get_logger(__name__)


# # ==========================================
# # 1. LIFESPAN / STARTUP SEQUENCE
# # ==========================================
# # This context manager runs exactly once before the server starts accepting requests.
# @asynccontextmanager
# async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
#     settings = get_settings()                             # Load environment variables
#     configure_logging(settings.log_level)                 # Set how detailed the logs should be
#     init_db(settings)                                     # Establish database connection
#     logger.info("application_started", extra={"version": APP_VERSION}) # Log successful boot
    
#     yield  # PAUSE HERE: The server goes live and handles traffic. 
#            # (Any code placed after this 'yield' would run during server shutdown).


# # ==========================================
# # 2. APPLICATION FACTORY
# # ==========================================
# # This function constructs and configures the FastAPI server instance.
# def create_app() -> FastAPI:
#     # Instantiate the core app, attaching metadata for docs and the lifespan logic
#     application = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
    
#     # --- 3. CORS SECURITY ---
#     # Disables Cross-Origin security restrictions. 
#     # Allows frontends from any domain ('*') to send any type of request to this backend.
#     application.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )

#     # --- 4. REQUEST TRACKING MIDDLEWARE ---
#     # Acts as a tollbooth: intercepts every incoming HTTP request.
#     @application.middleware("http")
#     async def request_context(request: Request, call_next):
#         request_id = new_request_id()                 # 1. Generate a unique tracking ID
#         request.state.request_id = request_id         # 2. Attach ID to the request's temporary state (temp storage)
        
#         response = await call_next(request)           # 3. Pass request to the actual endpoint and WAIT for it to finish | response = {"status_code": 200}}
        
#         response.headers["X-Request-ID"] = request_id # 4. Attach the same tracking ID to the final response headers
#         return response                               # 5. Send it back to the client

#     # --- 5. GLOBAL ERROR HANDLER ---
#     # Safety net: Catches any 'AdaptiveRAGError' thrown anywhere in the app to prevent server crashes.
#     @application.exception_handler(AdaptiveRAGError)
#     async def handle_app_error(_request: Request, exc: AdaptiveRAGError) -> JSONResponse:
#         # Log the error internally for debugging
#         logger.error("unhandled_adaptive_error", extra={"error": exc.message, "status_code": exc.status_code})
#         # Return a clean, formatted JSON response to the user with the exact error details
#         return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

#     # --- 6. ROUTER PLUG-INS ---
#     # Connects the external files (where the actual logic lives) to this main application.
#     # It prefixes all these routes (e.g., changing `/upload` to `/api/v1/upload`).
#     application.include_router(health.router, prefix=API_V1_PREFIX)
#     application.include_router(chat.router, prefix=API_V1_PREFIX)
#     application.include_router(documents.router, prefix=API_V1_PREFIX) # <-- Document ingestion traffic goes here!
    
#     return application


# # Execute the factory function to create the app instance.
# # An ASGI server (like Uvicorn) will look for this 'app' variable to run the backend.
# app = create_app()