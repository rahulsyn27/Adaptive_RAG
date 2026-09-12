"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, documents, health
from app.config.settings import get_settings
from app.core.constants import API_V1_PREFIX, APP_NAME, APP_VERSION
from app.core.exceptions import AdaptiveRAGError
from app.core.logging import configure_logging, get_logger, new_request_id
from app.database.database import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings)
    logger.info("application_started", extra={"version": APP_VERSION})
    yield


def create_app() -> FastAPI:
    application = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = new_request_id()
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(AdaptiveRAGError)
    async def handle_app_error(_request: Request, exc: AdaptiveRAGError) -> JSONResponse:
        logger.error("unhandled_adaptive_error", extra={"error": exc.message, "status_code": exc.status_code})
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    application.include_router(health.router, prefix=API_V1_PREFIX)
    application.include_router(chat.router, prefix=API_V1_PREFIX)
    application.include_router(documents.router, prefix=API_V1_PREFIX)
    return application


app = create_app()
