"""Structured logging helpers. Secrets are never attached to log records."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from app.core.constants import APP_NAME

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")

_REDACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "groq_api_key",
        "qdrant_api_key",
        "tavily_api_key",
        "password",
        "secret",
        "token",
    }
)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "app": APP_NAME,
            "request_id": getattr(record, "request_id", request_id_ctx.get()),
            "session_id": getattr(record, "session_id", session_id_ctx.get()),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in logging.LogRecord("", 0, "", 0, None, None, None).__dict__
            and key not in {"request_id", "session_id", "message", "asctime"}
        }
        for key, value in extras.items():
            if key.lower() in _REDACT_KEYS:
                continue
            payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the process."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def new_request_id() -> str:
    """Allocate a request identifier for the current context."""
    request_id = uuid4().hex[:12]
    request_id_ctx.set(request_id)
    return request_id


def bind_session(session_id: str) -> None:
    """Attach a session identifier to subsequent log records."""
    session_id_ctx.set(session_id or "-")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
