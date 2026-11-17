"""Shared constants for AdaptiveRAG."""

from enum import StrEnum

API_V1_PREFIX = "/api/v1"
APP_NAME = "AdaptiveRAG"
APP_VERSION = "0.1.0"

SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown", ".docx"})


class Route(StrEnum):
    """Adaptive execution paths chosen by the query router."""

    INDEX = "INDEX"
    GENERAL = "GENERAL"
    SEARCH = "SEARCH"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceType(StrEnum):
    DOCUMENT = "document"
    WEB = "web"
    MODEL = "model"
