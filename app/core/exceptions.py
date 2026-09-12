"""Application-level exceptions mapped to HTTP responses."""

from fastapi import HTTPException, status


class AdaptiveRAGError(Exception):
    """Base error for AdaptiveRAG."""

    def __init__(self, message: str, *, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ConfigurationError(AdaptiveRAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class InvalidDocumentTypeError(AdaptiveRAGError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"Unsupported document type for '{filename}'. Allowed: PDF, TXT, Markdown, DOCX.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class EmptyDocumentError(AdaptiveRAGError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"Document '{filename}' contains no extractable text.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class VectorStoreError(AdaptiveRAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class LLMError(AdaptiveRAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY)


class SearchToolError(AdaptiveRAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY)


class DatabaseError(AdaptiveRAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GraphExecutionError(AdaptiveRAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def to_http_exception(exc: AdaptiveRAGError) -> HTTPException:
    """Convert an application error into an HTTPException without a stack trace."""
    return HTTPException(status_code=exc.status_code, detail=exc.message)
