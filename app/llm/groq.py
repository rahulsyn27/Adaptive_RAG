"""LangChain Groq chat model factory."""

from langchain_groq import ChatGroq

from app.config.settings import Settings, get_settings
from app.core.exceptions import ConfigurationError, LLMError


def get_llm(settings: Settings | None = None) -> ChatGroq:
    """Return a configured ChatGroq instance.

    Callers should depend on LangChain's chat model interface, not Groq's SDK.
    """
    cfg = settings or get_settings()
    if not cfg.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is not configured.")
    try:
        return ChatGroq(
            api_key=cfg.groq_api_key,
            model=cfg.groq_model,
            temperature=cfg.groq_temperature,
            timeout=cfg.groq_timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - constructor is typically pure
        raise LLMError("Failed to initialize the Groq chat model.") from exc
