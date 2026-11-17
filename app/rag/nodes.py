"""Simplified node base with common boilerplate."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def _to_dict(state) -> dict[str, Any]:
    """Convert GraphState or dict to dict."""
    if hasattr(state, 'to_dict'):
        return state.to_dict()
    return state


class Node(ABC):
    """Base node with timing, logging, and error handling."""

    name: str = "node"

    def __call__(self, state) -> dict[str, Any]:
        state_dict = _to_dict(state)
        started = time.perf_counter()
        try:
            result = self.run(state_dict)
            return self._merge_state(state_dict, result)
        except Exception as exc:
            logger.error(f"{self.name}_failed", extra={"error": str(exc)})
            return self._handle_error(state_dict, exc)
        finally:
            elapsed = int((time.perf_counter() - started) * 1000)
            self._log_timing(state_dict, elapsed)

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the node logic. Return partial state updates."""
        pass

    def _merge_state(self, state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Merge updates into state, accumulating latency."""
        merged = {**state}
        for key, value in updates.items():
            if key in ("llm_latency_ms", "retrieval_latency_ms"):
                merged[key] = int(merged.get(key) or 0) + value
            elif key == "metadata":
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
        return merged

    def _handle_error(self, state: dict[str, Any], exc: Exception) -> dict[str, Any]:
        """Default error handling - can be overridden."""
        return {**state, "error": str(exc)}

    def _log_timing(self, state: dict[str, Any], elapsed_ms: int) -> None:
        """Log timing info - can be overridden."""
        pass


def make_node(func: Callable[[dict[str, Any]], dict[str, Any]], name: str) -> Node:
    """Create a simple node from a function."""
    class FuncNode(Node):
        def __init__(self):
            self.name = name

        def run(self, state: dict[str, Any]) -> dict[str, Any]:
            return func(state)

    return FuncNode()
