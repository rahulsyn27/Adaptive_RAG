"""HTTP client for the AdaptiveRAG FastAPI backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()

    def chat(self, query: str, session_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/v1/chat",
                json={"query": query, "session_id": session_id},
            )
            response.raise_for_status()
            return response.json()

    def upload(self, path: Path, description: str = "") -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            with path.open("rb") as handle:
                files = {"file": (path.name, handle)}
                data = {"description": description} if description else {}
                response = client.post(
                    f"{self.base_url}/api/v1/documents/upload",
                    files=files,
                    data=data,
                )
            response.raise_for_status()
            return response.json()

    def list_documents(self) -> dict[str, Any]:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(f"{self.base_url}/api/v1/documents")
            response.raise_for_status()
            return response.json()
