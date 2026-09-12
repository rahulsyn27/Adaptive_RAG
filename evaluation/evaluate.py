"""CLI evaluation for routing and answer quality."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from evaluation.metrics import EvaluationResult, compute_metrics

DATASETS = Path(__file__).resolve().parent / "datasets"


def load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or (DATASETS / "routing.json")
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["examples"])


def evaluate_routing_offline(examples: list[dict[str, Any]]) -> EvaluationResult:
    """Score recorded predictions shipped with the dataset (baseline / CI)."""
    predictions = []
    for item in examples:
        predicted = item.get("predicted_route") or item["expected_route"]
        predictions.append(
            {
                "expected_route": item["expected_route"],
                "predicted_route": predicted,
                "retrieved": item.get("retrieved", item["expected_route"] == "INDEX"),
                "grounded": item.get("grounded", True),
                "correct": item.get("correct", True),
                "retried": item.get("retried", False),
                "latency_ms": item.get("latency_ms", 0),
            }
        )
    return compute_metrics(predictions)


def evaluate_against_api(examples: list[dict[str, Any]], base_url: str) -> EvaluationResult:
    import httpx

    predictions: list[dict[str, Any]] = []
    for item in examples:
        started = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/api/v1/chat",
                json={"query": item["question"], "session_id": f"eval-{item.get('id', 'x')}"},
            )
            response.raise_for_status()
            body = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)
        route = body.get("route")
        predictions.append(
            {
                "expected_route": item["expected_route"],
                "predicted_route": route,
                "retrieved": (body.get("metadata") or {}).get("retrieved_documents", 0) > 0,
                "grounded": bool(body.get("sources")) if item["expected_route"] == "INDEX" else True,
                "correct": route == item["expected_route"],
                "retried": (body.get("metadata") or {}).get("retrieval_attempts", 0) > 1,
                "latency_ms": latency_ms,
            }
        )
    return compute_metrics(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveRAG routing and quality metrics")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--api-url", default=None, help="If set, call a live API instead of offline labels")
    args = parser.parse_args()
    examples = load_dataset(args.dataset)
    result = evaluate_against_api(examples, args.api_url) if args.api_url else evaluate_routing_offline(examples)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
