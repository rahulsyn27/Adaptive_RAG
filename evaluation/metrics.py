"""Evaluation metrics for AdaptiveRAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvaluationResult:
    routing_accuracy: float
    retrieval_success_rate: float
    answer_groundedness: float
    answer_correctness: float
    average_latency_ms: float
    retry_rate: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def compute_metrics(rows: list[dict[str, Any]]) -> EvaluationResult:
    """Compute portfolio metrics from per-example prediction records."""
    n = len(rows)
    routing_hits = sum(1 for row in rows if row.get("predicted_route") == row.get("expected_route"))
    index_rows = [row for row in rows if row.get("expected_route") == "INDEX"]
    retrieval_hits = sum(1 for row in index_rows if row.get("retrieved"))
    grounded_hits = sum(1 for row in rows if row.get("grounded"))
    correct_hits = sum(1 for row in rows if row.get("correct"))
    retries = sum(1 for row in rows if row.get("retried"))
    latencies = [float(row.get("latency_ms") or 0) for row in rows]
    average_latency = sum(latencies) / n if n else 0.0
    return EvaluationResult(
        routing_accuracy=_ratio(routing_hits, n),
        retrieval_success_rate=_ratio(retrieval_hits, len(index_rows)),
        answer_groundedness=_ratio(grounded_hits, n),
        answer_correctness=_ratio(correct_hits, n),
        average_latency_ms=average_latency,
        retry_rate=_ratio(retries, n),
        n=n,
    )
