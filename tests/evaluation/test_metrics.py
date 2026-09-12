"""Evaluation metric unit tests."""

from evaluation.metrics import compute_metrics


def test_compute_metrics() -> None:
    result = compute_metrics(
        [
            {
                "expected_route": "INDEX",
                "predicted_route": "INDEX",
                "retrieved": True,
                "grounded": True,
                "correct": True,
                "retried": False,
                "latency_ms": 100,
            },
            {
                "expected_route": "SEARCH",
                "predicted_route": "GENERAL",
                "retrieved": False,
                "grounded": True,
                "correct": False,
                "retried": True,
                "latency_ms": 50,
            },
        ]
    )
    assert result.n == 2
    assert result.routing_accuracy == 0.5
    assert result.retrieval_success_rate == 1.0
    assert result.retry_rate == 0.5
    assert result.average_latency_ms == 75.0
