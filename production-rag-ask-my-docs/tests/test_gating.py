"""Unit tests for regression gate behavior."""

from ask_my_docs.evaluation.gating import (
    AbsoluteThresholds,
    GateConfig,
    RegressionThresholds,
    evaluate_gate,
)


def _config() -> GateConfig:
    return GateConfig(
        absolute=AbsoluteThresholds(
            min_answer_f1=0.60,
            min_retrieval_recall_at_k=0.90,
            max_latency_p95_ms=45000.0,
            max_avg_cost_usd=0.001,
        ),
        regression=RegressionThresholds(
            max_answer_f1_drop=0.03,
            max_retrieval_recall_drop=0.02,
            max_latency_p95_increase_ms=5000.0,
            max_avg_cost_increase_usd=0.0001,
        ),
    )


def test_gate_passes_for_good_metrics() -> None:
    result = evaluate_gate(
        current_metrics={
            "answer_f1_mean": 0.82,
            "retrieval_recall_at_k_mean": 1.0,
            "latency_p95_ms": 22000.0,
            "avg_cost_usd": 0.00008,
        },
        baseline_metrics={
            "answer_f1_mean": 0.83,
            "retrieval_recall_at_k_mean": 1.0,
            "latency_p95_ms": 20000.0,
            "avg_cost_usd": 0.00007,
        },
        config=_config(),
    )

    assert result.passed
    assert result.failures == []


def test_gate_fails_when_quality_regresses_too_far() -> None:
    result = evaluate_gate(
        current_metrics={
            "answer_f1_mean": 0.70,
            "retrieval_recall_at_k_mean": 0.95,
            "latency_p95_ms": 22000.0,
            "avg_cost_usd": 0.00009,
        },
        baseline_metrics={
            "answer_f1_mean": 0.82,
            "retrieval_recall_at_k_mean": 1.0,
            "latency_p95_ms": 20000.0,
            "avg_cost_usd": 0.00007,
        },
        config=_config(),
    )

    assert not result.passed
    assert any("answer_f1 dropped" in failure for failure in result.failures)
