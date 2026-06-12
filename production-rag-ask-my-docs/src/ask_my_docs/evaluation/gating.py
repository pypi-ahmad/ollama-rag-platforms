"""Regression gate logic for CI quality controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel


class AbsoluteThresholds(BaseModel):
    """Absolute constraints required for every run."""

    min_answer_f1: float
    min_retrieval_recall_at_k: float
    max_latency_p95_ms: float
    max_avg_cost_usd: float


class RegressionThresholds(BaseModel):
    """Allowed degradation relative to baseline."""

    max_answer_f1_drop: float
    max_retrieval_recall_drop: float
    max_latency_p95_increase_ms: float
    max_avg_cost_increase_usd: float


class GateConfig(BaseModel):
    """Top-level gate configuration loaded from YAML."""

    absolute: AbsoluteThresholds
    regression: RegressionThresholds


@dataclass(slots=True)
class GateResult:
    """Result object for gate pass/fail reporting."""

    passed: bool
    failures: list[str]


def load_gate_config(path: Path) -> GateConfig:
    """Load gate configuration from YAML file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GateConfig.model_validate(payload)


def evaluate_gate(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    config: GateConfig,
) -> GateResult:
    """Evaluate absolute and baseline-relative regression gates."""

    failures: list[str] = []

    answer_f1 = current_metrics["answer_f1_mean"]
    recall = current_metrics["retrieval_recall_at_k_mean"]
    p95_latency = current_metrics["latency_p95_ms"]
    avg_cost = current_metrics["avg_cost_usd"]

    if answer_f1 < config.absolute.min_answer_f1:
        failures.append(
            "answer_f1_mean="
            f"{answer_f1:.4f} below "
            f"min_answer_f1={config.absolute.min_answer_f1:.4f}"
        )
    if recall < config.absolute.min_retrieval_recall_at_k:
        failures.append(
            "retrieval_recall_at_k_mean="
            f"{recall:.4f} below "
            "min_retrieval_recall_at_k="
            f"{config.absolute.min_retrieval_recall_at_k:.4f}"
        )
    if p95_latency > config.absolute.max_latency_p95_ms:
        failures.append(
            "latency_p95_ms="
            f"{p95_latency:.2f} above "
            f"max_latency_p95_ms={config.absolute.max_latency_p95_ms:.2f}"
        )
    if avg_cost > config.absolute.max_avg_cost_usd:
        failures.append(
            "avg_cost_usd="
            f"{avg_cost:.8f} above "
            f"max_avg_cost_usd={config.absolute.max_avg_cost_usd:.8f}"
        )

    if baseline_metrics is not None:
        baseline_answer_f1 = baseline_metrics["answer_f1_mean"]
        baseline_recall = baseline_metrics["retrieval_recall_at_k_mean"]
        baseline_p95 = baseline_metrics["latency_p95_ms"]
        baseline_cost = baseline_metrics["avg_cost_usd"]

        if baseline_answer_f1 - answer_f1 > config.regression.max_answer_f1_drop:
            failures.append(
                "answer_f1 dropped by "
                f"{baseline_answer_f1 - answer_f1:.4f}, "
                f"allowed={config.regression.max_answer_f1_drop:.4f}"
            )

        if baseline_recall - recall > config.regression.max_retrieval_recall_drop:
            failures.append(
                "retrieval_recall_at_k dropped by "
                f"{baseline_recall - recall:.4f}, "
                f"allowed={config.regression.max_retrieval_recall_drop:.4f}"
            )

        if p95_latency - baseline_p95 > config.regression.max_latency_p95_increase_ms:
            failures.append(
                "latency_p95_ms increased by "
                f"{p95_latency - baseline_p95:.2f}, "
                f"allowed={config.regression.max_latency_p95_increase_ms:.2f}"
            )

        if avg_cost - baseline_cost > config.regression.max_avg_cost_increase_usd:
            failures.append(
                "avg_cost_usd increased by "
                f"{avg_cost - baseline_cost:.8f}, "
                f"allowed={config.regression.max_avg_cost_increase_usd:.8f}"
            )

    return GateResult(passed=not failures, failures=failures)
