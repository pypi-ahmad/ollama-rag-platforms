"""Metric utilities for local benchmark runs."""

from __future__ import annotations

import math


def ns_to_sec(duration_ns: int | None) -> float | None:
    """Convert nanoseconds to seconds."""
    if duration_ns is None:
        return None
    return duration_ns / 1_000_000_000


def tokens_per_second(eval_count: int | None, eval_duration_ns: int | None) -> float | None:
    """Compute decode throughput from Ollama eval stats."""
    if eval_count is None or eval_duration_ns is None or eval_duration_ns <= 0:
        return None
    return eval_count / (eval_duration_ns / 1_000_000_000)


def percentile(values: list[float], p: float) -> float:
    """Return percentile using linear interpolation."""
    if not values:
        return 0.0

    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100)
    low = math.floor(rank)
    high = math.ceil(rank)

    if low == high:
        return sorted_values[low]

    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight
