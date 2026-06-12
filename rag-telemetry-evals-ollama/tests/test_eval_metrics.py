from __future__ import annotations

from rag_telemetry_evals.eval.evaluator import keyword_recall


def test_keyword_recall() -> None:
    answer = "The release train is Tuesday 16:00 UTC."
    score = keyword_recall(answer, ["tuesday", "16:00", "utc", "rollback"])
    assert round(score, 4) == 0.75
