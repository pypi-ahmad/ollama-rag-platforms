from __future__ import annotations

from offline_ollama_rag.eval.evaluator import keyword_recall, nonempty_rate


def test_keyword_recall() -> None:
    answer = "The release train is Tuesday 16:00 UTC."
    score = keyword_recall(answer, ["tuesday", "16:00", "utc", "rollback"])
    assert round(score, 4) == 0.75


def test_nonempty_rate() -> None:
    score = nonempty_rate(["yes", " ", "done"])
    assert round(score, 4) == 0.6667
