from __future__ import annotations

from multi_agent_system.eval.evaluator import keyword_recall


def test_keyword_recall() -> None:
    score = keyword_recall("Release is Tuesday 16:00 UTC", ["tuesday", "16:00", "utc", "canary"])
    assert round(score, 4) == 0.75
