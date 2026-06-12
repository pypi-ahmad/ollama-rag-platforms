"""Heuristic quality scoring for deterministic local benchmark prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from local_slm_ollama_benchmark.config import PromptCase


@dataclass(slots=True)
class QualityResult:
    """Quality score and detailed rubric signals for one response."""

    score: float
    details: dict[str, float | int | bool]


def evaluate_response(case: PromptCase, response_text: str) -> QualityResult:
    """Score a model response according to the prompt case rubric.

    Args:
        case: Prompt definition including evaluation instructions.
        response_text: Raw model response string.

    Returns:
        `QualityResult` with score in [0, 1] and rubric diagnostics.

    Example:
        >>> case = PromptCase(
        ...     id="math",
        ...     title="Math",
        ...     prompt="x",
        ...     evaluation_type="exact_match",
        ...     expected_answer="372",
        ... )
        >>> evaluate_response(case, "372").score
        1.0
    """
    response = response_text.strip()
    lowered = response.lower()

    base_score = 0.0
    details: dict[str, float | int | bool] = {}

    if case.evaluation_type == "keywords":
        hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in lowered)
        total = len(case.expected_keywords)
        base_score = hits / total if total > 0 else 0.0
        details = {
            "keyword_hits": hits,
            "keyword_total": total,
            "keyword_hit_ratio": round(base_score, 4),
        }

    elif case.evaluation_type == "json_keys":
        parsed = _try_parse_json(response)
        if parsed is None or not isinstance(parsed, dict):
            base_score = 0.0
            details = {"json_valid": False, "required_keys_present": 0, "required_keys_total": len(case.required_json_keys)}
        else:
            present = sum(1 for key in case.required_json_keys if key in parsed)
            total = len(case.required_json_keys)
            base_score = present / total if total > 0 else 0.0
            details = {
                "json_valid": True,
                "required_keys_present": present,
                "required_keys_total": total,
            }

    elif case.evaluation_type == "exact_match":
        expected = (case.expected_answer or "").strip()
        matched = response == expected
        base_score = 1.0 if matched else 0.0
        details = {"exact_match": matched}

    final_score = base_score
    if case.max_words is not None:
        words = _count_words(response)
        details["word_count"] = words
        details["max_words"] = case.max_words

        if words > case.max_words:
            # Soft penalty for verbosity; keeps score comparable across models.
            brevity_penalty = case.max_words / max(words, 1)
            final_score *= brevity_penalty
            details["brevity_penalty"] = round(brevity_penalty, 4)
        else:
            details["brevity_penalty"] = 1.0

    return QualityResult(score=round(max(0.0, min(1.0, final_score)), 4), details=details)


def _count_words(text: str) -> int:
    return len([token for token in text.split() if token.strip()])


def _try_parse_json(text: str) -> dict[str, object] | list[object] | None:
    """Try parsing direct JSON, then the first balanced JSON object in text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _extract_first_json_object(text)
        if candidate is None:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None
