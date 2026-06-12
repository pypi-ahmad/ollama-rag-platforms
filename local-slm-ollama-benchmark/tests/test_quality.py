from local_slm_ollama_benchmark.config import PromptCase
from local_slm_ollama_benchmark.quality import evaluate_response


def test_keyword_quality_scoring_hits_all_keywords() -> None:
    case = PromptCase(
        id="privacy",
        title="Privacy",
        prompt="x",
        evaluation_type="keywords",
        expected_keywords=["privacy", "latency", "cost"],
    )

    result = evaluate_response(
        case,
        "Local hosting improves privacy and latency, and can reduce cost at scale.",
    )
    assert result.score == 1.0
    assert result.details["keyword_hits"] == 3


def test_json_key_scoring_with_wrapped_json_block() -> None:
    case = PromptCase(
        id="json",
        title="JSON",
        prompt="x",
        evaluation_type="json_keys",
        required_json_keys=["root_cause", "severity", "mitigation"],
    )

    result = evaluate_response(
        case,
        "Use this object: {\"root_cause\": \"cache\", \"severity\": \"high\", \"mitigation\": \"rollback\"}",
    )
    assert result.score == 1.0
    assert result.details["json_valid"] is True


def test_exact_match_scoring() -> None:
    case = PromptCase(
        id="math",
        title="Math",
        prompt="x",
        evaluation_type="exact_match",
        expected_answer="372",
    )

    result = evaluate_response(case, "372")
    assert result.score == 1.0
