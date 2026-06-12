"""Unit tests for request cost estimation."""

from ask_my_docs.observability.cost import CostCalculator, TokenPricing


def test_cost_estimation_uses_prompt_and_completion_rates() -> None:
    calculator = CostCalculator(
        TokenPricing(prompt_cost_per_1k_tokens=0.002, completion_cost_per_1k_tokens=0.004)
    )

    cost = calculator.estimate(prompt_tokens=500, completion_tokens=250)

    assert cost == 0.002
