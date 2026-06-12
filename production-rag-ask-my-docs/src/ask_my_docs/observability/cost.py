"""Cost estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Pricing model for prompt/completion token billing."""

    prompt_cost_per_1k_tokens: float
    completion_cost_per_1k_tokens: float


class CostCalculator:
    """Estimate cost-per-request from token usage."""

    def __init__(self, pricing: TokenPricing) -> None:
        self._pricing = pricing

    def estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD request cost from token counts."""

        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        prompt_cost = (prompt_tokens / 1000.0) * self._pricing.prompt_cost_per_1k_tokens
        completion_cost = (completion_tokens / 1000.0) * self._pricing.completion_cost_per_1k_tokens
        return round(prompt_cost + completion_cost, 8)
