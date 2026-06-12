"""Async Ollama client wrapper with safe metadata parsing."""

from __future__ import annotations

import asyncio
from typing import Any

from ollama import AsyncClient

from multi_agent_system.schemas import ChatResult


class AsyncOllamaGateway:
    """Thin async wrapper around Ollama API for chat operations."""

    def __init__(self, host: str) -> None:
        self._client = AsyncClient(host=host)

    async def list_model_names(self) -> set[str]:
        """Return all locally available model names."""
        response = await self._client.list()
        return {model.model for model in response.models if model.model}

    async def ensure_required_models(self, planner_model: str, reviewer_model: str) -> None:
        """Ensure configured local models are available."""
        available = await self.list_model_names()
        missing = [m for m in [planner_model, reviewer_model] if m not in available]
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "Missing required Ollama model(s): "
                f"{missing_str}. Pull them first, for example: `ollama pull {missing[0]}`"
            )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResult:
        """Execute one non-streaming chat completion."""
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        response = await asyncio.wait_for(
            self._client.chat(
                model=model,
                messages=messages,
                options=options,
            ),
            timeout=timeout_seconds,
        )

        message = getattr(response, "message", None)
        text = ""
        if message is not None:
            text = (getattr(message, "content", "") or "").strip()

        return ChatResult(
            text=text,
            prompt_tokens=int(getattr(response, "prompt_eval_count", 0) or 0),
            completion_tokens=int(getattr(response, "eval_count", 0) or 0),
            total_duration_ns=int(getattr(response, "total_duration", 0) or 0),
            done_reason=str(getattr(response, "done_reason", "") or ""),
            fallback_used=False,
        )
