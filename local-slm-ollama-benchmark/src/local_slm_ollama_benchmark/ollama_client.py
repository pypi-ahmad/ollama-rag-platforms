"""Async Ollama API client."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel


LOGGER = logging.getLogger(__name__)


class GenerateResponse(BaseModel):
    """Structured fields returned by Ollama `/api/generate`."""

    model: str
    response: str
    done: bool
    done_reason: str | None = None
    thinking: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaClient:
    """Small async client for local Ollama operations."""

    def __init__(self, host: str, timeout_sec: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            timeout=timeout_sec,
        )

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close HTTP resources."""
        await self._client.aclose()

    async def list_models(self) -> list[str]:
        """Return model names available in the local Ollama registry."""
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        model_entries = payload.get("models", [])
        return [entry["name"] for entry in model_entries if isinstance(entry, dict) and "name" in entry]

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        options: dict[str, Any],
        keep_alive: str,
        think: bool | str | None = None,
    ) -> GenerateResponse:
        """Run one non-streaming generation call.

        Args:
            model: Installed local model name.
            prompt: Prompt text to send.
            options: Ollama options payload (temperature, seed, etc.).
            keep_alive: Keep-alive duration for model residency.

        Returns:
            Parsed `GenerateResponse` model.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
        }
        if think is not None:
            payload["think"] = think

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        parsed = GenerateResponse.model_validate(response.json())
        LOGGER.debug(
            "Generated response",
            extra={
                "model": model,
                "prompt_chars": len(prompt),
                "response_chars": len(parsed.response),
            },
        )
        return parsed
