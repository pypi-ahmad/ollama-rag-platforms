"""Async Ollama client wrapper with validation and typed metadata."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from ollama import AsyncClient

from rag_telemetry_evals.schemas import ChatResult


class AsyncOllamaGateway:
    """Thin async wrapper around the Ollama Python client."""

    def __init__(self, host: str) -> None:
        self._client = AsyncClient(host=host)

    async def list_model_names(self) -> set[str]:
        """Return all locally available Ollama model names."""
        response = await self._client.list()
        return {model.model for model in response.models if model.model}

    async def ensure_required_models(self, chat_model: str, embedding_model: str) -> None:
        """Validate local model availability."""
        available = await self.list_model_names()
        missing = [model for model in [chat_model, embedding_model] if model not in available]
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "Missing required Ollama model(s): "
                f"{missing_str}. Pull them first, for example: `ollama pull {missing[0]}`"
            )

    async def embed_texts(self, model: str, texts: list[str], timeout_seconds: float) -> np.ndarray:
        """Embed a batch of texts and return float32 matrix [n_texts, dim]."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        response = await asyncio.wait_for(
            self._client.embed(model=model, input=texts),
            timeout=timeout_seconds,
        )
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise RuntimeError("Ollama embed response did not contain embeddings")
        return np.asarray(embeddings, dtype=np.float32)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResult:
        """Run one non-streaming chat completion with metadata."""
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
            prompt_eval_duration_ns=int(getattr(response, "prompt_eval_duration", 0) or 0),
            eval_duration_ns=int(getattr(response, "eval_duration", 0) or 0),
            load_duration_ns=int(getattr(response, "load_duration", 0) or 0),
            done_reason=str(getattr(response, "done_reason", "") or ""),
        )
