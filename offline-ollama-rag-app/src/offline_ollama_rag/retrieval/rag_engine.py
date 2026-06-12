"""Offline RAG answering engine backed by Ollama and local vector retrieval."""

from __future__ import annotations

from offline_ollama_rag.config import Settings
from offline_ollama_rag.ollama_client import AsyncOllamaGateway
from offline_ollama_rag.retrieval.vector_index import VectorIndex
from offline_ollama_rag.schemas import RetrievedChunk

RAG_SYSTEM_PROMPT = (
    "You are an offline assistant. Answer using only the provided context snippets. "
    "If the answer is not in context, say: 'I don't know based on the local knowledge base.'"
)

BASELINE_SYSTEM_PROMPT = (
    "You are an offline assistant with no external access. "
    "If you are unsure, answer with: 'I don't know.'"
)


class OfflineRAGEngine:
    """Answer questions with or without retrieval context."""

    def __init__(self, settings: Settings, gateway: AsyncOllamaGateway, index: VectorIndex) -> None:
        self._settings = settings
        self._gateway = gateway
        self._index = index

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a question."""
        query_embedding = await self._gateway.embed_texts(
            self._settings.embedding_model,
            [question],
            timeout_seconds=self._settings.embedding_timeout_seconds,
        )
        if query_embedding.size == 0:
            return []

        retrieved = self._index.search(
            query_embedding=query_embedding[0],
            top_k=self._settings.retrieval_top_k,
        )
        return retrieved

    async def answer_with_rag(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        """Generate an answer conditioned on retrieved context."""
        retrieved = await self.retrieve(question)
        context = self._build_context_block(retrieved)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context snippets:\n{context}\n\n"
            "Answer in 2-5 sentences and mention the exact source filename when possible."
        )

        try:
            answer = await self._gateway.chat(
                model=self._settings.chat_model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._settings.generation_temperature,
                max_tokens=self._settings.generation_max_tokens,
                timeout_seconds=self._settings.generation_timeout_seconds,
            )
        except TimeoutError:
            answer = self._extractive_fallback(retrieved)
        return answer, retrieved

    async def answer_without_rag(self, question: str) -> str:
        """Generate a baseline answer without retrieval context."""
        try:
            answer = await self._gateway.chat(
                model=self._settings.chat_model,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=self._settings.generation_temperature,
                max_tokens=self._settings.generation_max_tokens,
                timeout_seconds=self._settings.generation_timeout_seconds,
            )
        except TimeoutError:
            answer = "I don't know."
        return answer

    @staticmethod
    def _build_context_block(retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "(no retrieved context)"

        lines: list[str] = []
        for idx, hit in enumerate(retrieved, start=1):
            lines.append(
                f"[{idx}] source={hit.chunk.source} score={hit.score:.4f} text={hit.chunk.text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extractive_fallback(retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "I don't know based on the local knowledge base."
        top = retrieved[0].chunk
        return (
            f"From {top.source}: {top.text} "
            "This answer was produced via extractive fallback because local generation timed out."
        )
