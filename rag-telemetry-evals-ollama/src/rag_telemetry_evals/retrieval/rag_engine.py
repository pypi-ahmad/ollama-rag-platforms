"""Local RAG engine with structured telemetry spans."""

from __future__ import annotations

from rag_telemetry_evals.config import Settings
from rag_telemetry_evals.ollama_client import AsyncOllamaGateway
from rag_telemetry_evals.retrieval.vector_index import VectorIndex
from rag_telemetry_evals.schemas import ChatResult, RetrievedChunk
from rag_telemetry_evals.telemetry.tracer import JsonlTelemetryTracer

RAG_SYSTEM_PROMPT = (
    "You are a careful enterprise assistant. "
    "Use only the provided context snippets and cite source filenames in your answer. "
    "If the answer is missing from context, explicitly say you do not know from local docs."
)

BASELINE_SYSTEM_PROMPT = (
    "You are a local assistant with no retrieval context. "
    "If uncertain, answer conservatively and avoid inventing details."
)


class LocalRAGEngine:
    """Answer questions with or without retrieval context and telemetry."""

    def __init__(
        self,
        settings: Settings,
        gateway: AsyncOllamaGateway,
        index: VectorIndex,
        tracer: JsonlTelemetryTracer,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._index = index
        self._tracer = tracer

    async def retrieve(self, question: str, trace_id: str) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a question."""
        with self._tracer.span(
            trace_id,
            "retrieve",
            {
                "question_chars": len(question),
                "top_k": self._settings.retrieval_top_k,
            },
        ) as attrs:
            try:
                query_embedding = await self._gateway.embed_texts(
                    model=self._settings.embedding_model,
                    texts=[question],
                    timeout_seconds=min(self._settings.embedding_timeout_seconds, 12.0),
                )
                if query_embedding.size == 0:
                    attrs["n_retrieved"] = 0
                    return []

                retrieved = self._index.search(
                    query_embedding=query_embedding[0],
                    top_k=self._settings.retrieval_top_k,
                )
                attrs["retrieval_mode"] = "vector"
            except TimeoutError:
                attrs["retrieval_mode"] = "keyword_fallback"
                retrieved = self._keyword_fallback(question)
            attrs["n_retrieved"] = len(retrieved)
            if retrieved:
                attrs["top_score"] = round(retrieved[0].score, 6)
            return retrieved

    async def answer_with_rag(self, question: str, trace_id: str) -> tuple[ChatResult, list[RetrievedChunk]]:
        """Generate an answer conditioned on retrieved context."""
        with self._tracer.span(trace_id, "answer_rag", {"question_chars": len(question)}):
            retrieved = await self.retrieve(question=question, trace_id=trace_id)
            context_block = self._build_context_block(retrieved)
            user_prompt = (
                f"Question: {question}\n\n"
                f"Context snippets:\n{context_block}\n\n"
                "Answer in 2-4 sentences. Quote exact thresholds and include source filenames."
            )

            try:
                with self._tracer.span(trace_id, "generate_rag", {"model": self._settings.chat_model}) as attrs:
                    result = await self._gateway.chat(
                        model=self._settings.chat_model,
                        messages=[
                            {"role": "system", "content": RAG_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self._settings.generation_temperature,
                        max_tokens=self._settings.generation_max_tokens,
                        timeout_seconds=self._settings.generation_timeout_seconds,
                    )
                    attrs["prompt_tokens"] = result.prompt_tokens
                    attrs["completion_tokens"] = result.completion_tokens
                    attrs["done_reason"] = result.done_reason
            except TimeoutError:
                fallback_answer = self._extractive_fallback(retrieved)
                result = ChatResult(text=fallback_answer, done_reason="timeout_fallback")

            return result, retrieved

    async def answer_without_rag(self, question: str, trace_id: str) -> ChatResult:
        """Generate baseline answer without retrieval."""
        with self._tracer.span(trace_id, "answer_baseline", {"question_chars": len(question)}):
            try:
                with self._tracer.span(
                    trace_id,
                    "generate_baseline",
                    {"model": self._settings.chat_model},
                ) as attrs:
                    result = await self._gateway.chat(
                        model=self._settings.chat_model,
                        messages=[
                            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        temperature=self._settings.generation_temperature,
                        max_tokens=self._settings.generation_max_tokens,
                        timeout_seconds=self._settings.generation_timeout_seconds,
                    )
                    attrs["prompt_tokens"] = result.prompt_tokens
                    attrs["completion_tokens"] = result.completion_tokens
                    attrs["done_reason"] = result.done_reason
            except TimeoutError:
                result = ChatResult(text="I do not know.", done_reason="timeout_fallback")

            return result

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
            return "I do not know from the local knowledge base."
        top = retrieved[0].chunk
        return (
            f"From {top.source}: {top.text} "
            "Generated via extractive fallback because the local model request timed out."
        )

    def _keyword_fallback(self, question: str) -> list[RetrievedChunk]:
        question_terms = self._tokenize(question)
        if not question_terms:
            return []

        scored: list[RetrievedChunk] = []
        for chunk in self._index.chunks:
            chunk_terms = self._tokenize(chunk.text)
            overlap = len(question_terms & chunk_terms)
            score = overlap / max(1, len(question_terms))
            scored.append(RetrievedChunk(chunk=chunk, score=float(score)))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self._settings.retrieval_top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        clean_tokens: set[str] = set()
        for token in text.split():
            normalized = token.strip(".,:;!?()[]{}\"'").lower()
            if normalized:
                clean_tokens.add(normalized)
        return clean_tokens
