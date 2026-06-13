"""Core RAG pipeline with tracing, latency, cost, and quality hooks."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict
from uuid import uuid4

from opentelemetry.trace import Tracer

from ask_my_docs.evaluation.metrics import retrieval_recall_at_k
from ask_my_docs.llm.ollama import OllamaGenerator
from ask_my_docs.models import RAGAnswer
from ask_my_docs.observability.cost import CostCalculator
from ask_my_docs.observability.metrics_store import MetricsStore, RequestMetricRecord
from ask_my_docs.retrieval.hybrid import HybridRetriever

_CITATION_RE = re.compile(r"\[(?:doc_id\s*[:=]\s*)?([A-Za-z0-9_.:-]+)]", flags=re.IGNORECASE)


class RAGPipeline:
    """Run one question end-to-end with full observability capture."""

    def __init__(
        self,
        retriever: HybridRetriever,
        tracer: Tracer,
        metrics_store: MetricsStore,
        cost_calculator: CostCalculator,
        generator: OllamaGenerator,
        store_raw_questions: bool = False,
    ) -> None:
        self._retriever = retriever
        self._tracer = tracer
        self._metrics_store = metrics_store
        self._cost_calculator = cost_calculator
        self._generator = generator
        self._store_raw_questions = store_raw_questions

    def answer(
        self,
        question: str,
        top_k: int,
        expected_doc_ids: set[str] | None = None,
    ) -> RAGAnswer:
        """Answer a question and persist request-level metrics."""

        request_id = str(uuid4())
        request_start = time.perf_counter()

        with self._tracer.start_as_current_span("rag.request") as request_span:
            request_span.set_attribute("rag.request_id", request_id)
            request_span.set_attribute("rag.top_k", top_k)

            retrieval_start = time.perf_counter()
            with self._tracer.start_as_current_span("rag.retrieve"):
                retrieved = self._retriever.search(query=question, top_k=top_k)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
            retrieved_doc_ids = [item.chunk.doc_id for item in retrieved]

            generation_start = time.perf_counter()
            with self._tracer.start_as_current_span("rag.generate"):
                generation = self._generator.generate(
                    question=question,
                    retrieved=retrieved,
                )
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

            answer_text = generation.text
            prompt_tokens = generation.prompt_tokens
            completion_tokens = generation.completion_tokens
            total_tokens = prompt_tokens + completion_tokens
            estimated_cost = self._cost_calculator.estimate(prompt_tokens, completion_tokens)
            citations = self._extract_citations(answer_text, retrieved_doc_ids)

            total_latency_ms = (time.perf_counter() - request_start) * 1000.0
            trace_id = format(request_span.get_span_context().trace_id, "032x")

            recall_at_k = (
                retrieval_recall_at_k(
                    expected_doc_ids=expected_doc_ids,
                    retrieved_doc_ids=retrieved_doc_ids,
                )
                if expected_doc_ids is not None
                else None
            )

            request_span.set_attribute("rag.model_name", generation.model)
            request_span.set_attribute("rag.latency_ms", round(total_latency_ms, 3))
            request_span.set_attribute("rag.estimated_cost_usd", estimated_cost)
            request_span.set_attribute("rag.retrieved_doc_count", len(retrieved_doc_ids))

            result = RAGAnswer(
                request_id=request_id,
                trace_id=trace_id,
                model_name=generation.model,
                question=question,
                answer=answer_text,
                citations=citations,
                retrieved_doc_ids=retrieved_doc_ids,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
                latency_ms=round(total_latency_ms, 3),
                retrieval_latency_ms=round(retrieval_latency_ms, 3),
                generation_latency_ms=round(generation_latency_ms, 3),
                retrieval_recall_at_k=recall_at_k,
            )

            self._metrics_store.record(
                RequestMetricRecord(
                    request_id=result.request_id,
                    trace_id=result.trace_id,
                    model_name=result.model_name,
                    question=self._question_for_storage(result.question),
                    latency_ms=result.latency_ms,
                    retrieval_latency_ms=result.retrieval_latency_ms,
                    generation_latency_ms=result.generation_latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    estimated_cost_usd=result.estimated_cost_usd,
                    retrieval_recall_at_k=result.retrieval_recall_at_k,
                    answer_f1=None,
                    exact_match=None,
                    timestamp_utc=result.timestamp_utc,
                )
            )

            return result

    def update_quality_metrics(self, request_id: str, answer_f1: float, exact_match: float) -> None:
        """Update quality fields for an existing request metric row."""

        self._metrics_store.update_quality(
            request_id=request_id,
            answer_f1=answer_f1,
            exact_match=exact_match,
        )

    @staticmethod
    def to_payload(result: RAGAnswer) -> dict[str, object]:
        """Convert answer dataclass into JSON-serializable dictionary."""

        return asdict(result)

    def _question_for_storage(self, question: str) -> str:
        if self._store_raw_questions:
            return question
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        return f"<redacted:sha256:{digest}:len={len(question)}>"

    @staticmethod
    def _extract_citations(answer_text: str, retrieved_doc_ids: list[str]) -> list[str]:
        """Extract valid doc citations from answer text with fallback."""

        valid_doc_ids = set(retrieved_doc_ids)
        citation_order: list[str] = []

        for citation in _CITATION_RE.findall(answer_text):
            if citation in valid_doc_ids and citation not in citation_order:
                citation_order.append(citation)

        if citation_order:
            return citation_order

        fallback: list[str] = []
        for doc_id in retrieved_doc_ids:
            if doc_id not in fallback:
                fallback.append(doc_id)
            if len(fallback) >= 2:
                break
        return fallback
