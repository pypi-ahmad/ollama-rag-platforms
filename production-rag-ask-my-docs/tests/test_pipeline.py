"""Unit tests for RAG pipeline observability behavior."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import duckdb
from opentelemetry import trace

from ask_my_docs.llm.ollama import OllamaGeneration, OllamaGenerator
from ask_my_docs.observability.cost import CostCalculator, TokenPricing
from ask_my_docs.observability.metrics_store import MetricsStore
from ask_my_docs.pipeline import RAGPipeline
from ask_my_docs.retrieval.hybrid import HybridRetriever, load_documents
from ask_my_docs.settings import RetrievalConfig


class _StubGenerator:
    def __init__(self, text: str, model: str = "stub-model") -> None:
        self._text = text
        self._model = model

    def generate(self, question: str, retrieved: object) -> OllamaGeneration:
        del question, retrieved
        return OllamaGeneration(
            model=self._model,
            text=self._text,
            prompt_tokens=10,
            completion_tokens=5,
        )


def _build_retriever(tmp_path: Path) -> HybridRetriever:
    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True)
    (docs_root / "policy.md").write_text("Policy content for test retrieval.", encoding="utf-8")
    documents = load_documents(docs_root)
    return HybridRetriever.from_documents(
        documents=documents,
        config=RetrievalConfig(chunk_size_tokens=64, chunk_overlap_tokens=0),
    )


def test_pipeline_redacts_question_in_metrics_by_default(tmp_path: Path) -> None:
    metrics_store = MetricsStore(db_path=tmp_path / "metrics.duckdb")
    pipeline = RAGPipeline(
        retriever=_build_retriever(tmp_path),
        tracer=trace.get_tracer("test-redaction-default"),
        metrics_store=metrics_store,
        cost_calculator=CostCalculator(
            TokenPricing(prompt_cost_per_1k_tokens=0.001, completion_cost_per_1k_tokens=0.002)
        ),
        generator=cast(OllamaGenerator, _StubGenerator(text="Answer with [policy].")),
        store_raw_questions=False,
    )
    question = "What are our retention rules?"
    result = pipeline.answer(question=question, top_k=1)

    with duckdb.connect(str(tmp_path / "metrics.duckdb")) as connection:
        row = connection.execute(
            "SELECT question FROM rag_request_metrics WHERE request_id = ?",
            [result.request_id],
        ).fetchone()

    assert row is not None
    stored_question = str(row[0])
    assert stored_question.startswith("<redacted:sha256:")
    assert question not in stored_question


def test_pipeline_can_store_raw_question_when_enabled(tmp_path: Path) -> None:
    metrics_store = MetricsStore(db_path=tmp_path / "metrics.duckdb")
    pipeline = RAGPipeline(
        retriever=_build_retriever(tmp_path),
        tracer=trace.get_tracer("test-redaction-enabled"),
        metrics_store=metrics_store,
        cost_calculator=CostCalculator(
            TokenPricing(prompt_cost_per_1k_tokens=0.001, completion_cost_per_1k_tokens=0.002)
        ),
        generator=cast(OllamaGenerator, _StubGenerator(text="Answer with [policy].")),
        store_raw_questions=True,
    )
    question = "What are our retention rules?"
    result = pipeline.answer(question=question, top_k=1)

    with duckdb.connect(str(tmp_path / "metrics.duckdb")) as connection:
        row = connection.execute(
            "SELECT question FROM rag_request_metrics WHERE request_id = ?",
            [result.request_id],
        ).fetchone()

    assert row is not None
    assert str(row[0]) == question


def test_citation_extractor_accepts_doc_id_prefix_format() -> None:
    citations = RAGPipeline._extract_citations(
        answer_text="Answer cites [doc_id: billing_disputes] and [doc_id=incident_sla].",
        retrieved_doc_ids=["billing_disputes", "incident_sla"],
    )
    assert citations == ["billing_disputes", "incident_sla"]
