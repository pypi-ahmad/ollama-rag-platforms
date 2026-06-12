"""Evaluation logic for retrieval and answer quality metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from rag_telemetry_evals.ollama_client import AsyncOllamaGateway
from rag_telemetry_evals.retrieval.rag_engine import LocalRAGEngine
from rag_telemetry_evals.schemas import EvalPrediction, EvalSummary, QAExample, RetrievedChunk


def load_eval_examples(path: Path) -> list[QAExample]:
    """Load evaluation questions from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [QAExample.model_validate(item) for item in payload]


def keyword_recall(answer: str, required_keywords: list[str]) -> float:
    """Compute keyword recall in [0, 1]."""
    if not required_keywords:
        return 0.0

    normalized = answer.lower()
    hits = sum(1 for keyword in required_keywords if keyword.lower() in normalized)
    return hits / len(required_keywords)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _retrieval_hit(retrieved: list[RetrievedChunk], expected_source: str) -> bool:
    return any(hit.chunk.source == expected_source for hit in retrieved)


def _lexical_similarity(answer: str, reference_answer: str) -> float:
    answer_terms = {
        token.strip(".,:;!?()[]{}\"'").lower() for token in answer.split() if token.strip()
    }
    ref_terms = {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in reference_answer.split()
        if token.strip()
    }
    if not answer_terms or not ref_terms:
        return 0.0
    overlap = len(answer_terms & ref_terms)
    union = len(answer_terms | ref_terms)
    return overlap / max(union, 1)


async def _semantic_similarity(
    gateway: AsyncOllamaGateway,
    embedding_model: str,
    answer: str,
    reference_answer: str,
    timeout_seconds: float,
) -> float:
    texts = [answer, reference_answer]
    try:
        embeddings = await gateway.embed_texts(
            model=embedding_model,
            texts=texts,
            timeout_seconds=min(timeout_seconds, 8.0),
        )
    except TimeoutError:
        logger.warning("Embedding timeout while scoring semantic similarity; using lexical fallback")
        return _lexical_similarity(answer, reference_answer)
    if embeddings.shape[0] < 2:
        return 0.0
    return _cosine_similarity(embeddings[0], embeddings[1])


async def run_evaluation(
    engine: LocalRAGEngine,
    gateway: AsyncOllamaGateway,
    embedding_model: str,
    embedding_timeout_seconds: float,
    examples: list[QAExample],
) -> tuple[list[EvalPrediction], EvalSummary]:
    """Run baseline and RAG predictions and aggregate metrics."""
    rows: list[EvalPrediction] = []

    for idx, example in enumerate(examples, start=1):
        logger.info("Evaluating question {}/{} ({})", idx, len(examples), example.question_id)

        baseline_trace = f"eval-{example.question_id}-baseline"
        rag_trace = f"eval-{example.question_id}-rag"

        baseline_result = await engine.answer_without_rag(example.question, trace_id=baseline_trace)
        rag_result, retrieved = await engine.answer_with_rag(example.question, trace_id=rag_trace)

        baseline_keyword = keyword_recall(baseline_result.text, example.required_keywords)
        rag_keyword = keyword_recall(rag_result.text, example.required_keywords)

        baseline_semantic = await _semantic_similarity(
            gateway=gateway,
            embedding_model=embedding_model,
            answer=baseline_result.text,
            reference_answer=example.reference_answer,
            timeout_seconds=embedding_timeout_seconds,
        )
        rag_semantic = await _semantic_similarity(
            gateway=gateway,
            embedding_model=embedding_model,
            answer=rag_result.text,
            reference_answer=example.reference_answer,
            timeout_seconds=embedding_timeout_seconds,
        )

        retrieval_hit = _retrieval_hit(retrieved, example.expected_source)
        rag_mentions_expected_source = example.expected_source.lower() in rag_result.text.lower()

        rows.append(
            EvalPrediction(
                question_id=example.question_id,
                question=example.question,
                expected_source=example.expected_source,
                reference_answer=example.reference_answer,
                baseline_answer=baseline_result.text,
                rag_answer=rag_result.text,
                baseline_keyword_recall=baseline_keyword,
                rag_keyword_recall=rag_keyword,
                baseline_semantic_similarity=baseline_semantic,
                rag_semantic_similarity=rag_semantic,
                retrieval_hit=retrieval_hit,
                rag_mentions_expected_source=rag_mentions_expected_source,
                baseline_latency_ms=baseline_result.latency_ms,
                rag_latency_ms=rag_result.latency_ms,
                baseline_prompt_tokens=baseline_result.prompt_tokens,
                rag_prompt_tokens=rag_result.prompt_tokens,
                baseline_completion_tokens=baseline_result.completion_tokens,
                rag_completion_tokens=rag_result.completion_tokens,
            )
        )

    summary = EvalSummary(
        n_questions=len(rows),
        retrieval_hit_rate=_safe_mean([1.0 if row.retrieval_hit else 0.0 for row in rows]),
        rag_mentions_expected_source_rate=_safe_mean(
            [1.0 if row.rag_mentions_expected_source else 0.0 for row in rows]
        ),
        baseline_keyword_recall_mean=_safe_mean([row.baseline_keyword_recall for row in rows]),
        rag_keyword_recall_mean=_safe_mean([row.rag_keyword_recall for row in rows]),
        keyword_recall_gain=_safe_mean([row.rag_keyword_recall for row in rows])
        - _safe_mean([row.baseline_keyword_recall for row in rows]),
        baseline_semantic_similarity_mean=_safe_mean(
            [row.baseline_semantic_similarity for row in rows]
        ),
        rag_semantic_similarity_mean=_safe_mean([row.rag_semantic_similarity for row in rows]),
        semantic_similarity_gain=_safe_mean([row.rag_semantic_similarity for row in rows])
        - _safe_mean([row.baseline_semantic_similarity for row in rows]),
        baseline_latency_ms_mean=_safe_mean([row.baseline_latency_ms for row in rows]),
        rag_latency_ms_mean=_safe_mean([row.rag_latency_ms for row in rows]),
        baseline_total_tokens_mean=_safe_mean(
            [float(row.baseline_prompt_tokens + row.baseline_completion_tokens) for row in rows]
        ),
        rag_total_tokens_mean=_safe_mean(
            [float(row.rag_prompt_tokens + row.rag_completion_tokens) for row in rows]
        ),
    )

    return rows, summary
