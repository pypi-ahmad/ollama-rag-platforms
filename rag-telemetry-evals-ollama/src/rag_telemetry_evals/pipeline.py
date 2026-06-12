"""End-to-end pipeline for index building, Q&A, telemetry, and evaluation."""

from __future__ import annotations

import json
from time import time
from typing import Any

from loguru import logger

from rag_telemetry_evals.config import Settings
from rag_telemetry_evals.data.documents import load_documents
from rag_telemetry_evals.eval.evaluator import load_eval_examples, run_evaluation
from rag_telemetry_evals.ollama_client import AsyncOllamaGateway
from rag_telemetry_evals.reporting.reporting import render_report, save_predictions, save_summary
from rag_telemetry_evals.retrieval.chunking import chunk_documents
from rag_telemetry_evals.retrieval.rag_engine import LocalRAGEngine
from rag_telemetry_evals.retrieval.vector_index import (
    VectorIndex,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)
from rag_telemetry_evals.telemetry.tracer import JsonlTelemetryTracer, summarize_traces


def _retrieval_to_records(retrieved: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in retrieved:
        records.append(
            {
                "score": round(item.score, 4),
                "source": item.chunk.source,
                "chunk_id": item.chunk.chunk_id,
                "text": item.chunk.text,
            }
        )
    return records


async def build_index(
    settings: Settings,
    gateway: AsyncOllamaGateway,
    tracer: JsonlTelemetryTracer,
) -> dict[str, Any]:
    """Build and persist vector index from local knowledge files."""
    await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)

    trace_id = "index-build"

    with tracer.span(trace_id, "load_documents"):
        docs = load_documents(settings.resolved_knowledge_dir)

    if not docs:
        raise RuntimeError(
            f"No documents found under {settings.resolved_knowledge_dir}. Add .md or .txt files first."
        )

    with tracer.span(
        trace_id,
        "chunk_documents",
        {
            "chunk_size_words": settings.chunk_size_words,
            "chunk_overlap_words": settings.chunk_overlap_words,
        },
    ):
        chunks = chunk_documents(
            documents=docs,
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
        )

    if not chunks:
        raise RuntimeError("Document chunking produced no chunks.")

    logger.info("Embedding {} chunks with {}", len(chunks), settings.embedding_model)
    with tracer.span(trace_id, "embed_chunks", {"n_chunks": len(chunks)}):
        embeddings = await gateway.embed_texts(
            settings.embedding_model,
            [chunk.text for chunk in chunks],
            timeout_seconds=settings.embedding_timeout_seconds,
        )

    with tracer.span(trace_id, "persist_index"):
        index = build_vector_index(chunks=chunks, embeddings=embeddings)
        save_vector_index(index, settings.embeddings_file, settings.chunks_file)

    logger.info(
        "Index built with {} chunks and dimension {}",
        len(chunks),
        index.embeddings.shape[1] if index.embeddings.size else 0,
    )

    return {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "embedding_dimension": int(index.embeddings.shape[1]),
        "embeddings_file": settings.embeddings_file.as_posix(),
        "chunks_file": settings.chunks_file.as_posix(),
    }


def load_index(settings: Settings) -> VectorIndex:
    """Load persisted vector index from disk."""
    if not settings.embeddings_file.exists() or not settings.chunks_file.exists():
        raise FileNotFoundError(
            "Index artifacts not found. Run `rag-telemetry-evals build-index` first."
        )
    return load_vector_index(settings.embeddings_file, settings.chunks_file)


async def answer_question(settings: Settings, question: str, use_rag: bool) -> dict[str, Any]:
    """Answer a question with or without retrieval context."""
    gateway = AsyncOllamaGateway(settings.ollama_host)
    await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)

    tracer = JsonlTelemetryTracer(settings.trace_file)
    index = load_index(settings)
    engine = LocalRAGEngine(settings, gateway, index, tracer)
    trace_id = f"ask-{int(time() * 1000)}"

    if use_rag:
        result, retrieved = await engine.answer_with_rag(question, trace_id=trace_id)
        return {
            "mode": "rag",
            "trace_id": trace_id,
            "question": question,
            "answer": result.text,
            "metadata": result.model_dump(),
            "retrieved": _retrieval_to_records(retrieved),
        }

    result = await engine.answer_without_rag(question, trace_id=trace_id)
    return {
        "mode": "baseline",
        "trace_id": trace_id,
        "question": question,
        "answer": result.text,
        "metadata": result.model_dump(),
        "retrieved": [],
    }


async def evaluate(settings: Settings, reset_trace_file: bool = True) -> dict[str, Any]:
    """Run retrieval and answer-quality evaluation, then save artifacts."""
    if reset_trace_file and settings.trace_file.exists():
        settings.trace_file.unlink()

    gateway = AsyncOllamaGateway(settings.ollama_host)
    await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)

    tracer = JsonlTelemetryTracer(settings.trace_file)
    index = load_index(settings)
    engine = LocalRAGEngine(settings, gateway, index, tracer)

    examples = load_eval_examples(settings.resolved_evaluation_file)
    logger.info("Running evaluation on {} questions", len(examples))
    rows, summary = await run_evaluation(
        engine=engine,
        gateway=gateway,
        embedding_model=settings.embedding_model,
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
        examples=examples,
    )

    save_predictions(rows, settings.predictions_file)
    save_summary(summary, settings.summary_file)

    telemetry_summary = summarize_traces(settings.trace_file, settings.telemetry_summary_file)
    render_report(
        summary=summary,
        rows=rows,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        telemetry_summary=telemetry_summary,
        output_path=settings.report_file,
    )

    return {
        "summary": json.loads(summary.model_dump_json()),
        "predictions_path": settings.predictions_file.as_posix(),
        "summary_path": settings.summary_file.as_posix(),
        "report_path": settings.report_file.as_posix(),
        "telemetry_trace_path": settings.trace_file.as_posix(),
        "telemetry_summary_path": settings.telemetry_summary_file.as_posix(),
    }


async def run_all(settings: Settings) -> dict[str, Any]:
    """Run index build and full evaluation pipeline."""
    if settings.trace_file.exists():
        settings.trace_file.unlink()
    if settings.telemetry_summary_file.exists():
        settings.telemetry_summary_file.unlink()

    gateway = AsyncOllamaGateway(settings.ollama_host)
    tracer = JsonlTelemetryTracer(settings.trace_file)

    index_info = await build_index(settings, gateway, tracer)
    eval_info = await evaluate(settings, reset_trace_file=False)

    payload = {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "index": index_info,
        "evaluation": eval_info,
        "run_summary_path": settings.run_summary_file.as_posix(),
    }

    settings.run_summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
