"""End-to-end pipeline for index building, Q&A, and evaluation."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from offline_ollama_rag.config import Settings
from offline_ollama_rag.data.documents import load_documents
from offline_ollama_rag.eval.evaluator import load_eval_examples, run_evaluation
from offline_ollama_rag.ollama_client import AsyncOllamaGateway
from offline_ollama_rag.reporting.reporting import render_report, save_predictions, save_summary
from offline_ollama_rag.retrieval.chunking import chunk_documents
from offline_ollama_rag.retrieval.rag_engine import OfflineRAGEngine
from offline_ollama_rag.retrieval.vector_index import (
    VectorIndex,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)


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


async def build_index(settings: Settings, gateway: AsyncOllamaGateway) -> dict[str, Any]:
    """Build and persist vector index from local knowledge files."""
    await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)

    docs = load_documents(settings.resolved_knowledge_dir)
    if not docs:
        raise RuntimeError(
            f"No documents found under {settings.resolved_knowledge_dir}. Add .md or .txt files first."
        )

    chunks = chunk_documents(
        documents=docs,
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
    )
    if not chunks:
        raise RuntimeError("Document chunking produced no chunks.")

    logger.info("Embedding {} chunks with {}", len(chunks), settings.embedding_model)
    embeddings = await gateway.embed_texts(
        settings.embedding_model,
        [chunk.text for chunk in chunks],
        timeout_seconds=settings.embedding_timeout_seconds,
    )

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
            "Index artifacts not found. Run `offline-ollama-rag build-index` first."
        )
    return load_vector_index(settings.embeddings_file, settings.chunks_file)


async def answer_question(settings: Settings, question: str, use_rag: bool) -> dict[str, Any]:
    """Answer a question with or without retrieval context."""
    gateway = AsyncOllamaGateway(settings.ollama_host)
    await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)

    index = load_index(settings)
    engine = OfflineRAGEngine(settings, gateway, index)

    if use_rag:
        answer, retrieved = await engine.answer_with_rag(question)
        return {
            "mode": "rag",
            "question": question,
            "answer": answer,
            "retrieved": _retrieval_to_records(retrieved),
        }

    answer = await engine.answer_without_rag(question)
    return {
        "mode": "baseline",
        "question": question,
        "answer": answer,
        "retrieved": [],
    }


async def evaluate(settings: Settings) -> dict[str, Any]:
    """Run baseline-vs-RAG evaluation and save artifacts."""
    gateway = AsyncOllamaGateway(settings.ollama_host)
    await gateway.ensure_required_models(settings.chat_model, settings.embedding_model)

    index = load_index(settings)
    engine = OfflineRAGEngine(settings, gateway, index)

    examples = load_eval_examples(settings.resolved_evaluation_file)
    logger.info("Running evaluation on {} questions", len(examples))
    rows, summary = await run_evaluation(engine=engine, examples=examples)

    predictions_path = settings.resolved_eval_dir / "predictions.csv"
    summary_path = settings.resolved_eval_dir / "summary.json"
    report_path = settings.resolved_report_dir / "offline_ollama_rag_report.md"

    save_predictions(rows, predictions_path)
    save_summary(summary, summary_path)
    render_report(
        summary=summary,
        rows=rows,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        output_path=report_path,
    )

    return {
        "summary": json.loads(summary.model_dump_json()),
        "predictions_path": predictions_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "report_path": report_path.as_posix(),
    }


async def run_all(settings: Settings) -> dict[str, Any]:
    """Run index build and evaluation pipeline."""
    gateway = AsyncOllamaGateway(settings.ollama_host)
    index_info = await build_index(settings, gateway)
    eval_info = await evaluate(settings)

    payload = {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "index": index_info,
        "evaluation": eval_info,
    }

    output_path = settings.resolved_artifacts_dir / "run_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["run_summary_path"] = output_path.as_posix()
    return payload
