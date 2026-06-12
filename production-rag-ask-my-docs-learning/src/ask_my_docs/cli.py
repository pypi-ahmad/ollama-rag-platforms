"""CLI entrypoint for ingesting docs, serving Q&A, and CI evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import typer

from ask_my_docs.evaluation import (
    apply_gate,
    evaluate_engine,
    load_eval_dataset,
    load_thresholds,
    save_eval_report,
)
from ask_my_docs.indexing import build_and_save_indexes
from ask_my_docs.ingestion import build_chunks_from_docs
from ask_my_docs.logging_config import configure_logging, logger
from ask_my_docs.pipeline import AskMyDocsEngine
from ask_my_docs.settings import SETTINGS
from ask_my_docs.types import RetrievedChunk

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_engine(index_dir: Path) -> AskMyDocsEngine:
    return AskMyDocsEngine.from_index(
        index_dir=index_dir,
        embedding_model=SETTINGS.embedding_model,
        reranker_model=SETTINGS.reranker_model,
        bm25_weight=SETTINGS.bm25_weight,
        vector_weight=SETTINGS.vector_weight,
        rrf_k=SETTINGS.hybrid_rrf_k,
        candidate_pool_size=SETTINGS.candidate_pool_size,
        default_top_k=SETTINGS.default_top_k,
    )


@app.command("build-index")
def build_index(
    docs_dir: Path = typer.Option(SETTINGS.docs_dir, help="Directory with domain docs"),
    index_dir: Path = typer.Option(SETTINGS.index_dir, help="Artifact directory for indexes"),
    chunk_size: int = typer.Option(SETTINGS.chunk_size_words, help="Chunk size in words"),
    chunk_overlap: int = typer.Option(SETTINGS.chunk_overlap_words, help="Chunk overlap in words"),
) -> None:
    """Ingest docs and build BM25/vector/reranker-ready retrieval artifacts."""
    configure_logging()
    logger.info("Building index from docs={} to index_dir={}", docs_dir, index_dir)

    chunks = build_chunks_from_docs(
        docs_dir=docs_dir,
        chunk_size_words=chunk_size,
        chunk_overlap_words=chunk_overlap,
    )

    chunks_df = pl.DataFrame([chunk.model_dump() for chunk in chunks])
    build_and_save_indexes(
        chunks_df=chunks_df,
        index_dir=index_dir,
        embedding_model=SETTINGS.embedding_model,
        reranker_model=SETTINGS.reranker_model,
    )
    typer.echo(f"Index build complete: {index_dir}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask your document corpus"),
    index_dir: Path = typer.Option(SETTINGS.index_dir, help="Index directory"),
    top_k: int = typer.Option(SETTINGS.default_top_k, help="Final contexts after reranking"),
) -> None:
    """Ask the indexed knowledge base and print citation-backed answer."""
    configure_logging()
    engine = _load_engine(index_dir)
    response = engine.ask(question=question, top_k=top_k)
    typer.echo(response.answer)


@app.command()
def evaluate(
    dataset: Path = typer.Option(Path("data/eval/qa.yaml"), help="Eval dataset YAML"),
    index_dir: Path = typer.Option(SETTINGS.index_dir, help="Index directory"),
    thresholds: Path = typer.Option(
        Path("configs/eval_thresholds.yaml"), help="Thresholds used for CI gate"
    ),
    output: Path = typer.Option(
        Path("artifacts/eval/report.json"),
        help="Evaluation report output",
    ),
    gate: bool = typer.Option(False, help="Fail with non-zero exit if thresholds are not met"),
) -> None:
    """Run retrieval/citation evaluation and optionally enforce pass/fail gates."""
    configure_logging()
    engine = _load_engine(index_dir)
    dataset_obj = load_eval_dataset(dataset)

    report = evaluate_engine(engine=engine, dataset=dataset_obj, retrieval_k=5)
    metric_map = report["metrics"]
    assert isinstance(metric_map, dict)
    thresholds_map = load_thresholds(thresholds)

    passed, failures = apply_gate(
        metrics={str(k): float(v) for k, v in metric_map.items()}, thresholds=thresholds_map
    )

    report["thresholds"] = thresholds_map
    report["gate_passed"] = passed
    report["gate_failures"] = failures
    save_eval_report(report, output)

    typer.echo("Evaluation metrics:")
    for key, value in metric_map.items():
        typer.echo(f"- {key}: {float(value):.4f}")

    if gate and not passed:
        typer.echo("\nGate failed:")
        for failure in failures:
            typer.echo(f"- {failure}")
        raise typer.Exit(code=1)

    typer.echo(f"\nEvaluation report: {output}")
    typer.echo("Gate status: PASS" if passed else "Gate status: FAIL")


@app.command("inspect-retrieval")
def inspect_retrieval(
    question: str = typer.Argument(..., help="Question to inspect retrieval internals"),
    index_dir: Path = typer.Option(SETTINGS.index_dir, help="Index directory"),
    top_k: int = typer.Option(5, help="Top chunks to display"),
) -> None:
    """Print retrieval scores to learn hybrid + rerank behavior."""
    configure_logging()
    engine = _load_engine(index_dir)
    traced = engine.retriever.retrieve_with_traces(question, top_k=top_k)
    top_chunks = cast(list[RetrievedChunk], traced["top_chunks"])

    def _fmt(value: float | None, ndigits: int) -> str:
        if value is None:
            return "-"
        return f"{value:.{ndigits}f}"

    for chunk in top_chunks:
        assert hasattr(chunk, "chunk_id")
        typer.echo(
            f"rank={chunk.rank} chunk_id={chunk.chunk_id} "
            f"bm25={_fmt(chunk.bm25_score, 4)} "
            f"vector={_fmt(chunk.vector_score, 4)} "
            f"hybrid={_fmt(chunk.hybrid_score, 6)} "
            f"rerank={_fmt(chunk.rerank_score, 4)}"
        )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Run FastAPI server for interactive use."""
    configure_logging()
    import uvicorn

    uvicorn.run("ask_my_docs.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
