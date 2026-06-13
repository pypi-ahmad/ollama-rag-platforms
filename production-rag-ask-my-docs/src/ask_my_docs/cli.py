"""CLI for ingestion, asking, observability summaries, and regression gating."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from loguru import logger

from ask_my_docs.evaluation.gating import evaluate_gate, load_gate_config
from ask_my_docs.evaluation.runner import (
    load_eval_examples,
    run_evaluation,
    save_eval_report,
)
from ask_my_docs.llm.ollama import OllamaGenerator
from ask_my_docs.observability.cost import CostCalculator, TokenPricing
from ask_my_docs.observability.metrics_store import MetricsStore
from ask_my_docs.observability.tracing import configure_tracing
from ask_my_docs.pipeline import RAGPipeline
from ask_my_docs.retrieval import HybridRetriever, load_documents
from ask_my_docs.settings import get_settings
from ask_my_docs.utils import read_json, write_json

app = typer.Typer(
    help="Production RAG with observability and regression gates",
    pretty_exceptions_enable=False,
)
DEFAULT_EVAL_OUTPUT = Path("artifacts/eval/current_metrics.json")
DEFAULT_BASELINE_PATH = Path("artifacts/baseline_metrics.json")

logger.remove()
logger.add(sys.stderr, level="INFO")


def _exit_with_error(message: str) -> NoReturn:
    logger.error("{}", message)
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _close_metrics_store(metrics_store: MetricsStore | None) -> None:
    if metrics_store is None:
        return
    metrics_store.close()


def _build_metrics_store() -> MetricsStore:
    settings = get_settings()
    return MetricsStore(db_path=settings.metrics_db_path)


def _build_pipeline() -> tuple[RAGPipeline, MetricsStore]:
    settings = get_settings()

    tracer = configure_tracing(service_name=settings.service_name, trace_path=settings.traces_path)
    metrics_store = _build_metrics_store()

    try:
        retriever = HybridRetriever.load(index_dir=settings.index_dir, config=settings.retrieval)
    except Exception:
        metrics_store.close()
        raise

    cost_calculator = CostCalculator(
        TokenPricing(
            prompt_cost_per_1k_tokens=settings.pricing.prompt_cost_per_1k_tokens,
            completion_cost_per_1k_tokens=settings.pricing.completion_cost_per_1k_tokens,
        )
    )
    generator = OllamaGenerator(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        list_timeout_seconds=settings.ollama_list_timeout_seconds,
        configured_model=settings.ollama_model,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        tracer=tracer,
        metrics_store=metrics_store,
        cost_calculator=cost_calculator,
        generator=generator,
        store_raw_questions=settings.observability_store_raw_questions,
    )
    return pipeline, metrics_store


@app.command()
def ingest(
    docs_dir: Annotated[
        Path | None,
        typer.Option(help="Directory with .md/.txt docs"),
    ] = None,
) -> None:
    """Build hybrid retrieval index from source documents."""

    try:
        settings = get_settings()
        source_dir = docs_dir or settings.docs_dir

        documents = load_documents(source_dir)
        retriever = HybridRetriever.from_documents(documents=documents, config=settings.retrieval)
        retriever.save(settings.index_dir)
    except typer.Exit:
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _exit_with_error(str(exc))

    logger.info("Indexed {} docs into {} chunks", len(documents), retriever.chunk_count)
    typer.echo(f"Indexed {len(documents)} docs into {retriever.chunk_count} chunks")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the RAG system")],
    top_k: Annotated[
        int | None,
        typer.Option(help="Number of retrieved chunks", min=1),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional JSON output path"),
    ] = None,
) -> None:
    """Run one RAG query and record observability metrics."""

    metrics_store: MetricsStore | None = None
    try:
        settings = get_settings()
        pipeline, metrics_store = _build_pipeline()
        result = pipeline.answer(question=question, top_k=top_k or settings.retrieval.top_k)
        payload = pipeline.to_payload(result)

        if output is not None:
            write_json(output, payload)

        typer.echo(
            f"answer={result.answer}\n"
            f"model={result.model_name}\n"
            f"citations={', '.join(result.citations)}\n"
            f"latency_ms={result.latency_ms:.2f} p50/p95 tracked in metrics db\n"
            f"estimated_cost_usd={result.estimated_cost_usd:.8f}"
        )
    except typer.Exit:
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _exit_with_error(str(exc))
    finally:
        _close_metrics_store(metrics_store)


@app.command(name="metrics-summary")
def metrics_summary(
    limit: Annotated[int | None, typer.Option(help="Optional row limit", min=1)] = None,
) -> None:
    """Print aggregate observability metrics from DuckDB."""

    metrics_store: MetricsStore | None = None
    try:
        metrics_store = _build_metrics_store()
        summary = metrics_store.summarize(limit=limit)
        typer.echo(
            "\n".join(
                [
                    f"request_count={summary['request_count']:.0f}",
                    f"latency_p50_ms={summary['latency_p50_ms']:.3f}",
                    f"latency_p95_ms={summary['latency_p95_ms']:.3f}",
                    f"avg_cost_usd={summary['avg_cost_usd']:.8f}",
                    f"avg_retrieval_recall_at_k={summary['avg_retrieval_recall_at_k']:.4f}",
                    f"avg_answer_f1={summary['avg_answer_f1']:.4f}",
                    f"avg_exact_match={summary['avg_exact_match']:.4f}",
                ]
            )
        )
    except typer.Exit:
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _exit_with_error(str(exc))
    finally:
        _close_metrics_store(metrics_store)


@app.command()
def eval(
    eval_path: Annotated[
        Path | None,
        typer.Option(help="JSONL eval dataset path"),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Where to store eval report"),
    ] = DEFAULT_EVAL_OUTPUT,
    gate: Annotated[
        bool,
        typer.Option(help="Apply regression gate after evaluation"),
    ] = False,
    baseline: Annotated[
        Path,
        typer.Option(help="Baseline report for regression checks"),
    ] = DEFAULT_BASELINE_PATH,
    thresholds: Annotated[
        Path | None,
        typer.Option(help="Gate YAML config path"),
    ] = None,
    set_baseline: Annotated[
        bool,
        typer.Option(help="Promote this report as new baseline"),
    ] = False,
) -> None:
    """Run quality/latency/cost evaluation and optional regression gate."""

    metrics_store: MetricsStore | None = None
    try:
        settings = get_settings()
        dataset_path = eval_path or settings.eval_path

        pipeline, metrics_store = _build_pipeline()
        examples = load_eval_examples(dataset_path)
        report = run_evaluation(
            pipeline=pipeline,
            examples=examples,
            top_k=settings.retrieval.top_k,
        )
        save_eval_report(path=output, report=report)

        typer.echo(
            "eval complete: "
            f"answer_f1_mean={report.aggregate['answer_f1_mean']:.4f}, "
            f"recall@k_mean={report.aggregate['retrieval_recall_at_k_mean']:.4f}, "
            f"p95_latency_ms={report.aggregate['latency_p95_ms']:.3f}, "
            f"avg_cost_usd={report.aggregate['avg_cost_usd']:.8f}"
        )

        if set_baseline:
            baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, baseline)
            typer.echo(f"baseline updated at {baseline}")

        if gate:
            gate_config = load_gate_config(path=thresholds or settings.thresholds_path)
            baseline_payload: dict[str, object] | None = None
            if baseline.exists():
                baseline_payload = read_json(baseline)

            baseline_metrics = None
            if baseline_payload is not None:
                aggregate_obj = baseline_payload.get("aggregate")
                if not isinstance(aggregate_obj, dict):
                    raise ValueError(f"Baseline report missing 'aggregate' metrics: {baseline}")
                baseline_metrics = {key: float(value) for key, value in aggregate_obj.items()}

            gate_result = evaluate_gate(
                current_metrics={key: float(value) for key, value in report.aggregate.items()},
                baseline_metrics=baseline_metrics,
                config=gate_config,
            )

            if gate_result.passed:
                typer.echo("regression gate: PASS")
            else:
                typer.echo("regression gate: FAIL")
                for failure in gate_result.failures:
                    typer.echo(f"- {failure}")
                raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _exit_with_error(str(exc))
    finally:
        _close_metrics_store(metrics_store)


if __name__ == "__main__":
    app()
