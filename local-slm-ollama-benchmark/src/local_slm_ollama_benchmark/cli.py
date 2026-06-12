"""Command-line interface for local SLM chat and benchmarking."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table

from local_slm_ollama_benchmark.benchmark import BenchmarkRunner
from local_slm_ollama_benchmark.config import GenerationConfig, load_benchmark_config
from local_slm_ollama_benchmark.metrics import ns_to_sec, tokens_per_second
from local_slm_ollama_benchmark.ollama_client import OllamaClient


LOGGER = logging.getLogger(__name__)
CONSOLE = Console()


def main() -> None:
    """Entrypoint for the local-slm-ollama-benchmark CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if not hasattr(args, "handler"):
        parser.print_help()
        return

    _configure_logging(args.log_level)

    try:
        args.handler(args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Command failed: %s", exc)
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-slm-ollama-benchmark",
        description="Run local Ollama chat and model benchmark workflows.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    subparsers = parser.add_subparsers(dest="command")

    benchmark_cmd = subparsers.add_parser(
        "benchmark",
        help="Run the benchmark suite on 3 local models and save artifacts.",
    )
    benchmark_cmd.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.toml"),
        help="Path to TOML benchmark config.",
    )
    benchmark_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root directory where run artifacts are written.",
    )
    benchmark_cmd.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run directory name.",
    )
    benchmark_cmd.add_argument(
        "--repeat-count",
        type=int,
        default=None,
        help="Optional override for repeat_count in config.",
    )
    benchmark_cmd.set_defaults(handler=_handle_benchmark)

    chat_cmd = subparsers.add_parser(
        "chat",
        help="Start an interactive local chat session with one Ollama model.",
    )
    chat_cmd.add_argument("--model", required=True, help="Model name, e.g. phi3.5:3.8b")
    chat_cmd.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    chat_cmd.add_argument(
        "--timeout-sec",
        type=float,
        default=180.0,
        help="Request timeout seconds.",
    )
    chat_cmd.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    chat_cmd.add_argument(
        "--num-predict",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    chat_cmd.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generation seed for repeatability.",
    )
    chat_cmd.add_argument(
        "--think",
        default="false",
        choices=["false", "true", "low", "medium", "high"],
        help="Reasoning mode for models that support it.",
    )
    chat_cmd.set_defaults(handler=_handle_chat)

    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _handle_benchmark(args: argparse.Namespace) -> None:
    asyncio.run(_run_benchmark(args))


def _handle_chat(args: argparse.Namespace) -> None:
    asyncio.run(_run_chat(args))


async def _run_benchmark(args: argparse.Namespace) -> None:
    config = load_benchmark_config(args.config)

    if args.repeat_count is not None:
        runtime_override = config.runtime.model_copy(update={"repeat_count": args.repeat_count})
        config = config.model_copy(update={"runtime": runtime_override})

    runner = BenchmarkRunner(config)
    artifacts = await runner.run(output_root=args.output_dir, run_name=args.run_name)

    CONSOLE.print(f"[bold green]Benchmark finished.[/bold green] Artifacts: {artifacts.run_dir}")
    _render_summary_table(artifacts.model_summary_rows)
    CONSOLE.print(f"Raw JSON: {artifacts.raw_json_path}")
    CONSOLE.print(f"Prompt CSV: {artifacts.prompt_csv_path}")
    CONSOLE.print(f"Model CSV: {artifacts.model_csv_path}")
    CONSOLE.print(f"Markdown report: {artifacts.report_path}")


async def _run_chat(args: argparse.Namespace) -> None:
    think_mode: bool | str = False if args.think == "false" else args.think
    options = GenerationConfig(
        temperature=args.temperature,
        num_predict=args.num_predict,
        seed=args.seed,
        think=think_mode,
    )

    CONSOLE.print("Type `/exit` to end the session.")
    async with OllamaClient(host=args.host, timeout_sec=args.timeout_sec) as client:
        installed_models = await client.list_models()
        if args.model not in installed_models:
            raise ValueError(
                f"Model '{args.model}' is not available. Installed models: {installed_models}"
            )

        while True:
            user_input = CONSOLE.input("\n[bold cyan]You[/bold cyan] > ").strip()
            if user_input.lower() in {"/exit", "exit", "quit"}:
                CONSOLE.print("Session closed.")
                return
            if not user_input:
                continue

            response = await client.generate(
                model=args.model,
                prompt=user_input,
                options={
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "num_predict": options.num_predict,
                    "seed": options.seed,
                },
                keep_alive="20m",
                think=options.think,
            )
            CONSOLE.print(f"[bold magenta]{args.model}[/bold magenta] > {response.response.strip()}")

            decode_tps = tokens_per_second(response.eval_count, response.eval_duration)
            CONSOLE.print(
                "[dim]"
                f"latency={_fmt(ns_to_sec(response.total_duration))}s "
                f"decode_tps={_fmt(decode_tps)} "
                f"eval_tokens={response.eval_count}"
                "[/dim]"
            )


def _render_summary_table(rows: list[dict[str, object]]) -> None:
    table = Table(title="Model Quality vs Speed")
    table.add_column("Model", style="bold")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("P95 Latency (s)", justify="right")
    table.add_column("Avg Tok/s", justify="right")
    table.add_column("Avg Quality", justify="right")
    table.add_column("Balanced", justify="right")

    for row in rows:
        table.add_row(
            str(row["model"]),
            str(row["avg_latency_sec"]),
            str(row["p95_latency_sec"]),
            str(row["avg_tokens_per_sec"]),
            str(row["avg_quality_score"]),
            str(row["balanced_score"]),
        )

    CONSOLE.print(table)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
