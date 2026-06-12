"""Benchmark runner for offline Ollama model comparison."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

from local_slm_ollama_benchmark.config import BenchmarkConfig, CostConfig, PromptCase
from local_slm_ollama_benchmark.metrics import ns_to_sec, percentile, tokens_per_second
from local_slm_ollama_benchmark.ollama_client import GenerateResponse, OllamaClient
from local_slm_ollama_benchmark.quality import evaluate_response


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptRunRow:
    """A single prompt inference observation."""

    model: str
    case_id: str
    case_title: str
    iteration: int
    wall_time_sec: float
    total_duration_sec: float | None
    load_duration_sec: float | None
    prompt_eval_duration_sec: float | None
    eval_duration_sec: float | None
    eval_tokens: int | None
    tokens_per_sec: float | None
    quality_score: float
    output_words: int
    response: str
    quality_details: dict[str, float | int | bool]


@dataclass(slots=True)
class BenchmarkArtifacts:
    """File paths emitted by one benchmark run."""

    run_dir: Path
    raw_json_path: Path
    prompt_csv_path: Path
    model_csv_path: Path
    report_path: Path
    model_summary_rows: list[dict[str, Any]]


class BenchmarkRunner:
    """Orchestrates real benchmark execution against a local Ollama server."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    async def run(self, output_root: Path, run_name: str | None = None) -> BenchmarkArtifacts:
        """Execute the full benchmark and persist artifacts.

        Example:
            >>> from pathlib import Path
            >>> config = load_benchmark_config(Path(\"configs/benchmark.toml\"))
            >>> runner = BenchmarkRunner(config)
            >>> # asyncio.run(runner.run(Path(\"artifacts\")))
        """
        run_started = datetime.now(timezone.utc)
        run_id = run_name or run_started.strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        prompt_rows: list[PromptRunRow] = []
        host = self.config.ollama_host.rstrip("/")

        async with OllamaClient(host=host, timeout_sec=self.config.runtime.request_timeout_sec) as client:
            installed_models = await client.list_models()
            missing_models = [name for name in self.config.models if name not in installed_models]
            if missing_models:
                raise RuntimeError(
                    "Missing models in local Ollama registry: "
                    f"{missing_models}. Install with `ollama pull <model>`."
                )

            for model_name in self.config.models:
                await self._warmup_model(client=client, model_name=model_name)
                for case in self.config.prompts:
                    for iteration in range(1, self.config.runtime.repeat_count + 1):
                        row = await self._run_single_prompt(
                            client=client,
                            model_name=model_name,
                            case=case,
                            iteration=iteration,
                        )
                        prompt_rows.append(row)

                        LOGGER.info(
                            "model=%s case=%s iter=%s latency=%.2fs tps=%s quality=%.3f",
                            model_name,
                            case.id,
                            iteration,
                            row.wall_time_sec,
                            "n/a" if row.tokens_per_sec is None else f"{row.tokens_per_sec:.2f}",
                            row.quality_score,
                        )

        model_summary_rows = _summarize_models(
            rows=prompt_rows,
            baseline_model=self.config.models[0],
            cost_config=self.config.cost,
        )

        report_notes = _build_tradeoff_notes(model_summary_rows)

        raw_json_path = run_dir / "raw_results.json"
        prompt_csv_path = run_dir / "prompt_runs.csv"
        model_csv_path = run_dir / "model_summary.csv"
        report_path = run_dir / "benchmark_report.md"

        payload = {
            "run_id": run_id,
            "started_at_utc": run_started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "ollama_host": host,
            "config": self.config.model_dump(),
            "system": collect_system_info(),
            "prompt_runs": [
                {
                    **asdict(row),
                    "wall_time_sec": round(row.wall_time_sec, 4),
                    "tokens_per_sec": None if row.tokens_per_sec is None else round(row.tokens_per_sec, 4),
                }
                for row in prompt_rows
            ],
            "model_summary": model_summary_rows,
            "tradeoff_notes": report_notes,
        }

        raw_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_prompt_csv(prompt_csv_path, prompt_rows)
        _write_model_csv(model_csv_path, model_summary_rows)
        _write_markdown_report(report_path, payload)

        return BenchmarkArtifacts(
            run_dir=run_dir,
            raw_json_path=raw_json_path,
            prompt_csv_path=prompt_csv_path,
            model_csv_path=model_csv_path,
            report_path=report_path,
            model_summary_rows=model_summary_rows,
        )

    async def _warmup_model(self, client: OllamaClient, model_name: str) -> None:
        LOGGER.info("warming up model=%s", model_name)
        await client.generate(
            model=model_name,
            prompt=self.config.runtime.warmup_prompt,
            options=self._generation_options(),
            keep_alive=self.config.runtime.keep_alive,
            think=self.config.generation.think,
        )

    async def _run_single_prompt(
        self,
        client: OllamaClient,
        model_name: str,
        case: PromptCase,
        iteration: int,
    ) -> PromptRunRow:
        start = time.perf_counter()
        response = await client.generate(
            model=model_name,
            prompt=case.prompt,
            options=self._generation_options(),
            keep_alive=self.config.runtime.keep_alive,
            think=self.config.generation.think,
        )
        wall_time = time.perf_counter() - start

        quality = evaluate_response(case=case, response_text=response.response)
        return _to_prompt_row(
            model_name=model_name,
            case=case,
            iteration=iteration,
            wall_time_sec=wall_time,
            response=response,
            quality_score=quality.score,
            quality_details=quality.details,
        )

    def _generation_options(self) -> dict[str, Any]:
        generation = self.config.generation
        return {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "num_predict": generation.num_predict,
            "seed": generation.seed,
        }


def _to_prompt_row(
    *,
    model_name: str,
    case: PromptCase,
    iteration: int,
    wall_time_sec: float,
    response: GenerateResponse,
    quality_score: float,
    quality_details: dict[str, float | int | bool],
) -> PromptRunRow:
    output_words = len([token for token in response.response.split() if token.strip()])
    row = PromptRunRow(
        model=model_name,
        case_id=case.id,
        case_title=case.title,
        iteration=iteration,
        wall_time_sec=wall_time_sec,
        total_duration_sec=ns_to_sec(response.total_duration),
        load_duration_sec=ns_to_sec(response.load_duration),
        prompt_eval_duration_sec=ns_to_sec(response.prompt_eval_duration),
        eval_duration_sec=ns_to_sec(response.eval_duration),
        eval_tokens=response.eval_count,
        tokens_per_sec=tokens_per_second(response.eval_count, response.eval_duration),
        quality_score=quality_score,
        output_words=output_words,
        response=response.response,
        quality_details=quality_details,
    )
    return row


def _summarize_models(
    *, rows: list[PromptRunRow], baseline_model: str, cost_config: CostConfig
) -> list[dict[str, Any]]:
    grouped: dict[str, list[PromptRunRow]] = defaultdict(list)
    for row in rows:
        grouped[row.model].append(row)

    if baseline_model not in grouped:
        raise ValueError(f"Baseline model '{baseline_model}' is not present in benchmark rows.")

    baseline_latency = statistics.fmean(row.wall_time_sec for row in grouped[baseline_model])
    baseline_quality = statistics.fmean(row.quality_score for row in grouped[baseline_model])

    avg_tps_lookup: dict[str, float] = {}
    avg_quality_lookup: dict[str, float] = {}
    for model_name, model_rows in grouped.items():
        tps_values = [row.tokens_per_sec for row in model_rows if row.tokens_per_sec is not None]
        avg_tps_lookup[model_name] = statistics.fmean(tps_values) if tps_values else 0.0
        avg_quality_lookup[model_name] = statistics.fmean(row.quality_score for row in model_rows)

    max_avg_tps = max(avg_tps_lookup.values()) if avg_tps_lookup else 1.0
    max_avg_quality = max(avg_quality_lookup.values()) if avg_quality_lookup else 1.0

    summary_rows: list[dict[str, Any]] = []
    for model_name in grouped:
        model_rows = grouped[model_name]
        latencies = [row.wall_time_sec for row in model_rows]
        quality_scores = [row.quality_score for row in model_rows]
        tps_values = [row.tokens_per_sec for row in model_rows if row.tokens_per_sec is not None]

        avg_latency = statistics.fmean(latencies)
        avg_quality = statistics.fmean(quality_scores)
        avg_tps = statistics.fmean(tps_values) if tps_values else 0.0

        speed_norm = avg_tps / max_avg_tps if max_avg_tps > 0 else 0.0
        quality_norm = avg_quality / max_avg_quality if max_avg_quality > 0 else 0.0
        balanced_score = 0.6 * quality_norm + 0.4 * speed_norm

        estimated_cost = _estimate_benchmark_electricity_cost(model_rows, cost_config)

        summary_rows.append(
            {
                "model": model_name,
                "sample_count": len(model_rows),
                "avg_latency_sec": round(avg_latency, 4),
                "p95_latency_sec": round(percentile(latencies, 95), 4),
                "avg_tokens_per_sec": round(avg_tps, 4),
                "avg_quality_score": round(avg_quality, 4),
                "quality_delta_vs_baseline": round(avg_quality - baseline_quality, 4),
                "speedup_vs_baseline": round(baseline_latency / avg_latency, 4),
                "balanced_score": round(balanced_score, 4),
                "estimated_benchmark_cost_usd": (
                    None if estimated_cost is None else round(estimated_cost, 6)
                ),
            }
        )

    summary_rows.sort(key=lambda row: row["balanced_score"], reverse=True)
    return summary_rows


def _estimate_benchmark_electricity_cost(
    rows: list[PromptRunRow], cost_config: CostConfig
) -> float | None:
    if (
        cost_config.electricity_rate_usd_per_kwh is None
        or cost_config.assumed_power_watts is None
    ):
        return None

    total_runtime_hours = sum(row.wall_time_sec for row in rows) / 3600
    power_kw = cost_config.assumed_power_watts / 1000
    energy_kwh = power_kw * total_runtime_hours
    return energy_kwh * cost_config.electricity_rate_usd_per_kwh


def _build_tradeoff_notes(model_summary_rows: list[dict[str, Any]]) -> list[str]:
    fastest = max(model_summary_rows, key=lambda row: row["avg_tokens_per_sec"])
    best_quality = max(model_summary_rows, key=lambda row: row["avg_quality_score"])
    best_balance = max(model_summary_rows, key=lambda row: row["balanced_score"])

    notes = [
        (
            f"Fastest decode throughput: {fastest['model']} "
            f"at {fastest['avg_tokens_per_sec']} tok/s."
        ),
        (
            f"Best average quality proxy: {best_quality['model']} "
            f"with score {best_quality['avg_quality_score']} (0-1 scale)."
        ),
        (
            f"Best quality/speed balance: {best_balance['model']} "
            f"(balanced_score={best_balance['balanced_score']})."
        ),
        "Privacy: all prompts and outputs stay on-device when Ollama runs locally.",
        "Latency: smaller models usually reduce p95 latency but may fail structure/correctness checks.",
        "Cost: local inference avoids per-token API billing but still incurs hardware and electricity cost.",
    ]
    return notes


def _write_prompt_csv(path: Path, rows: list[PromptRunRow]) -> None:
    fieldnames = [
        "model",
        "case_id",
        "case_title",
        "iteration",
        "wall_time_sec",
        "total_duration_sec",
        "load_duration_sec",
        "prompt_eval_duration_sec",
        "eval_duration_sec",
        "eval_tokens",
        "tokens_per_sec",
        "quality_score",
        "output_words",
    ]

    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "case_id": row.case_id,
                    "case_title": row.case_title,
                    "iteration": row.iteration,
                    "wall_time_sec": round(row.wall_time_sec, 4),
                    "total_duration_sec": _safe_round(row.total_duration_sec),
                    "load_duration_sec": _safe_round(row.load_duration_sec),
                    "prompt_eval_duration_sec": _safe_round(row.prompt_eval_duration_sec),
                    "eval_duration_sec": _safe_round(row.eval_duration_sec),
                    "eval_tokens": row.eval_tokens,
                    "tokens_per_sec": _safe_round(row.tokens_per_sec),
                    "quality_score": row.quality_score,
                    "output_words": row.output_words,
                }
            )


def _write_model_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    system = payload["system"]
    model_summary = payload["model_summary"]
    notes = payload["tradeoff_notes"]

    lines: list[str] = []
    lines.append("# Local SLM Benchmark Report")
    lines.append("")
    lines.append(f"- Run ID: `{payload['run_id']}`")
    lines.append(f"- Started (UTC): `{payload['started_at_utc']}`")
    lines.append(f"- Completed (UTC): `{payload['completed_at_utc']}`")
    lines.append(f"- Ollama host: `{payload['ollama_host']}`")
    lines.append("")

    lines.append("## Hardware and Runtime")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for key, value in system.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| Model | Samples | Avg Latency (s) | P95 Latency (s) | Avg Tok/s | Avg Quality | Speedup vs Baseline | Quality Delta vs Baseline | Balanced |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in model_summary:
        lines.append(
            "| {model} | {sample_count} | {avg_latency_sec} | {p95_latency_sec} | {avg_tokens_per_sec} | "
            "{avg_quality_score} | {speedup_vs_baseline} | {quality_delta_vs_baseline} | {balanced_score} |".format(
                **row
            )
        )
    lines.append("")

    lines.append("## Quality vs Speed Tradeoffs")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Cost Assumption")
    lines.append("")
    lines.append(
        "- `estimated_benchmark_cost_usd` is only computed if `assumed_power_watts` and "
        "`electricity_rate_usd_per_kwh` are set in config."
    )
    lines.append(
        "- This estimate covers the benchmark run itself, not full lifecycle costs (hardware purchase, maintenance)."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def collect_system_info() -> dict[str, str]:
    """Collect hardware/software context for reproducible benchmark records."""
    cpu_model = _get_cpu_model()
    ram_gb = _get_ram_gb()
    gpu_name = _get_gpu_name()

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cores": str(os.cpu_count() or "unknown"),
        "memory_gb": f"{ram_gb:.2f}" if ram_gb is not None else "unknown",
        "gpu": gpu_name,
        "ollama_version": _run_command(["ollama", "--version"]) or "unknown",
    }


def _get_cpu_model() -> str:
    lscpu_output = _run_command(["lscpu"])
    if not lscpu_output:
        return platform.processor() or "unknown"

    for line in lscpu_output.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()

    return platform.processor() or "unknown"


def _get_ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                parts = line.split()
                mem_kb = float(parts[1])
                return mem_kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return None


def _get_gpu_name() -> str:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
    )
    if query:
        return query.replace("\n", "; ")
    return "not detected"


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
