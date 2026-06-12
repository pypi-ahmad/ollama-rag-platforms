# Project 6: Multi-Agent System using Local Ollama

Notebook-based, production-style multi-agent architecture with local models, deterministic fallbacks, telemetry, and evaluation.

- Planner model: `phi3.5:3.8b`
- Reviewer model: `functiongemma:270m`
- Python: `3.12.10`
- Env/package manager: `uv`

## What this project demonstrates

- End-to-end orchestration across specialized agents:
  - `RouterAgent`
  - `RetrieverAgent`
  - `PlannerAgent`
  - `ReviewerAgent`
- Baseline vs multi-agent eval with persisted metrics.
- Stage-level telemetry traces for observability.
- Production reliability via timeout-bound fallbacks.

## Setup

```bash
git clone https://github.com/pypi-ahmad/multi-agent-system-ollama.git
cd multi-agent-system-ollama
uv python pin 3.12.10
uv sync --dev
cp .env.example .env
```

## Pull required models

```bash
ollama pull phi3.5:3.8b
ollama pull functiongemma:270m
```

## Run end-to-end

```bash
uv run multi-agent-system run-all
```

## Notebook tutorial flow

```bash
uv run python scripts/execute_notebooks.py
```

Notebook order:

1. `notebooks/01_setup_and_model_check.ipynb`
2. `notebooks/02_agent_workflow_walkthrough.ipynb`
3. `notebooks/03_batch_demo_runs.ipynb`
4. `notebooks/04_evaluation.ipynb`
5. `notebooks/05_telemetry_and_report.ipynb`

## Real results (executed on June 12, 2026)

From `artifacts/evals/summary.json`:

- `n_tasks`: `6`
- `baseline_keyword_recall_mean`: `0.0`
- `multi_agent_keyword_recall_mean`: `0.8333`
- `keyword_recall_gain`: `+0.8333`
- `retrieval_hit_rate`: `1.0`
- `source_citation_rate`: `1.0`
- `planner_fallback_rate`: `1.0`
- `reviewer_fallback_rate`: `0.0`
- `avg_total_latency_ms`: `13027.57`

From `artifacts/telemetry/summary.json`:

- `n_spans`: `30`
- `n_unique_traces`: `6`
- highest mean-latency stage: `plan_answer` (`8016.02 ms`)

Generated outputs include:

- `artifacts/runs/demo_runs.json`
- `artifacts/evals/predictions.csv`
- `artifacts/evals/summary.json`
- `artifacts/telemetry/traces.jsonl`
- `artifacts/telemetry/summary.json`
- `artifacts/reports/multi_agent_system_report.md`
- `artifacts/run_summary.json`

## CLI examples

```bash
uv run multi-agent-system check-models
uv run multi-agent-system run-demo
uv run multi-agent-system evaluate
uv run multi-agent-system summarize-telemetry
```
