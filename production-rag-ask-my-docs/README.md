# ask-my-docs-rag

Production-style Retrieval-Augmented Generation (RAG) on local Ollama with:
- hybrid retrieval (BM25 + FAISS)
- structured tracing
- DuckDB request metrics
- evaluation (quality, latency, cost)
- regression gating for CI

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Evaluation and Regression Gates](#evaluation-and-regression-gates)
- [Real Run Metrics](#real-run-metrics)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Observability Artifacts](#observability-artifacts)
- [Security Notes](#security-notes)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository demonstrates an end-to-end local RAG workflow with observability and quality controls that are typically needed before production rollout.

Core capabilities:
- index Markdown/TXT documents into a hybrid retriever
- answer grounded questions via Ollama
- persist request traces and metrics
- run dataset evaluations and compare against baseline metrics
- enforce quality/performance/cost thresholds in CI

## Architecture

```text
data/docs/*.md|*.txt
  -> chunking + indexing
  -> BM25 + FAISS hybrid retriever

question
  -> retrieve top-k chunks
  -> grounded prompt to Ollama (/api/generate)
  -> citation extraction
  -> persist traces + metrics

eval_set.jsonl
  -> run batch evaluation
  -> compute aggregate metrics
  -> optional regression gate vs baseline
```

## Prerequisites

- Linux/macOS
- Python `>=3.11` (recommended: `3.12.10`)
- [`uv`](https://docs.astral.sh/uv/)
- Ollama installed and running locally
- At least one local Ollama model pulled (`ollama list` should return entries)

## Installation

```bash
git clone https://github.com/pypi-ahmad/production-rag-ask-my-docs.git
cd production-rag-ask-my-docs

# Recommended local env
uv venv --python 3.12.10
source .venv/bin/activate

uv sync --all-extras
```

## Configuration

Settings are loaded from environment variables with prefix `ASK_MY_DOCS_`.

Common variables:

| Variable | Default | Description |
| --- | --- | --- |
| `ASK_MY_DOCS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `ASK_MY_DOCS_OLLAMA_MODEL` | unset | Optional explicit model name |
| `ASK_MY_DOCS_OLLAMA_TIMEOUT_SECONDS` | `120.0` | HTTP timeout for generation |
| `ASK_MY_DOCS_OLLAMA_LIST_TIMEOUT_SECONDS` | `10.0` | Timeout for `ollama list` model discovery |
| `ASK_MY_DOCS_DOCS_DIR` | `data/docs` | Input docs directory |
| `ASK_MY_DOCS_EVAL_PATH` | `data/eval/eval_set.jsonl` | Evaluation dataset path |
| `ASK_MY_DOCS_INDEX_DIR` | `artifacts/index` | Retrieval index artifacts |
| `ASK_MY_DOCS_TRACES_PATH` | `artifacts/observability/traces.jsonl` | JSONL trace output |
| `ASK_MY_DOCS_METRICS_DB_PATH` | `artifacts/observability/metrics.duckdb` | DuckDB metrics DB |
| `ASK_MY_DOCS_THRESHOLDS_PATH` | `configs/regression_thresholds.yaml` | Gate config path |
| `ASK_MY_DOCS_OBSERVABILITY_STORE_RAW_QUESTIONS` | `false` | Store raw prompt text in metrics DB (disable for privacy) |

Example:

```bash
export ASK_MY_DOCS_OLLAMA_MODEL=gemma4:12b
export ASK_MY_DOCS_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## CLI Usage

### 1. Build retrieval index

```bash
uv run ask-my-docs ingest --docs-dir data/docs
```

### 2. Ask one question

```bash
uv run ask-my-docs ask "What is the SLA for invoice disputes?" --top-k 4
```

Optional JSON payload output:

```bash
uv run ask-my-docs ask "What is the SLA for invoice disputes?" --output artifacts/sample_answer.json
```

### 3. Summarize observability metrics

```bash
uv run ask-my-docs metrics-summary
uv run ask-my-docs metrics-summary --limit 200
```

### 4. Run evaluation

```bash
uv run ask-my-docs eval \
  --eval-path data/eval/eval_set.jsonl \
  --output artifacts/eval/current_metrics.json
```

### 5. Set baseline metrics

```bash
uv run ask-my-docs eval \
  --eval-path data/eval/eval_set.jsonl \
  --output artifacts/eval/current_metrics.json \
  --set-baseline \
  --baseline artifacts/baseline_metrics.json
```

### 6. Apply regression gate

```bash
uv run ask-my-docs eval \
  --eval-path data/eval/eval_set.jsonl \
  --output artifacts/eval/gate_metrics.json \
  --baseline artifacts/baseline_metrics.json \
  --thresholds configs/regression_thresholds.yaml \
  --gate
```

## Evaluation and Regression Gates

Aggregate evaluation metrics include:
- `answer_f1_mean`
- `exact_match_mean`
- `retrieval_recall_at_k_mean`
- `latency_p50_ms`
- `latency_p95_ms`
- `avg_cost_usd`
- `avg_tokens`

Gate checks:
- absolute thresholds (minimum quality, max latency/cost)
- regression thresholds (allowed degradation vs baseline)

Gate config lives at [configs/regression_thresholds.yaml](configs/regression_thresholds.yaml).

## Real Run Metrics

Latest validated end-to-end run:
- date: `2026-06-13`
- model: `gemma4:12b` (local Ollama)
- workflow: `ingest -> ask -> eval --set-baseline -> eval --gate`
- gate result: `PASS`

Single `ask` run summary:
- latency_ms: `31128.992`
- retrieval_latency_ms: `0.348`
- estimated_cost_usd: `0.00026640`
- citations: `billing_disputes`
- artifact: `artifacts/eval/e2e_live_ask_gemma4_12b.json`

Evaluation aggregate (`5` examples):

| Metric | Value |
| --- | --- |
| `answer_f1_mean` | `0.7157244987` |
| `exact_match_mean` | `0.0` |
| `retrieval_recall_at_k_mean` | `1.0` |
| `latency_p50_ms` | `22585.387` |
| `latency_p95_ms` | `30052.5802` |
| `avg_cost_usd` | `0.00025797` |
| `avg_tokens` | `713.6` |

Artifacts:
- `artifacts/eval/e2e_live_current_metrics_gemma4_12b.json`
- `artifacts/eval/e2e_live_baseline_metrics_gemma4_12b.json`
- `artifacts/eval/e2e_live_gate_metrics_gemma4_12b.json`

## Project Structure

```text
src/ask_my_docs/
  cli.py
  pipeline.py
  settings.py
  models.py
  utils.py
  retrieval/hybrid.py
  llm/ollama.py
  observability/
    cost.py
    metrics_store.py
    tracing.py
  evaluation/
    metrics.py
    runner.py
    gating.py

configs/regression_thresholds.yaml
data/docs/
data/eval/eval_set.jsonl
tests/
.github/workflows/ci.yml
```

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run mypy src
```

## Testing

```bash
uv run pytest -q
```

Current automated tests cover:
- cost estimator behavior
- metric summary percentiles
- gate pass/fail logic
- Ollama model list parsing/selection

## Observability Artifacts

Default output paths:
- traces: `artifacts/observability/traces.jsonl`
- request metrics DB: `artifacts/observability/metrics.duckdb`
- retriever index: `artifacts/index/`
- evaluation reports: `artifacts/eval/*.json`

## Security Notes

- Ollama calls are local HTTP by default (`127.0.0.1`).
- Do not commit secrets in `.env` files.
- Cost values are estimates and should not be used as billing truth.
- If exposing this service beyond localhost, add authentication and transport security.

## Limitations

- Semantic retrieval uses deterministic hash embeddings for reproducibility, not SOTA semantic quality.
- Local model latency depends heavily on hardware/model size.
- Current test suite focuses on core utilities; end-to-end CLI/pipeline integration coverage is limited.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Run lint/type/tests locally.
4. Open a pull request with a clear change summary and test evidence.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
