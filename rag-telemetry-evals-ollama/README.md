# Project 5: RAG System with Telemetry and Evals

Notebook-based local RAG project focused on observability and measurable evaluation.

- Chat model: `phi3.5:3.8b`
- Embedding model: `embeddinggemma:latest`
- Python: `3.12.10`
- Env/package manager: `uv`

## What this project demonstrates

- End-to-end local RAG pipeline with artifacted eval outputs.
- Telemetry tracing (`jsonl`) for retrieval/generation spans.
- Baseline vs RAG quality comparison using keyword and semantic metrics.
- Structured evaluation/report generation for portfolio-ready evidence.

## Setup

```bash
git clone https://github.com/pypi-ahmad/rag-telemetry-evals-ollama.git
cd rag-telemetry-evals-ollama
uv python pin 3.12.10
uv sync --dev
cp .env.example .env
```

## Pull required models

```bash
ollama pull embeddinggemma:latest
ollama pull phi3.5:3.8b
```

## Run end-to-end pipeline

```bash
uv run rag-telemetry-evals run-all
```

## Notebook tutorial flow

```bash
uv run python scripts/execute_notebooks.py
```

Notebook order:

1. `notebooks/01_setup_and_model_check.ipynb`
2. `notebooks/02_index_build_tutorial.ipynb`
3. `notebooks/03_question_answering_and_traces.ipynb`
4. `notebooks/04_evaluation_tutorial.ipynb`
5. `notebooks/05_telemetry_analysis.ipynb`

## Real results (executed on June 12, 2026)

From `artifacts/evals/summary.json`:

- `n_questions`: `8`
- `retrieval_hit_rate`: `1.0`
- `rag_mentions_expected_source_rate`: `1.0`
- `baseline_keyword_recall_mean`: `0.0625`
- `rag_keyword_recall_mean`: `1.0`
- `keyword_recall_gain`: `+0.9375`
- `baseline_semantic_similarity_mean`: `0.0674`
- `rag_semantic_similarity_mean`: `0.3222`
- `semantic_similarity_gain`: `+0.2548`
- `baseline_latency_ms_mean`: `2789.48`
- `rag_latency_ms_mean`: `1953.58`

From `artifacts/telemetry/summary.json`:

- `n_spans`: `40`
- `n_unique_traces`: `16`
- highest mean-latency span: `answer_rag` (`23659.07 ms`)

Generated outputs include:

- `artifacts/evals/predictions.csv`
- `artifacts/evals/summary.json`
- `artifacts/reports/rag_telemetry_eval_report.md`
- `artifacts/telemetry/traces.jsonl`
- `artifacts/telemetry/summary.json`
- `artifacts/run_summary.json`

## CLI examples

```bash
uv run rag-telemetry-evals build-index
uv run rag-telemetry-evals ask "What is the enterprise first-response SLA?" --rag
uv run rag-telemetry-evals ask "What is the enterprise first-response SLA?" --baseline
uv run rag-telemetry-evals evaluate
uv run rag-telemetry-evals summarize-telemetry
```
