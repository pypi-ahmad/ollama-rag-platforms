# Project 4: Offline LLM App using Local Ollama

Notebook-based, end-to-end local RAG app using only local Ollama models.

- Chat model: `granite4.1:3b`
- Embedding model: `embeddinggemma:latest`
- Python: `3.12.10`
- Env/package manager: `uv`
- Interface: CLI + Streamlit + tutorial notebooks

## What this project demonstrates

- Local-only RAG (no cloud model dependency).
- Vector retrieval over local markdown knowledge files.
- Baseline vs RAG eval with persisted artifacts.
- Reliability controls: model checks, bounded timeouts, extractive fallback.

## Setup

```bash
git clone https://github.com/pypi-ahmad/offline-ollama-rag-app.git
cd offline-ollama-rag-app
uv python pin 3.12.10
uv sync --dev
cp .env.example .env
```

## Pull required models

```bash
ollama pull embeddinggemma:latest
ollama pull granite4.1:3b
```

## Run end-to-end pipeline

```bash
uv run offline-ollama-rag run-all
```

## Notebook tutorial flow

```bash
uv run python scripts/execute_notebooks.py
```

Notebook order:

1. `notebooks/01_setup_and_model_check.ipynb`
2. `notebooks/02_index_build_tutorial.ipynb`
3. `notebooks/03_question_answering_tutorial.ipynb`
4. `notebooks/04_evaluation_and_report.ipynb`

## Real results (executed on June 12, 2026)

From `artifacts/evals/summary.json`:

- `n_questions`: `6`
- `baseline_keyword_recall_mean`: `0.0`
- `rag_keyword_recall_mean`: `1.0`
- `keyword_recall_gain`: `+1.0`
- `baseline_nonempty_rate`: `1.0`
- `rag_nonempty_rate`: `1.0`

Generated artifacts include:

- `artifacts/index/embeddings.npy`
- `artifacts/index/chunks.json`
- `artifacts/evals/predictions.csv`
- `artifacts/evals/summary.json`
- `artifacts/reports/offline_ollama_rag_report.md`
- `artifacts/run_summary.json`

## Run app

```bash
uv run offline-ollama-rag serve-app --port 8503
```

## CLI examples

```bash
uv run offline-ollama-rag build-index
uv run offline-ollama-rag ask "What is the enterprise first-response SLA?"
uv run offline-ollama-rag ask "What is the enterprise first-response SLA?" --no-rag
uv run offline-ollama-rag evaluate
```
