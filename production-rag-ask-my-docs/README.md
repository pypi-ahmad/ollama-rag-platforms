# Production RAG Monitoring and Observability (Local Ollama)

This project is a production-style RAG system focused on what usually gets skipped:
- tracing
- latency percentile tracking (`p50`, `p95`)
- cost-per-request
- quality metrics
- CI regression gating

It uses your local Ollama runtime and automatically picks from `ollama list`.

## Goal

Build a runnable RAG pipeline where every request is observable, measurable, and gateable before deployment.

## Architecture

```text
data/docs/*.md
  -> ingest + chunking
  -> hybrid index (BM25 + FAISS)

question
  -> hybrid retrieval
  -> local Ollama generation (/api/generate)
  -> citations extraction + fallback
  -> request metrics + traces persisted

eval dataset
  -> quality/latency/cost aggregates
  -> baseline comparison
  -> gate pass/fail for CI
```

## Project Structure

```text
src/ask_my_docs/
  cli.py
  settings.py
  models.py
  utils.py
  pipeline.py
  retrieval/hybrid.py
  llm/ollama.py
  observability/cost.py
  observability/metrics_store.py
  observability/tracing.py
  evaluation/metrics.py
  evaluation/runner.py
  evaluation/gating.py

configs/regression_thresholds.yaml
data/docs/
data/eval/eval_set.jsonl
.github/workflows/ci.yml
```

## Tutorial Walkthrough

## 1) Ingestion and Retrieval

File: `src/ask_my_docs/retrieval/hybrid.py`

- Loads docs from `data/docs`.
- Chunks text with overlap.
- Builds lexical retrieval with BM25.
- Builds semantic retrieval with deterministic hash embeddings + FAISS.
- Combines lexical and semantic scores via weighted normalization.

Why this design:
- deterministic/offline retrieval keeps runs reproducible and fast to iterate
- no external embedding dependency needed for demo and CI

Tradeoff:
- retrieval semantics are weaker than high-quality embedding models, but stable and portable.

## 2) Local LLM via Ollama

File: `src/ask_my_docs/llm/ollama.py`

- Calls `ollama list`, parses available models, and chooses:
1. `ASK_MY_DOCS_OLLAMA_MODEL` if set and present
2. first local model (size != `-`)
3. first listed model as fallback
- Sends grounded prompt to Ollama `/api/generate`.
- Returns answer text + token counts used for cost estimation.

Tradeoff:
- fully local and private
- latency depends on your local model size and hardware

## 3) Pipeline and Observability

File: `src/ask_my_docs/pipeline.py`

Each request records:
- `rag.request`, `rag.retrieve`, `rag.generate` spans
- total/retrieval/generation latency
- prompt/completion/total tokens
- estimated cost
- retrieval recall@k (during eval)

Citations:
- extracted from `[doc_id]` markers in LLM output
- fallback to top retrieved docs if model omits markers

## 4) Metrics Store

File: `src/ask_my_docs/observability/metrics_store.py`

Metrics are stored in DuckDB (`artifacts/observability/metrics.duckdb`) with summaries:
- `latency_p50_ms`
- `latency_p95_ms`
- `avg_cost_usd`
- `avg_retrieval_recall_at_k`
- `avg_answer_f1`
- `avg_exact_match`

## 5) Evaluation and Regression Gate

Files:
- `src/ask_my_docs/evaluation/runner.py`
- `src/ask_my_docs/evaluation/gating.py`
- `configs/regression_thresholds.yaml`

Evaluation computes:
- `answer_f1_mean`
- `exact_match_mean`
- `retrieval_recall_at_k_mean`
- `latency_p50_ms`, `latency_p95_ms`
- `avg_cost_usd`

Gate checks:
- absolute thresholds
- max allowed degradation vs baseline

## 6) CI Workflow

File: `.github/workflows/ci.yml`

CI runs:
1. lint (`ruff`)
2. type check (`mypy`)
3. tests (`pytest`)
4. `ask-my-docs ingest`
5. `ask-my-docs eval`
6. `ask-my-docs eval --gate`

## Quickstart

```bash
cd production-rag-ask-my-docs
uv sync --all-extras

uv run ask-my-docs ingest --docs-dir data/docs
uv run ask-my-docs ask "What is the SLA for invoice disputes?" --top-k 4

uv run ask-my-docs eval \
  --eval-path data/eval/eval_set.jsonl \
  --output artifacts/eval/current_metrics.json \
  --set-baseline \
  --baseline artifacts/baseline_metrics.json

uv run ask-my-docs eval \
  --eval-path data/eval/eval_set.jsonl \
  --output artifacts/eval/gate_metrics.json \
  --gate \
  --baseline artifacts/baseline_metrics.json \
  --thresholds configs/regression_thresholds.yaml

uv run ask-my-docs metrics-summary
```

Optional model pinning:

```bash
export ASK_MY_DOCS_OLLAMA_MODEL=gemma4:12b
```

Optional Ollama endpoint:

```bash
export ASK_MY_DOCS_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## Results Snapshot

See `artifacts/eval/run_output.txt`.

Key results from the current run:
- model: `gemma4:12b`
- ask latency: `39834.35 ms`
- eval `answer_f1_mean`: `0.7122`
- eval `retrieval_recall@k_mean`: `1.0000`
- eval `latency_p95_ms`: `59970.037`
- gate status: `PASS`

## Tests

```bash
uv run pytest -q
```

Current tests cover:
- cost estimator correctness
- metrics store percentile summaries
- regression gate pass/fail behavior
- Ollama list parser/model resolution

## Important Notes

- Cost is estimated using configured token rates in `settings.py`.
- Local Ollama inference is much slower than hosted APIs for larger models; thresholds are tuned accordingly.
- If you switch models/hardware, re-baseline before enforcing strict regression checks.

## Setup

```bash
git clone https://github.com/pypi-ahmad/production-rag-ask-my-docs.git
cd production-rag-ask-my-docs
```
