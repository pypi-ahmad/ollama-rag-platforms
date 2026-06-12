# Ask My Docs: practical Production RAG

This project includes:

- Reusable backend modules in `src/ask_my_docs` for production use.
- A hands-on notebook walkthrough in `notebooks/01_rag_walkthrough.ipynb` that uses those same modules.

The pipeline is exactly:

1. Ingest
2. Index
3. Retrieve (hybrid BM25 + vectors)
4. Rerank (cross-encoder)
5. Answer (citation-enforced)
6. Evaluate (CI-gated)

## Architecture

`data/docs` -> chunking -> `BM25 + FAISS` indexes -> hybrid retrieval -> cross-encoder reranking -> citation-enforced answer -> eval metrics + gate thresholds.

## Project Layout

```text
src/ask_my_docs/
  ingestion.py
  indexing.py
  retrieval/hybrid.py
  answering.py
  pipeline.py
  evaluation.py
  cli.py
  api.py

notebooks/
  01_rag_walkthrough.ipynb

data/docs/
data/eval/qa.yaml
configs/eval_thresholds.yaml
.github/workflows/ci.yml
```

## Quickstart

```bash
cd production-rag-ask-my-docs-learning
uv sync --all-extras --python 3.11
```

### CLI: End-to-End

```bash
uv run ask-my-docs build-index --docs-dir data/docs --index-dir artifacts/index
uv run ask-my-docs ask "What is the SLA for invoice disputes?" --index-dir artifacts/index --top-k 5
uv run ask-my-docs evaluate --dataset data/eval/qa.yaml --index-dir artifacts/index --thresholds configs/eval_thresholds.yaml --output artifacts/eval/report.json --gate
```

### Notebook Walkthrough

Open and run:

```bash
uv run jupyter notebook notebooks/01_rag_walkthrough.ipynb
```

Or execute headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_rag_walkthrough.ipynb --output 01_rag_walkthrough.executed.ipynb --output-dir notebooks
```

## Results

Run executed on June 12, 2026:

- `ruff`: `All checks passed!`
- `mypy`: `Success: no issues found in 14 source files`
- `pytest`: `6 passed in 21.00s`
- `build-index`: succeeded, wrote `artifacts/index/*`
- `evaluate --gate`: `PASS`
  - `hybrid_recall_at_5 = 1.0000`
  - `vector_recall_at_5 = 1.0000`
  - `hybrid_mrr_at_5 = 1.0000`
  - `citation_coverage = 1.0000`
  - `citation_validity = 1.0000`
  - `keyword_recall = 0.8333`

Artifacts:

- `artifacts/run_logs/uv_sync.txt`
- `artifacts/run_logs/pytest.txt`
- `artifacts/run_logs/build_index.txt`
- `artifacts/run_logs/ask.txt`
- `artifacts/run_logs/evaluate.txt`
- `artifacts/run_logs/inspect_retrieval.txt`
- `artifacts/run_logs/notebook_execute.txt`
- `artifacts/run_logs/ruff.txt`
- `artifacts/run_logs/mypy.txt`
- `artifacts/eval/report.json`
- `notebooks/01_rag_walkthrough.executed.ipynb`

## Why this design

- Hybrid retrieval improves robustness over lexical-only or vector-only retrieval.
- Cross-encoder reranking improves precision in top results.
- Citation enforcement reduces unsupported generation.
- CI-gated evaluation prevents silent regressions.

## Setup

```bash
git clone https://github.com/pypi-ahmad/production-rag-ask-my-docs-learning.git
cd production-rag-ask-my-docs-learning
```
