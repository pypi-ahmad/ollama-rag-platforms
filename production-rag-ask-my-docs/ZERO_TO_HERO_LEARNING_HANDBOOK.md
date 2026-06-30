# 📘 MASTER LEARNING HANDBOOK: ask-my-docs-rag

Authoring mode: static analysis only. No project code was executed, compiled, or tested for this handbook.  
Scope: full local tree excluding env-generated folders (`.venv`, `.git` internals).  
Evidence standard: high-density source grounding via exact paths and line references.

---

## 🌐 Module 1: Theoretical Foundations & Architecture

### 1.1 CS Theory & Definitions (then how this repo implements each)

1. **Retrieval-Augmented Generation (RAG)**
- Definition: a generation architecture that first retrieves relevant context from a corpus, then conditions an LLM prompt on that context.
- In this repo:
  - Retrieval happens before generation in `RAGPipeline.answer` (`rag.retrieve` span before `rag.generate`).  
    Source: `src/ask_my_docs/pipeline.py:53-64`
  - Retrieved chunks are passed to Ollama prompt construction as `[doc_id] chunk_text` blocks.  
    Source: `src/ask_my_docs/llm/ollama.py:162-174`

2. **Hybrid Retrieval**
- Definition: combine lexical retrieval (keyword/statistical) and semantic retrieval (embedding similarity).
- In this repo:
  - Lexical engine: `BM25Okapi`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:13,198-200`
  - Semantic engine: FAISS `IndexFlatIP` with dense vectors.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:201`
  - Final score = weighted sum of normalized lexical + normalized semantic scores.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:114-121,350-353`
  - Default weights are `lexical_weight=0.45`, `semantic_weight=0.55`.  
    Source: `src/ask_my_docs/settings.py:26-27`

3. **BM25 (Best Matching 25)**
- Definition: probabilistic ranking over tokenized text emphasizing term frequency and inverse document frequency.
- In this repo:
  - Built from chunk token lists with `BM25Okapi(tokenized_chunks)`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:198-200`
  - Query lexical scores from `self._bm25.get_scores(tokenize(query))`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:340`

4. **Vector Similarity Search (FAISS IP)**
- Definition: nearest-neighbor search in vector space using similarity metric (inner product here).
- In this repo:
  - Query embedding searched with `self._faiss_index.search(query_embedding[None, :], search_k)`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:344-346`
  - `search_k` heuristic is `min(len(chunks), max(k*4, k))`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:344`

5. **Deterministic Hash Embeddings**
- Definition: fixed embedding function from tokens to vector dimensions via hash, useful for reproducibility/offline behavior.
- In this repo:
  - `HashingEmbedder.embed` uses `blake2b`, chooses index/sign, accumulates vector, then L2 normalizes.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:67-87`
  - Embedding dimension is configured by `RetrievalConfig.embedding_dim` (default `256`).  
    Source: `src/ask_my_docs/settings.py:28`

6. **Chunking**
- Definition: partition source docs into overlapping windows to improve retrieval granularity.
- In this repo:
  - Default chunk params: `chunk_size_tokens=120`, `chunk_overlap_tokens=20`.  
    Source: `src/ask_my_docs/settings.py:23-24`
  - Chunk IDs follow `"{doc_id}::chunk-{n}"`; metadata stores `start_word`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:99-108`

7. **Object-Oriented + Dataclass Design**
- Definition:
  - OOP organizes behavior/state in classes.
  - Dataclasses provide lightweight typed records.
- In this repo:
  - OOP service classes: `HybridRetriever`, `OllamaGenerator`, `MetricsStore`, `RAGPipeline`.  
    Source: `src/ask_my_docs/retrieval/hybrid.py:124`, `src/ask_my_docs/llm/ollama.py:98`, `src/ask_my_docs/observability/metrics_store.py:35`, `src/ask_my_docs/pipeline.py:23`
  - Dataclass records: `Document`, `Chunk`, `RetrievedChunk`, `RAGAnswer`, `RequestMetricRecord`, `EvalExample`, `EvalReport`.  
    Source: `src/ask_my_docs/models.py:10,25,35,46`, `src/ask_my_docs/observability/metrics_store.py:15`, `src/ask_my_docs/evaluation/runner.py:16,25`

8. **Observability (Tracing + Metrics)**
- Definition: collect operational telemetry for latency, costs, and quality signals.
- In this repo:
  - Tracing: OpenTelemetry exporter writes JSONL spans (`name`, `trace_id`, `duration_ms`, `attributes`).  
    Source: `src/ask_my_docs/observability/tracing.py:18-44`
  - Request metrics: DuckDB table `rag_request_metrics` captures latency/tokens/cost/quality fields.  
    Source: `src/ask_my_docs/observability/metrics_store.py:69-85`

9. **Regression Gating**
- Definition: fail deployment/CI when quality drops or latency/cost worsens beyond policy.
- In this repo:
  - Gate config model in YAML: absolute + regression blocks.  
    Source: `configs/regression_thresholds.yaml:1-11`, `src/ask_my_docs/evaluation/gating.py:12-31`
  - Gate eval compares current metrics to absolute bounds and optional baseline deltas.  
    Source: `src/ask_my_docs/evaluation/gating.py:52-124`

10. **Privacy-by-default Question Storage**
- Definition: avoid storing raw user input unless explicitly enabled.
- In this repo:
  - `observability_store_raw_questions=False` default.  
    Source: `src/ask_my_docs/settings.py:55`
  - Question is hashed/redacted as `<redacted:sha256:{digest}:len={n}>` unless enabled.  
    Source: `src/ask_my_docs/pipeline.py:150-154`
  - Verified by tests.  
    Source: `tests/test_pipeline.py:55,68,82`

### 1.2 System Topology (ASCII)

```text
User CLI
  |
  | ask-my-docs ingest / ask / metrics-summary / eval
  v
src/ask_my_docs/cli.py
  |
  +--> settings: AppSettings (env prefix ASK_MY_DOCS_)
  |     (src/ask_my_docs/settings.py)
  |
  +--> ingest path
  |      load_documents -> HybridRetriever.from_documents -> save index artifacts
  |      (src/ask_my_docs/retrieval/hybrid.py)
  |
  +--> ask path
  |      _build_pipeline:
  |        configure_tracing -> JsonLineSpanExporter -> traces.jsonl
  |        MetricsStore(DuckDB) -> rag_request_metrics table
  |        HybridRetriever.load(index)
  |        OllamaGenerator(/api/generate, temp=0.1)
  |        CostCalculator(TokenPricing)
  |      RAGPipeline.answer:
  |        retrieve -> generate -> citations -> cost -> metrics row
  |
  +--> eval path
         load_eval_examples(jsonl)
         run_evaluation (F1, EM, recall@k, p50/p95, cost, tokens)
         optional gate (absolute + regression vs baseline)
         optional baseline update
```

Primary evidence: `src/ask_my_docs/cli.py:56-90,93-273`, `src/ask_my_docs/pipeline.py:42-133`, `src/ask_my_docs/retrieval/hybrid.py:214-383`, `src/ask_my_docs/llm/ollama.py:150-222`, `src/ask_my_docs/observability/metrics_store.py:69-313`, `src/ask_my_docs/evaluation/runner.py:50-137`, `src/ask_my_docs/evaluation/gating.py:45-124`.

### 1.3 Technology Stack with Exact Roles and Real Parameters

| Component | Exact dependency/config | Role in project | Where used |
|---|---|---|---|
| Python runtime | `requires-python = ">=3.11"` | Language/runtime baseline | `pyproject.toml:6` |
| Typer | `typer>=0.13.0` | CLI command surface (`ingest`, `ask`, `metrics-summary`, `eval`) | `pyproject.toml:20`, `src/ask_my_docs/cli.py:28,93,118,157,188` |
| Pydantic + pydantic-settings | `pydantic>=2.9.0`, `pydantic-settings>=2.5.0` | Typed config + env loading | `pyproject.toml:16-17`, `src/ask_my_docs/settings.py:8-38` |
| Env prefix | `ASK_MY_DOCS_` | Namespaces runtime env vars | `src/ask_my_docs/settings.py:35` |
| Retriever defaults | `chunk_size_tokens=120`, `chunk_overlap_tokens=20`, `top_k=4`, `lexical_weight=0.45`, `semantic_weight=0.55`, `embedding_dim=256` | Retrieval behavior + index compatibility | `src/ask_my_docs/settings.py:23-28` |
| rank-bm25 | `rank-bm25>=0.2.2` | Lexical scoring | `pyproject.toml:19`, `src/ask_my_docs/retrieval/hybrid.py:13,340` |
| FAISS CPU | `faiss-cpu>=1.9.0` | Semantic nearest-neighbor index/search | `pyproject.toml:9`, `src/ask_my_docs/retrieval/hybrid.py:10,201,345` |
| NumPy | `numpy>=1.26.0` | Embeddings matrix + score arrays + normalization | `pyproject.toml:11`, `src/ask_my_docs/retrieval/hybrid.py:11,73-87,114-121,344-353` |
| Ollama HTTP API | `POST {base_url}/api/generate` | Grounded generation backend | `src/ask_my_docs/llm/ollama.py:180-187` |
| Generation parameter | `options: {"temperature": 0.1}` | Low-variance answer generation | `src/ask_my_docs/llm/ollama.py:177` |
| Model discovery | `subprocess.run(["ollama","list"])` + local-first selection | Selects configured or first local model | `src/ask_my_docs/llm/ollama.py:125-148`, `63-96` |
| OpenTelemetry | `opentelemetry-api>=1.28.0`, `opentelemetry-sdk>=1.28.0` | Structured trace spans | `pyproject.toml:12-13`, `src/ask_my_docs/observability/tracing.py:10-13,59-71` |
| DuckDB | `duckdb>=1.1.0` | Request metrics warehouse + aggregate summaries | `pyproject.toml:8`, `src/ask_my_docs/observability/metrics_store.py:35-313` |
| Polars | `polars>=1.9.0` | Eval aggregation frame/quantiles | `pyproject.toml:15`, `src/ask_my_docs/evaluation/runner.py:8,108-122` |
| YAML | `pyyaml>=6.0.2` | Gate config parsing | `pyproject.toml:18`, `src/ask_my_docs/evaluation/gating.py:8,45-49` |
| orjson | `orjson>=3.10.0` | Fast JSON read/write | `pyproject.toml:14`, `src/ask_my_docs/utils.py:7,27,38` |
| loguru | `loguru>=0.7.2` | CLI logging and clean error output | `pyproject.toml:10`, `src/ask_my_docs/cli.py:11,35-41` |
| CI workflow | GitHub Actions `quality-and-regression-gate` | Lint + typing + tests + ingest + eval + gate | `.github/workflows/ci.yml:1-49` |

---

## 📂 Module 2: Strict Repository Tour & Mapping

### 2.1 Directory Mapping (complete scope)

| File/Directory Path | Primary Responsibility | Key Classes/Functions Exported | Actual Configurations/Variables Defined |
|---|---|---|---|
| `.` | Repository root for RAG platform | - | - |
| `./.agents` | Local agent metadata folder (empty in current scope) | - | - |
| `./.codex` | Local codex metadata folder (empty in current scope) | - | - |
| `./.github` | CI/CD and automation metadata | - | - |
| `./.github/workflows` | GitHub Action workflow definitions | - | `ci` workflow |
| `./artifacts` | Stored outputs (metrics baselines/tutorial artifacts) | - | JSON baselines, notebook artifacts |
| `./artifacts/notebook_tutorial` | Tutorial-scoped reproducible outputs | - | `baseline_metrics.json`, `current_metrics.json`, etc. |
| `./artifacts/notebook_tutorial/index` | Persisted retrieval index artifacts | - | `chunk_count`, `embedding_dim`, weights manifest |
| `./artifacts/notebook_tutorial/observability` | Tutorial observability outputs | - | `metrics.duckdb`, `traces.jsonl` |
| `./configs` | Runtime/CI policy configs | - | regression threshold values |
| `./data` | Source docs + evaluation set | - | policy text corpus + eval JSONL |
| `./data/docs` | Ground-truth documentation corpus for retrieval | - | 5 policy documents |
| `./data/eval` | Offline evaluation dataset | - | 5 JSONL evaluation examples |
| `./notebooks` | Stepwise educational notebook | - | `tutorial_observability_ollama.ipynb` |
| `./src` | Python package source root | - | - |
| `./src/ask_my_docs` | Main package root | package exports/version | `__version__ = "0.1.0"` |
| `./src/ask_my_docs/evaluation` | Quality metrics + evaluation runner + gating | metric/gating APIs | aggregate metric definitions |
| `./src/ask_my_docs/llm` | LLM backend integration layer | Ollama backend exports | temperature/model resolution behavior |
| `./src/ask_my_docs/observability` | Tracing/cost/metrics persistence | observability exports | schema/indexes/summarization |
| `./src/ask_my_docs/retrieval` | Document loading/chunking/hybrid search | retriever exports | chunking, embedding, weighting |
| `./tests` | Unit tests for core reliability behavior | test modules | expected behavior constraints |

### 2.2 File Mapping (complete scope)

| File/Directory Path | Primary Responsibility | Key Classes/Functions Exported | Actual Configurations/Variables Defined |
|---|---|---|---|
| `./.coverage` | Coverage artifact DB (SQLite format) | - | coverage tables (`arc`, `file`, `line_bits`, etc.) |
| `./.github/workflows/ci.yml` | CI pipeline definition | - | job `quality-and-regression-gate`, steps for lint/type/test/ingest/eval/gate |
| `./.gitignore` | Ignore patterns for generated artifacts | - | ignores `.venv`, caches, `artifacts/index`, `artifacts/eval`, `artifacts/observability` |
| `./LICENSE` | Project license | - | MIT license text |
| `./README.md` | User-facing project guide | - | env var table, CLI usage, metrics and limitations |
| `./artifacts/baseline_metrics.json` | Baseline eval report used by gate | - | `aggregate` and `per_example` metric payloads |
| `./artifacts/notebook_tutorial/baseline_metrics.json` | Notebook-local baseline metrics | - | aggregate/per-example evaluation report |
| `./artifacts/notebook_tutorial/current_metrics.json` | Notebook-local current run metrics | - | aggregate/per-example evaluation report |
| `./artifacts/notebook_tutorial/index/chunks.jsonl` | Chunk artifact for retriever | - | keys: `chunk_id`, `doc_id`, `text`, `metadata.start_word`, `metadata.path` |
| `./artifacts/notebook_tutorial/index/embeddings.npy` | Dense embeddings matrix | - | shape `(5, 256)` (from file metadata) |
| `./artifacts/notebook_tutorial/index/manifest.json` | Index compatibility manifest | - | `chunk_count=5`, `embedding_dim=256`, `lexical_weight=0.45`, `semantic_weight=0.55` |
| `./artifacts/notebook_tutorial/index/semantic.index` | FAISS index binary | - | semantic vector index artifact |
| `./artifacts/notebook_tutorial/observability/metrics.duckdb` | Observability DB artifact | - | DuckDB metrics warehouse file |
| `./artifacts/notebook_tutorial/observability/traces.jsonl` | Span trace log artifact | - | JSONL spans with `name`, `trace_id`, `duration_ms`, `attributes` |
| `./artifacts/notebook_tutorial/sample_answer.json` | Example `ask` payload output | - | answer/citations/tokens/cost/latency/request identifiers |
| `./configs/regression_thresholds.yaml` | Gate policy config | - | absolute + regression threshold values |
| `./data/docs/access_control.md` | Source policy document | - | access review cadence, MFA requirement, retention duration |
| `./data/docs/billing_disputes.md` | Source policy document | - | SLA and escalation policy values |
| `./data/docs/data_retention.md` | Source policy document | - | retention, deletion, backup rotation values |
| `./data/docs/incident_sla.md` | Source policy document | - | sev1/sev2 acknowledgement and restoration targets |
| `./data/docs/onboarding_handover.md` | Source policy document | - | onboarding handover and checkpoint cadence |
| `./data/eval/eval_set.jsonl` | Evaluation dataset | - | keys per row: `question`, `reference_answer`, `expected_doc_ids` |
| `./notebooks/tutorial_observability_ollama.ipynb` | Educational end-to-end tutorial | notebook cells | 10-step walkthrough with isolated artifact paths |
| `./pyproject.toml` | Python packaging/tooling config | script entrypoint `ask-my-docs` | dependencies, dev dependencies, ruff/mypy/pytest config |
| `./src/ask_my_docs/__init__.py` | Package metadata | `__version__` | `__version__ = "0.1.0"` |
| `./src/ask_my_docs/cli.py` | CLI orchestration layer | `app`, command funcs: `ingest`, `ask`, `metrics_summary`, `eval` | `DEFAULT_EVAL_OUTPUT`, `DEFAULT_BASELINE_PATH` |
| `./src/ask_my_docs/evaluation/__init__.py` | Evaluation package marker | `__all__` | empty export list |
| `./src/ask_my_docs/evaluation/gating.py` | Regression gate logic | `AbsoluteThresholds`, `RegressionThresholds`, `GateConfig`, `GateResult`, `load_gate_config`, `evaluate_gate` | all gate fields and pass/fail logic |
| `./src/ask_my_docs/evaluation/metrics.py` | Eval metric functions | `exact_match_score`, `token_f1_score`, `retrieval_recall_at_k` | token-normalized EM/F1 definitions |
| `./src/ask_my_docs/evaluation/runner.py` | Eval runner and report writer | `EvalExample`, `EvalReport`, `load_eval_examples`, `run_evaluation`, `save_eval_report` | aggregate fields (`answer_f1_mean`, `latency_p95_ms`, `avg_cost_usd`, etc.) |
| `./src/ask_my_docs/llm/__init__.py` | LLM package exports | `OllamaGeneration`, `OllamaGenerator`, `OllamaModelInfo`, parser/resolver funcs | `__all__` export contract |
| `./src/ask_my_docs/llm/ollama.py` | Ollama integration backend | `OllamaModelInfo`, `OllamaGeneration`, `parse_ollama_list`, `resolve_ollama_model`, `OllamaGenerator` | prompt template, `temperature=0.1`, `/api/generate` endpoint |
| `./src/ask_my_docs/models.py` | Shared datamodel layer | `Document`, `Chunk`, `RetrievedChunk`, `RAGAnswer` | typed fields for IDs, tokens, costs, latencies, timestamps |
| `./src/ask_my_docs/observability/__init__.py` | Observability exports | `CostCalculator`, `TokenPricing`, `MetricsStore`, `RequestMetricRecord`, `configure_tracing` | `__all__` export contract |
| `./src/ask_my_docs/observability/cost.py` | Cost estimation logic | `TokenPricing`, `CostCalculator` | prompt/completion cost computation and rounding |
| `./src/ask_my_docs/observability/metrics_store.py` | DuckDB metrics persistence/summarization | `RequestMetricRecord`, `MetricsStore` methods (`record`, `update_quality`, `summarize`, `close`) | `rag_request_metrics` schema, upsert behavior, p50/p95 SQL |
| `./src/ask_my_docs/observability/tracing.py` | Trace exporter setup | `JsonLineSpanExporter`, `configure_tracing` | `_CONFIGURED` guard, JSONL span payload keys |
| `./src/ask_my_docs/pipeline.py` | Core request pipeline | `RAGPipeline` methods: `answer`, `update_quality_metrics`, `to_payload` | `_CITATION_RE`, redaction format, span attributes |
| `./src/ask_my_docs/retrieval/__init__.py` | Retrieval package exports | `HybridRetriever`, `load_documents` | `__all__` export contract |
| `./src/ask_my_docs/retrieval/hybrid.py` | Hybrid retriever implementation | `_sanitize_doc_id`, `load_documents`, `HashingEmbedder`, `_chunk_document`, `_normalize`, `HybridRetriever` | manifest fields, weighted scoring, index compatibility checks |
| `./src/ask_my_docs/settings.py` | Runtime settings contract | `PricingConfig`, `RetrievalConfig`, `AppSettings`, `get_settings` | all `ASK_MY_DOCS_*` defaults and retrieval/pricing defaults |
| `./src/ask_my_docs/utils.py` | Shared utility helpers | `tokenize`, `ensure_dir`, `read_json`, `write_json` | `_WORD_RE`, stable JSON serialization options |
| `./tests/test_cli.py` | CLI behavior tests | test functions | validates clean errors, decoupled metrics-summary behavior |
| `./tests/test_costing.py` | Cost calculator tests | test functions | validates prompt+completion rate arithmetic |
| `./tests/test_gating.py` | Gate logic tests | test functions | validates pass/fail under explicit thresholds |
| `./tests/test_metrics_store.py` | Metrics store tests | test functions | validates p50/p95 aggregation and upsert semantics |
| `./tests/test_ollama.py` | Ollama utility tests | test functions | validates `ollama list` parsing/model selection/timeout handling |
| `./tests/test_pipeline.py` | Pipeline behavior tests | test functions + `_StubGenerator` | validates question redaction/raw mode/citation parser |
| `./tests/test_retrieval.py` | Retrieval artifact/tests | test functions | validates unique doc IDs and embedding-dimension mismatch errors |
| `./uv.lock` | Fully resolved dependency lock | - | resolved package versions and metadata for reproducible installs |

---

## 🔍 Module 3: Line-by-Line Code & Output Breakdown

This module follows the real runtime flows in the codebase and ties them to exact variable names, schemas, and artifacts.

### 3.1 Flow A: `ingest` (source docs -> hybrid index)

#### Step-by-step execution

1. CLI command enters `ingest(...)`.  
Source: `src/ask_my_docs/cli.py:93`

2. `docs_dir` resolves to CLI option or `settings.docs_dir`.  
Source: `src/ask_my_docs/cli.py:100-101`, `src/ask_my_docs/settings.py:48`

3. `load_documents(source_dir)` recursively reads `.md`/`.txt` files.  
Source: `src/ask_my_docs/cli.py:103`, `src/ask_my_docs/retrieval/hybrid.py:27-65`

4. Document IDs are derived from relative path stem and sanitized by `_sanitize_doc_id`; collisions use `blake2b` suffix.  
Source: `src/ask_my_docs/retrieval/hybrid.py:22-25,38-52`

5. `HybridRetriever.from_documents(...)` chunks docs via `_chunk_document`.  
Source: `src/ask_my_docs/cli.py:104`, `src/ask_my_docs/retrieval/hybrid.py:175-195,89-112`

6. Embeddings are built by `HashingEmbedder(dim=config.embedding_dim)`, and FAISS index is `IndexFlatIP`.  
Source: `src/ask_my_docs/retrieval/hybrid.py:194-202`

7. Retriever saves artifacts (`chunks.jsonl`, `embeddings.npy`, `semantic.index`, `manifest.json`).  
Source: `src/ask_my_docs/retrieval/hybrid.py:308-330`

8. Manifest includes:
- `chunk_count`
- `embedding_dim`
- `lexical_weight`
- `semantic_weight`  
Source: `src/ask_my_docs/retrieval/hybrid.py:322-327`

#### Real input schema
- Source docs are plain text markdown/txt files in `data/docs`.  
Source: `data/docs/*.md`

Example facts from corpus:
- Access reviews every 90 days; MFA required for admin/support-console accounts.  
  Source: `data/docs/access_control.md:3`
- Billing disputes acknowledged in 1 business day; enterprise resolved within 24 hours.  
  Source: `data/docs/billing_disputes.md:3`

#### Real output schema
`chunks.jsonl` row keys:
- `chunk_id`
- `doc_id`
- `text`
- `metadata` with `start_word`, `path`  
Source: `artifacts/notebook_tutorial/index/chunks.jsonl:1-5`

`manifest.json`:
- `"chunk_count": 5`
- `"embedding_dim": 256`
- `"lexical_weight": 0.45`
- `"semantic_weight": 0.55`  
Source: `artifacts/notebook_tutorial/index/manifest.json:2-5`

### 3.2 Flow B: `ask` (question -> retrieved context -> Ollama -> observed answer)

#### Step-by-step execution

1. CLI `ask(question, top_k, output)` constructs pipeline via `_build_pipeline()`.  
Source: `src/ask_my_docs/cli.py:118-132`

2. `_build_pipeline()` wires:
- tracing (`configure_tracing`)
- metrics store (`MetricsStore`)
- retriever (`HybridRetriever.load`)
- cost calculator (`CostCalculator(TokenPricing(...))`)
- generator (`OllamaGenerator(...)`)  
Source: `src/ask_my_docs/cli.py:56-90`

3. `pipeline.answer(question, top_k)` starts request span and generates `request_id`.  
Source: `src/ask_my_docs/pipeline.py:42-55`

4. Retrieval phase:
- `retrieved = self._retriever.search(query=question, top_k=top_k)`  
Source: `src/ask_my_docs/pipeline.py:58-60`

5. Generation phase:
- `generation = self._generator.generate(question=question, retrieved=retrieved)`  
Source: `src/ask_my_docs/pipeline.py:64-67`

6. Cost and citation extraction:
- tokens = `prompt_tokens + completion_tokens`
- `estimated_cost = cost_calculator.estimate(...)`
- citations from `_extract_citations(...)` using regex:
  - `\[(?:doc_id\s*[:=]\s*)?([A-Za-z0-9_.:-]+)]`  
Source: `src/ask_my_docs/pipeline.py:69-77`, `20`

7. Question persistence:
- If `store_raw_questions=False`, stored question is `<redacted:sha256:{digest}:len={len(question)}>`  
Source: `src/ask_my_docs/pipeline.py:150-154`

8. Metrics row is persisted as `RequestMetricRecord(...)`.  
Source: `src/ask_my_docs/pipeline.py:113-132`

9. CLI prints summary and optionally writes full payload JSON via `write_json`.  
Source: `src/ask_my_docs/cli.py:131-145`, `src/ask_my_docs/utils.py:34-38`

#### Retrieval scoring logic details

Inside `HybridRetriever.search`:
- `k = top_k or config.top_k`, bounded to chunk count.  
  Source: `src/ask_my_docs/retrieval/hybrid.py:335-338`
- Lexical scores: `self._bm25.get_scores(tokenize(query))`.  
  Source: `src/ask_my_docs/retrieval/hybrid.py:340`
- Semantic search over FAISS with `search_k = min(len(chunks), max(k*4, k))`.  
  Source: `src/ask_my_docs/retrieval/hybrid.py:344-346`
- Min-max normalization via `_normalize`.  
  Source: `src/ask_my_docs/retrieval/hybrid.py:114-121,350-351`
- Combined score: weighted lexical + semantic.  
  Source: `src/ask_my_docs/retrieval/hybrid.py:352-353`
- Top candidates via `np.argpartition` then deterministic sorted tie-break by combined, lexical, semantic.  
  Source: `src/ask_my_docs/retrieval/hybrid.py:360-369`

#### Real output payload structure (actual keys)

`sample_answer.json` keys:
- `answer`
- `citations`
- `completion_tokens`
- `estimated_cost_usd`
- `generation_latency_ms`
- `latency_ms`
- `model_name`
- `prompt_tokens`
- `question`
- `request_id`
- `retrieval_latency_ms`
- `retrieval_recall_at_k`
- `retrieved_doc_ids`
- `timestamp_utc`
- `total_tokens`
- `trace_id`  
Source: `artifacts/notebook_tutorial/sample_answer.json:2-24`

Real values example:
- `model_name = "gemma4:12b"`
- `estimated_cost_usd = 0.0002682`
- `latency_ms = 29088.981`
- `retrieval_latency_ms = 0.436`
- `retrieval_recall_at_k = null` (for direct ask without expected docs)  
Source: `artifacts/notebook_tutorial/sample_answer.json:7-15`

### 3.3 Flow C: `eval` + regression gate

#### Step-by-step execution

1. CLI `eval(...)` loads dataset path and builds pipeline.  
Source: `src/ask_my_docs/cli.py:188-222`

2. Examples are loaded from JSONL into `EvalExample(question, reference_answer, expected_doc_ids)`.  
Source: `src/ask_my_docs/evaluation/runner.py:16-21,50-67`

3. For each example:
- call `pipeline.answer(... expected_doc_ids=...)`
- compute `answer_f1` and `exact_match`
- backfill DB row via `pipeline.update_quality_metrics(...)`  
Source: `src/ask_my_docs/evaluation/runner.py:74-86`, `src/ask_my_docs/pipeline.py:135-141`

4. Aggregate metrics computed with Polars:
- `answer_f1_mean`
- `exact_match_mean`
- `retrieval_recall_at_k_mean`
- `latency_p50_ms`
- `latency_p95_ms`
- `avg_cost_usd`
- `avg_tokens`  
Source: `src/ask_my_docs/evaluation/runner.py:108-123`

5. Report saved as JSON (`aggregate` + `per_example`).  
Source: `src/ask_my_docs/evaluation/runner.py:25-37,128-137`

6. Optional baseline update:
- copy output to baseline path.  
Source: `src/ask_my_docs/cli.py:238-242`

7. Optional gate:
- YAML loaded by `load_gate_config`
- baseline aggregate parsed if available
- `evaluate_gate` returns `GateResult(passed, failures)`  
Source: `src/ask_my_docs/cli.py:244-267`, `src/ask_my_docs/evaluation/gating.py:45-124`

#### Real eval input schema

Each `data/eval/eval_set.jsonl` row uses:
- `question: str`
- `reference_answer: str`
- `expected_doc_ids: list[str]`  
Source: `data/eval/eval_set.jsonl:1-5`

#### Real eval output schema

Report top-level keys:
- `aggregate`
- `per_example`  
Source: `artifacts/baseline_metrics.json:2,12`

Aggregate keys include:
- `answer_f1_mean`
- `avg_cost_usd`
- `latency_p50_ms`
- `latency_p95_ms`
- `retrieval_recall_at_k_mean`  
Source: `artifacts/baseline_metrics.json:3-10`

Per-example fields include:
- `predicted_answer`
- `retrieved_doc_ids`
- `total_tokens`  
Source: `artifacts/baseline_metrics.json:23,28,34` (and repeated blocks)

### 3.4 Flow D: `metrics-summary` (DuckDB aggregate observability)

1. CLI invokes `metrics_store.summarize(limit=...)`.  
Source: `src/ask_my_docs/cli.py:157-176`

2. SQL computes:
- `COUNT(*)`
- `quantile_cont(latency_ms, 0.50)` as `latency_p50_ms`
- `quantile_cont(latency_ms, 0.95)` as `latency_p95_ms`
- averages for cost/recall/F1/EM  
Source: `src/ask_my_docs/observability/metrics_store.py:279-285`

3. Summaries return default zeros if table empty/none.  
Source: `src/ask_my_docs/observability/metrics_store.py:290-307`

4. CLI prints formatted key=value lines.  
Source: `src/ask_my_docs/cli.py:163-177`

Test-verified behaviors:
- `metrics-summary` works even when retrieval index is missing.  
  Source: `tests/test_cli.py:27-32`
- non-positive limit is rejected.  
  Source: `tests/test_metrics_store.py:43`, `tests/test_cli.py:36`

### 3.5 Trace and metrics data structures (actual)

`traces.jsonl` row keys:
- `name`, `trace_id`, `span_id`, `parent_span_id`
- `start_time_unix_ns`, `end_time_unix_ns`, `duration_ms`
- `status`, `attributes`  
Source: `src/ask_my_docs/observability/tracing.py:32-44`

Observed span names and attributes in artifact:
- `rag.retrieve`, `rag.generate`, `rag.request`
- request attributes include `rag.request_id`, `rag.top_k`, `rag.model_name`, `rag.latency_ms`, `rag.estimated_cost_usd`, `rag.retrieved_doc_count`  
Source: `artifacts/notebook_tutorial/observability/traces.jsonl:1-3`

DuckDB metrics table columns:
- `request_id`, `trace_id`, `model_name`, `question`
- `latency_ms`, `retrieval_latency_ms`, `generation_latency_ms`
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `estimated_cost_usd`, `retrieval_recall_at_k`, `answer_f1`, `exact_match`, `timestamp_utc`  
Source: `src/ask_my_docs/observability/metrics_store.py:69-85`

---

## 🛠️ Module 4: Step-by-Step Setup & Development Guide

This module explains exactly how to bring up this repository on a clean machine and how boot actually happens internally.

### 4.1 Environment prerequisites (exact)

1. OS: Linux/macOS.  
   Source: `README.md:61`
2. Python: `>=3.11` (README recommends `3.12.10`).  
   Source: `pyproject.toml:6`, `README.md:62`
3. Package/env manager: `uv`.  
   Source: `README.md:63`
4. Ollama installed and running locally with at least one local model.  
   Source: `README.md:64-66`

### 4.2 Installation path (documented project path)

Documented setup sequence:
1. Clone repo and enter directory.
2. Create venv with `uv venv --python 3.12.10`.
3. Activate env.
4. Install dependencies via `uv sync --all-extras`.  
Source: `README.md:69-77`

Tooling config that backs this:
- script entrypoint `ask-my-docs = "ask_my_docs.cli:app"`.  
  Source: `pyproject.toml:32-33`
- pytest source root and test paths.  
  Source: `pyproject.toml:42-44`
- ruff + mypy strict configuration.  
  Source: `pyproject.toml:46-55`

### 4.3 `.env` / environment variable contract (exact names + defaults)

All settings use prefix `ASK_MY_DOCS_`.  
Source: `src/ask_my_docs/settings.py:35`, `README.md:81`

| Variable | Default in code | Purpose |
|---|---|---|
| `ASK_MY_DOCS_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `ASK_MY_DOCS_OLLAMA_MODEL` | `None` | Optional pinned model |
| `ASK_MY_DOCS_OLLAMA_TIMEOUT_SECONDS` | `120.0` | Generate timeout |
| `ASK_MY_DOCS_OLLAMA_LIST_TIMEOUT_SECONDS` | `10.0` | `ollama list` timeout |
| `ASK_MY_DOCS_DOCS_DIR` | `data/docs` | Source corpus directory |
| `ASK_MY_DOCS_EVAL_PATH` | `data/eval/eval_set.jsonl` | Eval dataset |
| `ASK_MY_DOCS_ARTIFACTS_DIR` | `artifacts` | Top-level artifacts path |
| `ASK_MY_DOCS_INDEX_DIR` | `artifacts/index` | Retriever artifacts |
| `ASK_MY_DOCS_TRACES_PATH` | `artifacts/observability/traces.jsonl` | Trace log output |
| `ASK_MY_DOCS_METRICS_DB_PATH` | `artifacts/observability/metrics.duckdb` | Observability DB |
| `ASK_MY_DOCS_THRESHOLDS_PATH` | `configs/regression_thresholds.yaml` | Gate policy config |
| `ASK_MY_DOCS_OBSERVABILITY_STORE_RAW_QUESTIONS` | `False` | Privacy toggle |

Defaults source: `src/ask_my_docs/settings.py:43-55` and README table `README.md:87-97`.

### 4.4 Conceptual boot sequence (what happens when you run each command)

#### `ask-my-docs ingest`
1. Read docs.
2. Build retriever.
3. Persist index artifacts for later querying.  
Source: `src/ask_my_docs/cli.py:93-114`

#### `ask-my-docs ask "..." --top-k N`
1. Build runtime pipeline components.
2. Load retriever from index artifacts.
3. Retrieve top-k chunks.
4. Build grounded prompt and call Ollama.
5. Compute cost/latency/citations.
6. Persist request metrics row and traces.
7. Print summary and optional JSON output.  
Source: `src/ask_my_docs/cli.py:56-90,118-154`, `src/ask_my_docs/pipeline.py:42-133`

#### `ask-my-docs eval ... [--gate] [--set-baseline]`
1. Load eval JSONL.
2. Run full ask flow per example.
3. Score quality metrics.
4. Compute aggregate report.
5. Optionally set baseline.
6. Optionally enforce gate and fail command on violations.  
Source: `src/ask_my_docs/cli.py:188-273`, `src/ask_my_docs/evaluation/runner.py:69-137`, `src/ask_my_docs/evaluation/gating.py:52-124`

### 4.5 Development/CI lifecycle

CI job `quality-and-regression-gate` runs:
1. `uv sync --all-extras`
2. `ruff check src tests`
3. `mypy src`
4. `pytest -q`
5. `ask-my-docs ingest`
6. `ask-my-docs eval` for candidate metrics
7. `ask-my-docs eval --gate` with baseline + thresholds  
Source: `.github/workflows/ci.yml:9-49`

### 4.6 Notebook learning path already included in repo

Notebook `tutorial_observability_ollama.ipynb` is a real staged walkthrough:
1. Configure isolated tutorial artifacts.
2. Discover local Ollama models.
3. Build index.
4. Build pipeline.
5. Ask one question.
6. Evaluate.
7. Apply gate.
8. Inspect DuckDB summary.
9. Inspect trace samples.
10. Next improvements.  
Source: `notebooks/tutorial_observability_ollama.ipynb:14,85,140,204,263,340,395,450,515,559`

---

## 💼 Module 5: Tech Interview & Hiring Preparation

### 5.1 Five core technical interview questions

1. **Why did this repo choose deterministic hash embeddings instead of model embeddings, and what tradeoff does that create in retrieval quality vs reproducibility?**
2. **Walk through exactly how `RAGPipeline.answer` enforces observability and privacy in one request lifecycle.**
3. **How does `HybridRetriever.search` combine BM25 and FAISS scores, and where can ranking instability happen?**
4. **How does the regression gate prevent silent quality/cost/latency drift in CI, and what assumptions does it make about baseline validity?**
5. **What failure modes can break index compatibility between ingest and query, and how does the code guard against them?**

### 5.2 Three hard engineering scenarios

1. **Database latency triples:** `metrics-summary` and request writes become slow. How would you optimize `MetricsStore` while preserving current schema behavior and upsert semantics?
2. **Model fleet changes:** `ollama list` now returns mostly cloud rows (`size "-"`) and intermittent timeout spikes. How should model-selection robustness evolve while keeping the current fallback intent?
3. **Domain growth:** docs expand from 5 to 50k documents. What specific code points must change to keep retrieval and evaluation practical?

### 5.3 Detailed model answers

#### Q1 Answer
The deterministic embedder is explicitly implemented as a local hashing projection (`HashingEmbedder.embed`) with no external embedding model dependency. This makes indexing reproducible and offline-friendly (same tokens -> same vectors), which aligns with local/Ollama-first architecture.  
Tradeoff: semantic quality ceiling is lower than modern learned embedding models, and README explicitly acknowledges this limitation.  
Evidence: `src/ask_my_docs/retrieval/hybrid.py:67-87`, `README.md:275`.

#### Q2 Answer
`RAGPipeline.answer` creates request-scoped telemetry and IDs:
- `request_id = str(uuid4())`
- spans for `rag.request`, `rag.retrieve`, `rag.generate`
- attributes for top-k, model, latency, estimated cost, retrieved count  
Then it computes tokens/cost/citations and persists one `RequestMetricRecord`. Privacy is enforced by `_question_for_storage`: raw question is redacted hash unless `store_raw_questions` is enabled.  
Evidence: `src/ask_my_docs/pipeline.py:50-55,58-67,88-93,113-132,150-154`; tests `tests/test_pipeline.py:55,68,82`.

#### Q3 Answer
Search path:
1. BM25 lexical scores.
2. FAISS semantic scores for top `search_k`.
3. Min-max normalize each score vector.
4. Weighted sum with configured lexical/semantic weights.
5. Candidate selection via `argpartition`, then sorted tie-break using combined -> lexical -> semantic.  
Instability can happen when many normalized scores collapse to near-equal values (especially with sparse token overlap or low-variance vectors), causing tie-heavy ranking dependence on secondary sort fields.  
Evidence: `src/ask_my_docs/retrieval/hybrid.py:340-369`, config `src/ask_my_docs/settings.py:26-27`.

#### Q4 Answer
Gate design enforces two layers:
- Absolute minima/maxima (`min_answer_f1`, `max_latency_p95_ms`, etc.).
- Relative baseline drift caps (`max_answer_f1_drop`, etc.) when baseline exists.  
In CI, this blocks regressions after candidate metrics are computed. Baseline validity assumption is crucial: stale or poor baseline weakens governance. The CLI also validates baseline payload has `aggregate`.  
Evidence: `configs/regression_thresholds.yaml:1-11`, `src/ask_my_docs/evaluation/gating.py:52-124`, `src/ask_my_docs/cli.py:244-257`, `.github/workflows/ci.yml:38-49`.

#### Q5 Answer
Compatibility guards include:
- chunk/embedding row count alignment.
- embedding dimensional consistency across config, embeddings, and FAISS index.
- optional manifest consistency checks (`chunk_count`, `embedding_dim`).  
These checks fail early with explicit `ValueError` messages, and tests verify mismatch handling.  
Evidence: `src/ask_my_docs/retrieval/hybrid.py:130-172,238-307`, `tests/test_retrieval.py:38`.

#### Scenario 1 Answer (DB latency triples)
Keep architecture, optimize internals:
1. Confirm bottleneck split between write path (`record`) and read path (`summarize`) using span-level and DB-level timings.
2. Preserve current upsert semantics (`ON CONFLICT(request_id)`) to maintain idempotent request row behavior.
3. Add DB maintenance and possibly partitioning/rollup strategy while keeping `rag_request_metrics` contract stable.
4. For summary-heavy workloads, materialize rolling aggregates and keep existing `summarize` API unchanged.
5. Avoid schema-breaking changes because tests and CLI formatting depend on specific keys (`latency_p50_ms`, etc.).  
Evidence anchor: `src/ask_my_docs/observability/metrics_store.py:218-251,266-313`, tests `tests/test_metrics_store.py:36-38,88-89`.

#### Scenario 2 Answer (model discovery instability)
Current behavior:
- parse rows from `ollama list`.
- prefer configured model if present.
- else prefer first local model (`size != "-"`).
- timeout raises runtime error.  
Enhancements:
1. Cache successful model name with TTL and fallback to cached local model when `ollama list` intermittently times out.
2. Add retry with backoff for `_list_models`.
3. Keep explicit error if configured model is missing, but include candidate hinting.
4. Preserve current local-first selection intent.  
Evidence anchor: `src/ask_my_docs/llm/ollama.py:63-96,125-148`; test timeout behavior `tests/test_ollama.py:43,53`.

#### Scenario 3 Answer (50k docs)
Specific pressure points and likely changes:
1. `load_documents` and in-memory chunk list scale limits; move toward streaming ingestion/chunk persistence.
2. FAISS index type: `IndexFlatIP` is brute-force; switch to approximate indexes for high scale.
3. Hash embeddings quality and collision risks grow with corpus size; replace with stronger embedding model pipeline while retaining manifest compatibility checks.
4. Evaluation runner currently loops synchronously over examples; introduce batching/parallelism and richer dataset stratification.
5. Metrics DB summarization still works, but retention/archival strategy becomes mandatory.  
Evidence anchor: `src/ask_my_docs/retrieval/hybrid.py:175-202`, `src/ask_my_docs/evaluation/runner.py:69-106`, `src/ask_my_docs/observability/metrics_store.py:266-313`.

---

## Appendix A: Real Constants and Contracts You Must Memorize

1. Env prefix: `ASK_MY_DOCS_` (`settings.py:35`).
2. Retrieval defaults: `120/20`, `top_k=4`, `0.45/0.55`, `dim=256` (`settings.py:23-28`).
3. Ollama generation temp: `0.1` (`ollama.py:177`).
4. Citation regex accepts both `[doc_id: X]` and `[doc_id=X]` (`pipeline.py:20`; tested in `tests/test_pipeline.py:97-101`).
5. Metrics table name: `rag_request_metrics` (`metrics_store.py:69`).
6. Quantile metrics computed directly in DuckDB (`metrics_store.py:279-280`).
7. Gate thresholds from YAML:
- absolute min F1 `0.60`
- absolute min recall@k `0.95`
- absolute max p95 latency `70000.0`
- absolute max avg cost `0.0004`
- max F1 drop `0.05`
- max recall drop `0.02`
- max p95 increase `15000.0`
- max avg cost increase `0.00008`  
Source: `configs/regression_thresholds.yaml:1-11`

## Appendix B: Static-Learning Exercises (Zero -> Hero progression)

1. Trace one command end-to-end (`ask`) by opening only `cli.py`, `pipeline.py`, `llm/ollama.py`, `retrieval/hybrid.py`, `metrics_store.py` and writing the call chain yourself.
2. Recompute one sample cost manually from `prompt_tokens` and `completion_tokens` in `sample_answer.json` using pricing defaults in `settings.py`.
3. Manually verify whether one baseline per-example answer has citation list consistent with `retrieved_doc_ids`.
4. Inspect how many failure strings gate can emit and map each to one specific threshold field.
5. Compare README claimed architecture to actual code paths and note any drift.

## Appendix C: PDF Handbook Status

A companion file `ZERO_TO_HERO_PDF_EXPORT.md` is included with deterministic export instructions so you can generate a print-ready PDF locally without changing repository code.

