# Ollama Rag Platforms

Local LLM, RAG, and agent platform projects powered by Ollama-centric stacks.

## Repository Purpose

This repository groups related projects into a single, navigable codebase with consistent structure and indexing.

## Project Index

| # | Project | Folder | Source Directory | Source Repository | Source Commit |
|---|---|---|---|---|---|
| 1 | `duckdb-analytics-mcp` | `duckdb-analytics-mcp` | `duckdb-analytics-mcp` | https://github.com/pypi-ahmad/duckdb-analytics-mcp.git | `47e16c55f9` |
| 2 | `local-slm-ollama-benchmark` | `local-slm-ollama-benchmark` | `local-slm-ollama-benchmark` | https://github.com/pypi-ahmad/local-slm-ollama-benchmark.git | `ebb3f48818` |
| 3 | `multi-agent-system-ollama` | `multi-agent-system-ollama` | `multi-agent-system-ollama` | https://github.com/pypi-ahmad/multi-agent-system-ollama.git | `ef76d29df3` |
| 4 | `offline-ollama-rag-app` | `offline-ollama-rag-app` | `offline-ollama-rag-app` | https://github.com/pypi-ahmad/offline-ollama-rag-app.git | `de95d3e09b` |
| 5 | `production-rag-ask-my-docs` | `production-rag-ask-my-docs` | `production-rag-ask-my-docs` | https://github.com/pypi-ahmad/production-rag-ask-my-docs.git | `75b40e645b` |
| 6 | `production-rag-ask-my-docs-learning` | `production-rag-ask-my-docs-learning` | `production-rag-ask-my-docs-learning` | https://github.com/pypi-ahmad/production-rag-ask-my-docs-learning.git | `6b5f3fa61c` |
| 7 | `rag-telemetry-evals-ollama` | `rag-telemetry-evals-ollama` | `rag-telemetry-evals-ollama` | https://github.com/pypi-ahmad/rag-telemetry-evals-ollama.git | `befa1e70b4` |

## Layout

- Each top-level folder is a standalone project migrated from the source workspace.
- Heavy local-only artifacts (virtual environments, datasets, model weights, caches) are intentionally excluded.

## Getting Started

```bash
git clone <this-repo-url>
cd <this-repo-folder>
cd <project-folder>
```

## Maintenance Notes

- Keep project-level documentation inside each project folder.
- Use this repository as a curated portfolio layer across related workstreams.
