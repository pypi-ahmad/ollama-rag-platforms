# RAG Telemetry + Evaluation Report

Generated at: `2026-06-12T11:36:14+00:00`

## Runtime Configuration

- Chat model: `phi3.5:3.8b`
- Embedding model: `embeddinggemma:latest`
- Questions evaluated: **8**

## Core Metrics

- Retrieval hit rate: **1.0000**
- RAG cites expected source rate: **1.0000**
- Baseline keyword recall mean: **0.0625**
- RAG keyword recall mean: **1.0000**
- Keyword recall gain: **0.9375**
- Baseline semantic similarity mean: **0.0674**
- RAG semantic similarity mean: **0.3222**
- Semantic similarity gain: **0.2548**
- Baseline latency mean (ms): **2789.48**
- RAG latency mean (ms): **1953.58**

## Telemetry Summary

- Total spans: **40**
- Unique traces: **16**

| Span | Count | Mean ms | P95 ms | Errors |
|---|---:|---:|---:|---:|
| answer_baseline | 8 | 14053.542 | 15018.568 | 0 |
| answer_rag | 8 | 23659.074 | 27035.473 | 0 |
| generate_baseline | 8 | 14051.945 | 15018.2 | 6 |
| generate_rag | 8 | 13216.854 | 15017.881 | 6 |
| retrieve | 8 | 10439.702 | 12014.285 | 0 |


## Per-question Metrics

| ID | Retrieval Hit | Baseline Keyword | RAG Keyword | Baseline Semantic | RAG Semantic |
|---|---:|---:|---:|---:|---:|
| q1 | 1 | 0.000 | 1.000 | 0.000 | 0.054 |
| q2 | 1 | 0.000 | 1.000 | 0.205 | 0.638 |
| q3 | 1 | 0.000 | 1.000 | 0.000 | 0.149 |
| q4 | 1 | 0.000 | 1.000 | 0.000 | 0.088 |
| q5 | 1 | 0.000 | 1.000 | 0.000 | 0.095 |
| q6 | 1 | 0.000 | 1.000 | 0.034 | 0.095 |
| q7 | 1 | 0.000 | 1.000 | 0.241 | 0.736 |
| q8 | 1 | 0.500 | 1.000 | 0.060 | 0.724 |
