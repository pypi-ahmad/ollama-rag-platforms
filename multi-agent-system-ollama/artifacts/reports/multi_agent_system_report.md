# Multi-Agent System Report

Generated at: `2026-06-12T11:40:48+00:00`

## Setup

- Planner model: `phi3.5:3.8b`
- Reviewer model: `functiongemma:270m`
- Tasks evaluated: **6**

## Metrics

- Baseline keyword recall mean: **0.0000**
- Multi-agent keyword recall mean: **0.8333**
- Keyword recall gain: **0.8333**
- Retrieval hit rate: **1.0000**
- Source citation rate: **1.0000**
- Planner fallback rate: **1.0000**
- Reviewer fallback rate: **0.0000**
- Average total latency (ms): **13027.57**

## Telemetry

- Total spans: **30**
- Unique traces: **6**

| Span | Count | Mean ms | P95 ms | Errors |
|---|---:|---:|---:|---:|
| baseline_answer | 6 | 5011.008 | 5024.523 | 0 |
| plan_answer | 6 | 8016.021 | 8024.712 | 0 |
| retrieve | 6 | 0.068 | 0.077 | 0 |
| review_answer | 6 | 0.016 | 0.023 | 0 |
| route | 6 | 0.014 | 0.018 | 0 |


## Per-task Keyword Recall

| Task ID | Baseline | Multi-Agent | Gain | Retrieval Hit |
|---|---:|---:|---:|---:|
| t1 | 0.000 | 1.000 | 1.000 | 1 |
| t2 | 0.000 | 1.000 | 1.000 | 1 |
| t3 | 0.000 | 0.000 | 0.000 | 1 |
| t4 | 0.000 | 1.000 | 1.000 | 1 |
| t5 | 0.000 | 1.000 | 1.000 | 1 |
| t6 | 0.000 | 1.000 | 1.000 | 1 |
