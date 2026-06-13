# Local SLM Benchmark Report

- Run ID: `granite_3b_run_20260612`
- Started (UTC): `2026-06-12T02:41:59.133543+00:00`
- Completed (UTC): `2026-06-12T02:43:05.743748+00:00`
- Ollama host: `http://127.0.0.1:11434`

## Hardware and Runtime

| Field | Value |
|---|---|
| python | 3.14.5 |
| platform | Linux-7.0.0-22-generic-x86_64-with-glibc2.43 |
| cpu_model | AMD Ryzen 7 7735HS with Radeon Graphics |
| logical_cores | 16 |
| memory_gb | 30.09 |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB |
| ollama_version | ollama version is 0.30.6 |

## Model Comparison

| Model | Samples | Avg Latency (s) | P95 Latency (s) | Avg Tok/s | Avg Quality | Speedup vs Baseline | Quality Delta vs Baseline | Balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| granite4.1:3b | 10 | 0.9676 | 2.2728 | 107.752 | 0.6667 | 1.2582 | 0.3334 | 1.0 |
| phi3.5:3.8b | 10 | 1.3327 | 2.3514 | 98.5975 | 0.6559 | 0.9134 | 0.3226 | 0.9563 |
| functiongemma:270m | 10 | 1.2174 | 1.5759 | 83.6432 | 0.3333 | 1.0 | 0.0 | 0.6105 |

## Quality vs Speed Tradeoffs

- Fastest decode throughput: granite4.1:3b at 107.752 tok/s.
- Best average quality proxy: granite4.1:3b with score 0.6667 (0-1 scale).
- Best quality/speed balance: granite4.1:3b (balanced_score=1.0).
- Privacy: all prompts and outputs stay on-device when Ollama runs locally.
- Latency: smaller models usually reduce p95 latency but may fail structure/correctness checks.
- Cost: local inference avoids per-token API billing but still incurs hardware and electricity cost.

## Cost Assumption

- `estimated_benchmark_cost_usd` is only computed if `assumed_power_watts` and `electricity_rate_usd_per_kwh` are set in config.
- This estimate covers the benchmark run itself, not full lifecycle costs (hardware purchase, maintenance).
