# Local SLM Benchmark Report

- Run ID: `real_run_20260612_v2`
- Started (UTC): `2026-06-11T19:11:40.723590+00:00`
- Completed (UTC): `2026-06-11T19:12:43.161671+00:00`
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
| phi3.5:3.8b | 10 | 1.3568 | 2.3789 | 98.404 | 0.6559 | 0.4387 | 0.3726 | 0.6544 |
| qwen3.5:9b | 10 | 2.4658 | 4.7413 | 44.5181 | 0.7326 | 0.2414 | 0.4493 | 0.653 |
| functiongemma:270m | 10 | 0.5952 | 0.6544 | 335.7286 | 0.2833 | 1.0 | 0.0 | 0.632 |

## Quality vs Speed Tradeoffs

- Fastest decode throughput: functiongemma:270m at 335.7286 tok/s.
- Best average quality proxy: qwen3.5:9b with score 0.7326 (0-1 scale).
- Best quality/speed balance: phi3.5:3.8b (balanced_score=0.6544).
- Privacy: all prompts and outputs stay on-device when Ollama runs locally.
- Latency: smaller models usually reduce p95 latency but may fail structure/correctness checks.
- Cost: local inference avoids per-token API billing but still incurs hardware and electricity cost.

## Cost Assumption

- `estimated_benchmark_cost_usd` is only computed if `assumed_power_watts` and `electricity_rate_usd_per_kwh` are set in config.
- This estimate covers the benchmark run itself, not full lifecycle costs (hardware purchase, maintenance).
