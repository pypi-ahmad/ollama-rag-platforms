# Local SLM Benchmark Report

- Run ID: `notebook_run_20260612_082848`
- Started (UTC): `2026-06-12T02:58:48.970839+00:00`
- Completed (UTC): `2026-06-12T03:02:17.525526+00:00`
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
| granite4.1:3b | 10 | 1.0704 | 2.4492 | 104.1202 | 0.6667 | 1.138 | 0.3834 | 0.8211 |
| phi3.5:3.8b | 10 | 15.8485 | 53.1894 | 94.981 | 0.6637 | 0.0769 | 0.3804 | 0.799 |
| functiongemma:270m | 10 | 1.2182 | 1.3502 | 188.3842 | 0.2833 | 1.0 | 0.0 | 0.655 |

## Quality vs Speed Tradeoffs

- Fastest decode throughput: functiongemma:270m at 188.3842 tok/s.
- Best average quality proxy: granite4.1:3b with score 0.6667 (0-1 scale).
- Best quality/speed balance: granite4.1:3b (balanced_score=0.8211).
- Privacy: all prompts and outputs stay on-device when Ollama runs locally.
- Latency: smaller models usually reduce p95 latency but may fail structure/correctness checks.
- Cost: local inference avoids per-token API billing but still incurs hardware and electricity cost.

## Cost Assumption

- `estimated_benchmark_cost_usd` is only computed if `assumed_power_watts` and `electricity_rate_usd_per_kwh` are set in config.
- This estimate covers the benchmark run itself, not full lifecycle costs (hardware purchase, maintenance).
