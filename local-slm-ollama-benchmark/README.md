# Local SLM App with Ollama (Offline Benchmark Project)

This project builds a **fully local SLM workflow** on top of Ollama:

- run models offline on your own hardware
- benchmark inference performance across 3 models on the same machine
- compare quality vs speed with reproducible artifacts
- use a local CLI chat app for day-to-day testing

It is designed for real constraints: **privacy, latency, and cost**.

## Background

Cloud APIs are great for rapid prototyping, but many applied AI workloads have hard constraints:

- **Privacy/compliance**: sensitive prompts and outputs must remain on-device.
- **Latency**: WAN/API roundtrips can dominate user-perceived response time.
- **Cost**: per-token billing can grow nonlinearly with traffic.

This project helps evaluate when local inference is the right fit.

## What is implemented

- `local-slm-ollama-benchmark benchmark`:
  - runs the same prompt suite across 3 local models
  - captures wall-clock latency + Ollama timings + decode throughput
  - scores outputs with deterministic quality rubrics
  - writes `raw_results.json`, `prompt_runs.csv`, `model_summary.csv`, and `benchmark_report.md`
- `local-slm-ollama-benchmark chat`:
  - interactive local chat against any installed Ollama model
  - prints per-turn latency/tokens-per-second

## Project layout

```text
local-slm-ollama-benchmark/
├── configs/
│   └── benchmark.toml              # models, prompts, generation + runtime settings
├── artifacts/
│   └── real_run_20260612_v2/       # real benchmark artifacts from this build
├── src/local_slm_ollama_benchmark/
│   ├── benchmark.py                # end-to-end benchmark orchestration + report writers
│   ├── cli.py                      # benchmark/chat commands
│   ├── config.py                   # Pydantic config models + TOML loader
│   ├── ollama_client.py            # async Ollama HTTP client
│   ├── quality.py                  # rubric-based quality scoring
│   └── metrics.py                  # timing/throughput helpers
├── tests/
│   ├── test_quality.py
│   └── test_metrics.py
└── pyproject.toml
```

## Prerequisites

- Linux/macOS with Ollama installed and running
- Python managed via `uv`
- 3 local models available in Ollama

Example model setup used in this project:

```bash
ollama pull functiongemma:270m
ollama pull phi3.5:3.8b
ollama pull qwen3.5:9b
```

## Quickstart

```bash
cd local-slm-ollama-benchmark
uv sync
```

Run tests:

```bash
uv run pytest -q
```

Run the benchmark:

```bash
uv run local-slm-ollama-benchmark benchmark \
  --config configs/benchmark.toml \
  --output-dir artifacts \
  --run-name real_run_20260612_v2
```

Run local chat:

```bash
uv run local-slm-ollama-benchmark chat --model phi3.5:3.8b --think false
```

## Results Summary

Run ID: `real_run_20260612_v2`

Hardware/runtime captured automatically:

- CPU: AMD Ryzen 7 7735HS
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- RAM: 30.09 GB
- Ollama: 0.30.6

Model summary (`artifacts/real_run_20260612_v2/model_summary.csv`):

| Model | Avg Latency (s) | P95 Latency (s) | Avg Tok/s | Avg Quality | Balanced Score |
|---|---:|---:|---:|---:|---:|
| `phi3.5:3.8b` | 1.3568 | 2.3789 | 98.4040 | 0.6559 | 0.6544 |
| `qwen3.5:9b` | 2.4658 | 4.7413 | 44.5181 | 0.7326 | 0.6530 |
| `functiongemma:270m` | 0.5952 | 0.6544 | 335.7286 | 0.2833 | 0.6320 |

### Tradeoff interpretation

- **Fastest**: `functiongemma:270m` (very high tokens/sec, lowest latency) but weak quality on general tasks.
- **Highest quality**: `qwen3.5:9b` on this rubric, but clearly slower.
- **Best balance**: `phi3.5:3.8b` (near-top balanced score with much lower latency than `qwen3.5:9b`).

This is exactly the practical quality-vs-speed frontier you need for production selection.

## Important implementation note (`think=false`)

`qwen3.5:9b` can emit hidden reasoning traces unless you disable thinking mode.
If `think` is on, `response` can be empty while tokens are still generated.

This project sets `generation.think = false` in `configs/benchmark.toml` for fair output-based comparisons.

## How quality scoring works

Quality scoring is deterministic and transparent (no external judge model):

- `keywords`: required terms must appear
- `json_keys`: valid JSON and required keys present
- `exact_match`: strict answer match
- optional `max_words` soft-penalty for verbosity

This keeps benchmarking offline and reproducible. You can edit prompts/rubrics in `configs/benchmark.toml`.

## Output artifacts

Each run writes:

- `raw_results.json`: full responses + per-call timing/quality details
- `prompt_runs.csv`: one row per prompt run
- `model_summary.csv`: aggregate model-level metrics
- `benchmark_report.md`: human-readable benchmark report

## Privacy, latency, and cost constraints (practical view)

- **Privacy**: local Ollama execution keeps prompt/response data on your machine, reducing exposure to third-party API logging.
- **Latency**: local execution removes network roundtrips; model size still drives inference delay.
- **Cost**: local removes per-token API billing but shifts cost to hardware + electricity + ops maintenance.

For cost estimation, set in config:

```toml
[cost]
electricity_rate_usd_per_kwh = 0.15
assumed_power_watts = 85
```

Then benchmark output will include `estimated_benchmark_cost_usd` (for the run duration).

## Example chat output (real)

```text
You > Give me one sentence on why local inference helps privacy.
phi3.5:3.8b > Local inference enhances privacy by processing sensitive data directly on the user's device...
latency=3.62s decode_tps=98.93 eval_tokens=35
```

## Next customizations

- Replace prompt suite with your domain tasks (customer support, claim triage, report drafting).
- Add domain-specific quality checks (exact fields, policy constraints, abstention behavior).
- Run repeated benchmarks after quantization/model updates to track drift.

## Setup

```bash
git clone https://github.com/pypi-ahmad/local-slm-ollama-benchmark.git
cd local-slm-ollama-benchmark
```
