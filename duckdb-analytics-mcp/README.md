# DuckDB Analytics MCP

[![CI](https://github.com/pypi-ahmad/duckdb-analytics-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/pypi-ahmad/duckdb-analytics-mcp/actions/workflows/ci.yml)
[![Python 3.12.10](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-grade, read-only MCP server for querying local analytics datasets (`.csv`, `.parquet`, `.json`, `.jsonl`) with DuckDB.

## Table of Contents

- [Project Status](#project-status)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [MCP Client Integration](#mcp-client-integration)
- [Tools and Resources](#tools-and-resources)
- [Security Model](#security-model)
- [Limitations](#limitations)
- [Development and Testing](#development-and-testing)
- [Docker](#docker)
- [Versioning and Releases](#versioning-and-releases)
- [Contributing and Community](#contributing-and-community)
- [License](#license)

## Project Status

- Active and maintained.
- Current language/runtime target: Python `3.12.10`.
- Scope: read-only analytics over local datasets exposed through MCP.

## Key Features

- MCP transports: `stdio` and `streamable-http`
- Read-only SQL over a constrained `source` alias
- AST-based SQL guardrails (`sqlglot`) to prevent mutation and external file reads
- Query pagination + timeout enforcement with DuckDB interruption
- Tool response formats: `markdown` and `json`
- Dataset catalog and schema resources
- Optional bearer-token auth with required scope enforcement

## Architecture

Core modules:

- `server.py`: MCP tool/resource registration, request validation, auth, response/error shaping
- `service.py`: DuckDB execution, timeout behavior, pagination, health
- `security.py`: SQL guardrails and query-shape enforcement
- `catalog.py`: dataset discovery, cache, and safe path handling
- `config.py`: environment-based runtime settings

Request flow:

1. Client calls MCP tool/resource.
2. Request is validated with Pydantic models.
3. Dataset is resolved from catalog and registered as `source`.
4. Guarded SQL executes in constrained DuckDB connection.
5. Server returns structured result or sanitized error.

## Requirements

- Linux/macOS with `uv` installed
- Python `3.12.10`
- Local dataset files under configured `DATASET_DIR`

## Quickstart

```bash
git clone https://github.com/pypi-ahmad/duckdb-analytics-mcp.git
cd duckdb-analytics-mcp
uv python pin 3.12.10
uv venv --python 3.12.10 --allow-existing
uv sync --dev
cp .env.example .env
```

## Configuration

All settings are env-driven (`.env` supported via `pydantic-settings`).

| Variable | Default | Description |
|---|---|---|
| `SERVER_NAME` | `duckdb_analytics_mcp` | MCP server name |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) |
| `HOST` | `127.0.0.1` | HTTP bind host |
| `PORT` | `8000` | HTTP bind port |
| `TRANSPORT` | `streamable-http` | Default transport (`stdio` or `streamable-http`) |
| `DATASET_DIR` | `data` | Dataset root path |
| `DEFAULT_LIMIT` | `25` | Default list/query page size |
| `MAX_LIMIT` | `200` | Max list/query page size |
| `MAX_QUERY_CHARS` | `4000` | Max SQL text length |
| `QUERY_TIMEOUT_SECONDS` | `20` | Timeout for describe/query operations |
| `MAX_SAMPLE_ROWS` | `25` | Max sample rows for describe |
| `CATALOG_CACHE_TTL_SECONDS` | `5` | Dataset catalog cache TTL (`0` disables cache) |
| `DUCKDB_THREADS` | `4` | DuckDB thread count |
| `DUCKDB_MEMORY_LIMIT` | `1GB` | DuckDB memory cap |
| `STATIC_BEARER_TOKEN` | unset | Enables token auth when auth URLs are also set |
| `AUTH_ISSUER_URL` | unset | Required with static token |
| `AUTH_RESOURCE_SERVER_URL` | unset | Required with static token |
| `AUTH_REQUIRED_SCOPE` | `analytics.read` | Required token scope |

Auth requirement:

- If `STATIC_BEARER_TOKEN` is set, both `AUTH_ISSUER_URL` and `AUTH_RESOURCE_SERVER_URL` must be set.

## Running the Server

### Streamable HTTP

```bash
uv run duckdb-analytics-mcp run --transport streamable-http
```

Endpoint: `http://127.0.0.1:8000/mcp`

### stdio

```bash
uv run duckdb-analytics-mcp run --transport stdio
```

### Doctor

```bash
uv run duckdb-analytics-mcp doctor
```

## MCP Client Integration

### Generic stdio config

```json
{
  "command": "uv",
  "args": ["run", "duckdb-analytics-mcp", "run", "--transport", "stdio"],
  "cwd": "/path/to/duckdb-analytics-mcp",
  "env": {
    "DATASET_DIR": "data"
  }
}
```

### Generic streamable-http base URL

```text
http://127.0.0.1:8000/mcp
```

## Tools and Resources

### Tools

- `duckdb_analytics_health`
- `duckdb_analytics_list_datasets`
- `duckdb_analytics_describe_dataset`
- `duckdb_analytics_query_dataset`

### Resources

- `dataset://catalog`
- `dataset://schema/{dataset_name}`

## Security Model

`duckdb_analytics_query_dataset` enforces:

- single statement only
- no SQL comments
- read-only query shape (`SELECT`/`WITH` query AST)
- source isolation: only `source` alias and in-query CTE references are allowed
- external relation and file-read patterns blocked (`read_*`, `*_scan`, and non-identifier table sources)

Additional controls:

- extension autoload/autoinstall disabled in DuckDB runtime
- auth gate available via static bearer token and required scope
- sanitized error messages for DB/internal failures

## Limitations

- Read-only analytics only (no DDL/DML support)
- Very expensive queries may still require tighter memory/thread/timeout tuning
- Empty, high-offset pages may trigger fallback count query for total row metadata

## Development and Testing

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run pytest -q -m e2e
```

Test coverage includes:

- unit tests for SQL guardrails, catalog behavior, service logic
- tool-level error handling tests
- real end-to-end tests for `stdio`, `streamable-http`, and auth on/off flows

## Docker

```bash
docker build -t duckdb-analytics-mcp:latest .
docker run --rm -p 8000:8000 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e TRANSPORT=streamable-http \
  duckdb-analytics-mcp:latest
```

Or:

```bash
docker compose up --build
```

## Versioning and Releases

- Semantic Versioning (`MAJOR.MINOR.PATCH`)
- Notable changes tracked in [CHANGELOG.md](CHANGELOG.md)
- Breaking changes documented explicitly in release notes

## Contributing and Community

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security reporting: [SECURITY.md](SECURITY.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## License

MIT License. See [LICENSE](LICENSE).
