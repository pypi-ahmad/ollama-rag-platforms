# DuckDB Analytics MCP

Read-only MCP server for querying local datasets (`.csv`, `.parquet`, `.json`, `.jsonl`) through DuckDB with explicit SQL guardrails, pagination, timeouts, and optional bearer-token auth.

## What This Server Provides

- MCP transports: `stdio` and `streamable-http`
- Tool responses in either `markdown` or `json`
- Dataset catalog + schema resources
- Read-only SQL execution over a registered `source` table alias
- Guardrails for query size, statement shape, and banned SQL keywords
- Optional auth with required scope enforcement

## Supported Dataset Formats

Files are discovered recursively under `DATASET_DIR`.

- `.csv`
- `.parquet`
- `.json`
- `.jsonl`

## Setup

```bash
git clone https://github.com/pypi-ahmad/duckdb-analytics-mcp.git
cd duckdb-analytics-mcp
uv python pin 3.12.10
uv sync --dev
cp .env.example .env
```

## Configuration

Runtime is configured through environment variables (loaded from `.env` by default).

| Variable | Default | Description |
|---|---|---|
| `SERVER_NAME` | `duckdb_analytics_mcp` | MCP server name |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`) |
| `HOST` | `127.0.0.1` | Bind host for HTTP transport |
| `PORT` | `8000` | Bind port for HTTP transport |
| `TRANSPORT` | `streamable-http` | Default transport (`stdio` or `streamable-http`) |
| `DATASET_DIR` | `data` | Dataset root; relative paths resolve from project root |
| `DEFAULT_LIMIT` | `25` | Default page size |
| `MAX_LIMIT` | `200` | Upper bound for list/query page size |
| `MAX_QUERY_CHARS` | `4000` | Max SQL length |
| `QUERY_TIMEOUT_SECONDS` | `20` | Timeout for describe/query operations |
| `MAX_SAMPLE_ROWS` | `25` | Max rows returned for dataset samples |
| `DUCKDB_THREADS` | `4` | DuckDB execution threads |
| `DUCKDB_MEMORY_LIMIT` | `1GB` | DuckDB memory limit |
| `STATIC_BEARER_TOKEN` | unset | Enables token verification when auth URLs are also set |
| `AUTH_ISSUER_URL` | unset | Auth issuer URL (required with `STATIC_BEARER_TOKEN`) |
| `AUTH_RESOURCE_SERVER_URL` | unset | Resource server URL (required with `STATIC_BEARER_TOKEN`) |
| `AUTH_REQUIRED_SCOPE` | `analytics.read` | Required access scope |

Auth note:
- If `STATIC_BEARER_TOKEN` is set, both `AUTH_ISSUER_URL` and `AUTH_RESOURCE_SERVER_URL` must also be set.

## Run

### Streamable HTTP

```bash
uv run duckdb-analytics-mcp run --transport streamable-http
```

Endpoint: `http://127.0.0.1:8000/mcp`

### stdio

```bash
uv run duckdb-analytics-mcp run --transport stdio
```

## MCP Client Integration

### Generic stdio server config

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

### Generic streamable-http config

Use base URL:

```text
http://127.0.0.1:8000/mcp
```

## Tool Reference

### `duckdb_analytics_health`

Returns server and catalog health metadata.

Parameters:
- `response_format` (`markdown` or `json`, default `markdown`)

### `duckdb_analytics_list_datasets`

Lists datasets with pagination.

Parameters:
- `limit` (default `25`)
- `offset` (default `0`)
- `response_format` (`markdown` or `json`, default `markdown`)

### `duckdb_analytics_describe_dataset`

Returns dataset metadata, schema, row count, and sample rows.

Parameters:
- `dataset` (relative catalog name, for example `sales/orders.csv`)
- `sample_rows` (default `10`, bounded by `MAX_SAMPLE_ROWS`)
- `response_format` (`markdown` or `json`, default `markdown`)

### `duckdb_analytics_query_dataset`

Runs guarded read-only SQL against the dataset using the table alias `source`.

Parameters:
- `dataset` (relative catalog name)
- `sql` (must start with `SELECT` or `WITH`)
- `limit` (default `25`, bounded by `MAX_LIMIT`)
- `offset` (default `0`)
- `response_format` (`markdown` or `json`, default `markdown`)

### Error Behavior

- `describe`/`query` return tool-level errors for invalid dataset names, SQL guard violations, and timeout conditions.
- In `json` mode, tool errors are returned as `{ "error": "..." }`.
- In `markdown` mode, tool errors are returned as `Error: ...`.

## Resource Reference

- `dataset://catalog` — full dataset catalog
- `dataset://schema/{dataset_name}` — schema + sample rows for one dataset

## SQL Safety Model

`duckdb_analytics_query_dataset` enforces:

- single statement only
- no SQL comments (`--`, `/* ... */`)
- query must start with `SELECT` or `WITH`
- banned keywords (for example `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`, `COPY`, `INSTALL`, `LOAD`)

## CLI Commands

```bash
uv run duckdb-analytics-mcp --help
uv run duckdb-analytics-mcp run --help
uv run duckdb-analytics-mcp doctor
```

`doctor` prints deployment-readiness JSON, including health and dataset counts.

## Docker

```bash
docker build -t duckdb-analytics-mcp:latest .
docker run --rm -p 8000:8000 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e TRANSPORT=streamable-http \
  duckdb-analytics-mcp:latest
```

Or with compose:

```bash
docker compose up --build
```

## Development Verification

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Troubleshooting

- `Dataset '<name>' not found`:
  - Run `duckdb_analytics_list_datasets` first and use an exact dataset name from catalog.
- `Operation timed out after ... seconds`:
  - Simplify query, lower scan volume, or increase `QUERY_TIMEOUT_SECONDS`.
- Empty catalog:
  - Confirm files exist under `DATASET_DIR` and use supported suffixes.
- Auth startup error:
  - If `STATIC_BEARER_TOKEN` is set, also set `AUTH_ISSUER_URL` and `AUTH_RESOURCE_SERVER_URL`.

## Project Structure

```text
src/duckdb_analytics_mcp/
  cli.py
  config.py
  server.py
  service.py
  catalog.py
  security.py
  formatter.py

data/
tests/
notebooks/
```
