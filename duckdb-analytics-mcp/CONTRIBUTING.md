# Contributing

## Setup

```bash
uv python pin 3.12.10
uv venv --python 3.12.10 --allow-existing
uv sync --dev
```

## Development Workflow

1. Create a branch from `main`.
2. Implement focused changes with tests.
3. Run full checks before opening a PR:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run pytest -q -m e2e
```

## Pull Request Expectations

- Explain intent and risk.
- Include test evidence.
- Keep changes scoped; avoid unrelated refactors.
- Keep secrets out of code and history.

## Reporting Bugs

Open an issue with:

- environment details
- reproduction steps
- expected vs actual behavior
- logs or stack traces (sanitized)
