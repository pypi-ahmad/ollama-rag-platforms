FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
COPY data ./data
COPY .env.example ./

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "duckdb-analytics-mcp", "run", "--transport", "streamable-http"]
