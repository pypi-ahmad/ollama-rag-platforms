"""Convenience script to run end-to-end index + evaluation."""

from __future__ import annotations

import asyncio
import json

from offline_ollama_rag.config import get_settings
from offline_ollama_rag.logging_utils import configure_logging
from offline_ollama_rag.pipeline import run_all


async def _main() -> None:
    settings = get_settings()
    configure_logging()
    payload = await run_all(settings)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
