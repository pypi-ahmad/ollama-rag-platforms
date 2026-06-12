"""Convenience script to run end-to-end index + telemetry + evaluation."""

from __future__ import annotations

import asyncio
import json

from rag_telemetry_evals.config import get_settings
from rag_telemetry_evals.logging_utils import configure_logging
from rag_telemetry_evals.pipeline import run_all


async def _main() -> None:
    settings = get_settings()
    configure_logging()
    payload = await run_all(settings)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
