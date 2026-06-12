"""Run complete project pipeline and print final payload."""

from __future__ import annotations

import asyncio
import json

from multi_agent_system.config import get_settings
from multi_agent_system.logging_utils import configure_logging
from multi_agent_system.pipeline import run_all


async def _main() -> None:
    settings = get_settings()
    configure_logging()
    payload = await run_all(settings)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
