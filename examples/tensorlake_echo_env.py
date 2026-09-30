#!/usr/bin/env python3
"""Hello-world example running the Echo environment on Tensorlake.

Boots the echo-env server inside a Tensorlake sandbox via ``TensorlakeProvider``,
then talks to it through ``EchoEnv`` over the sandbox's public URL.

Register the sandbox image first, from the public echo-env image:

    tl sbx image import ghcr.io/huggingface/openenv-echo-env:latest --registered-name echo-env

Usage:
    PYTHONPATH=src:envs uv run python examples/tensorlake_echo_env.py [image-name]

Requires:
    TENSORLAKE_API_KEY environment variable.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from echo_env import EchoEnv
from openenv.core.containers.runtime.tensorlake_provider import TensorlakeProvider

logger = logging.getLogger(__name__)


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.environ.get("TENSORLAKE_API_KEY"):
        raise SystemExit("Set TENSORLAKE_API_KEY to create the sandbox.")

    image = sys.argv[1] if len(sys.argv) > 1 else "echo-env"
    # public=True: anyone with the sandbox URL can reach the server while it runs.
    provider = TensorlakeProvider(image=image, public=True)

    logger.info("Starting Tensorlake sandbox from image %s...", image)
    base_url = await asyncio.to_thread(provider.start_container)
    try:
        await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        logger.info("Server ready")

        async with EchoEnv(base_url=base_url) as env:
            await env.reset()
            tools = await env.list_tools()
            logger.info("Available tools: %s", [t.name for t in tools])
            echoed = await env.call_tool("echo_message", message="Hello, World!")
            logger.info("echo_message -> %s", echoed)
    finally:
        logger.info("Stopping sandbox...")
        await asyncio.to_thread(provider.stop_container)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
