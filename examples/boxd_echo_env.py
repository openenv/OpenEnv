#!/usr/bin/env python3
"""Hello-world example running the Echo environment on boxd.

Creates a boxd machine via ``BoxdProvider``, runs the echo-env image in it with
Docker, then talks to it through ``EchoEnv`` over the machine's HTTPS URL
(https/wss). The machine is deleted when the provider closes.

Usage:
    pip install "openenv[boxd]"
    export BOXD_API_KEY=...   (https://boxd.sh?ref=OpenEnv)
    PYTHONPATH=src:envs uv run python examples/boxd_echo_env.py

Optional environment variables:
    OPENENV_IMAGE        registry image, or "dockerfile" to build
                         envs/echo_env/server/Dockerfile in the machine
                         (default: ghcr.io/huggingface/openenv-echo-env:latest)
    BOXD_REGION          "fra" or "nyc" (read by the boxd SDK)
    OPENENV_NAME_PREFIX  machine name prefix (default: openenv)
"""

import asyncio
import logging
import os

from echo_env import EchoEnv
from openenv.core.containers.runtime.boxd_provider import BoxdProvider

logger = logging.getLogger(__name__)


async def _interact(base_url: str) -> None:
    """Run the async WebSocket interaction against the running machine."""
    async with EchoEnv(base_url=base_url) as env:
        await env.reset()

        tools = await env.list_tools()
        logger.info("Available tools: %s", [t.name for t in tools])

        echoed = await env.call_tool("echo_message", message="Hello, World!")
        logger.info("echo_message -> %s", echoed)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    image = os.environ.get(
        "OPENENV_IMAGE", "ghcr.io/huggingface/openenv-echo-env:latest"
    )
    if image == "dockerfile":
        image = BoxdProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")

    with BoxdProvider(
        name_prefix=os.environ.get("OPENENV_NAME_PREFIX", "openenv"),
    ) as provider:
        logger.info("Starting boxd machine...")
        base_url = provider.start_container(image)
        logger.info("Machine %s is up", provider.machine_id)
        provider.wait_for_ready(base_url, timeout_s=180)
        logger.info("Server ready.")
        asyncio.run(_interact(base_url))
        logger.info("Deleting machine...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
