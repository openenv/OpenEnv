#!/usr/bin/env python3
"""Hello-world example running the Echo environment on Novita.

Boots the echo-env server inside a Novita sandbox via ``NovitaSandboxProvider``,
then talks to it through ``EchoEnv`` (an MCP tool client) over the sandbox's
exposed host.

Echo is the smallest environment in the repo — no task data, no external
services, no model calls — so it is the quickest way to check that a sandbox
provider works end to end. For the same reason it is the right first target
when wiring up a new provider.

Usage:
    PYTHONPATH=src:envs uv run python examples/novita_echo_env.py

Requires:
    NOVITA_API_KEY environment variable (and optionally NOVITA_DOMAIN).
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
from openenv.core.containers.runtime.novita_provider import NovitaSandboxProvider

DOCKERFILE = "envs/echo_env/server/Dockerfile"

logger = logging.getLogger(__name__)


async def _interact(base_url: str) -> None:
    """Exercise the Echo MCP tools over a sandbox-backed session.

    Args:
        base_url (`str`):
            Sandbox URL running the echo server.
    """
    # The client is given no provider: the sandbox outlives this session and is
    # released once, in main(), after the interaction finishes.
    async with EchoEnv(base_url=base_url) as env:
        await env.reset()

        tools = await env.list_tools()
        logger.info("Available tools: %s", [t.name for t in tools])

        echoed = await env.call_tool("echo_message", message="Hello, World!")
        logger.info("echo_message -> %s", echoed)

        with_length = await env.call_tool("echo_with_length", message="Hello, World!")
        logger.info("echo_with_length -> %s", with_length)


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.environ.get("NOVITA_API_KEY"):
        raise SystemExit("Set NOVITA_API_KEY to create the sandbox.")

    image = NovitaSandboxProvider.image_from_dockerfile(str(_REPO_ROOT / DOCKERFILE))
    provider = NovitaSandboxProvider()

    # The first start builds a Novita template from the Dockerfile, which can
    # take minutes; later starts reuse the cached template.
    logger.info("Starting Novita sandbox (first run builds the template)...")
    base_url = await asyncio.to_thread(provider.start_container, image)
    try:
        await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        logger.info("Server ready at %s", base_url)

        await _interact(base_url)
    finally:
        logger.info("Stopping sandbox...")
        await asyncio.to_thread(provider.stop_container)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
