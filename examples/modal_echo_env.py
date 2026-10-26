#!/usr/bin/env python3
"""Hello-world example running the Echo environment on Modal.

Boots the echo-env server inside a Modal sandbox via ``ModalProvider``, then
talks to it through ``EchoEnv`` over the encrypted Modal tunnel (https/wss).

Usage:
    pip install "modal>=1.4"
    modal setup  (one-time auth)
    PYTHONPATH=src:envs uv run python examples/modal_echo_env.py

Requires:
    A configured Modal account/token (see https://modal.com/docs/guide).
"""

import asyncio
import logging

from echo_env import EchoEnv
from openenv.core.containers.runtime.modal_provider import ModalProvider

logger = logging.getLogger(__name__)


async def _interact(base_url: str) -> None:
    """Run the async WebSocket interaction against the running sandbox."""
    async with EchoEnv(base_url=base_url) as env:
        await env.reset()

        tools = await env.list_tools()
        logger.info("Available tools: %s", [t.name for t in tools])

        echoed = await env.call_tool("echo_message", message="Hello, World!")
        logger.info("echo_message -> %s", echoed)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    image = ModalProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")

    with ModalProvider(app_name="openenv-echo") as provider:
        logger.info("Starting Modal sandbox (building image on first run)...")
        base_url = provider.start_container(image)
        logger.info("Sandbox up - waiting for server...")
        provider.wait_for_ready(base_url, timeout_s=180)
        logger.info("Server ready.")
        asyncio.run(_interact(base_url))
        logger.info("Stopping sandbox...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
