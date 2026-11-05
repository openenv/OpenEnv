#!/usr/bin/env python3
"""Hello-world example running the Echo environment on Kubernetes Agent Sandbox.

Claims a sandbox from a pre-created SandboxWarmPool via ``AgentSandboxProvider``
and talks to the echo-env server through the provider's loopback proxy URL.

Cluster prerequisites (kubectl):
  1. Install the agent-sandbox controller and deploy the sandbox-router:
     https://github.com/kubernetes-sigs/agent-sandbox#installation
  2. Apply a SandboxTemplate + SandboxWarmPool whose template runs the OpenEnv
     echo image (CMD serves OpenEnv on port 8000). See
     docs/source/guides/runtime-providers.md#agentsandboxprovider for the YAML.
  3. A Gateway API resource routing to the sandbox-router, or its URL.

Usage:
    pip install "openenv[agent-sandbox]"   # requires Python >= 3.11
    PYTHONPATH=src:envs uv run python examples/agent_sandbox_echo_env.py \
        <warmpool-name> <gateway-name>

For a local Kind cluster whose Gateway serves plaintext http, set
ALLOW_INSECURE=1.
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
from openenv.core.containers.runtime.agent_sandbox_provider import AgentSandboxProvider

logger = logging.getLogger(__name__)


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        raise SystemExit("usage: agent_sandbox_echo_env.py <warmpool> <gateway-name>")
    warmpool, gateway_name = sys.argv[1], sys.argv[2]

    provider = AgentSandboxProvider(
        warmpool=warmpool,
        gateway_name=gateway_name,
        allow_insecure=os.environ.get("ALLOW_INSECURE") == "1",
    )

    logger.info("Claiming a sandbox from warm pool %s...", warmpool)
    # start_container is inside the try: a failed provisioning can still
    # leave a claim behind that stop_container() deletes by name.
    try:
        base_url = await asyncio.to_thread(provider.start_container)
        await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        logger.info("Server ready")

        async with EchoEnv(base_url=base_url) as env:
            await env.reset()
            tools = await env.list_tools()
            logger.info("Available tools: %s", [t.name for t in tools])
            echoed = await env.call_tool("echo_message", message="Hello, World!")
            logger.info("echo_message -> %s", echoed)
    finally:
        logger.info("Terminating sandbox...")
        await asyncio.to_thread(provider.stop_container)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
