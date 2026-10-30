"""Conditional W&B Sandbox example: requires verified HTTPS ingress.

The cwsandbox v0.23.3 documented bare service address is unsupported. This
example fails closed and cleans up when the backend does not return an explicit
HTTPS URL. Setting url_scheme or naming a port "https" does not configure TLS.
Confirm runner TLS support before provisioning; this is not a verified hosted
W&B end-to-end example. Public ingress below is an explicit opt-in.

Usage:
    wandb login
    # or: export WANDB_API_KEY=...

    PYTHONPATH=src uv run --extra wandb \
      --with "git+https://huggingface.co/spaces/openenv/echo_env" \
      python examples/wandb_sandbox_echo_env.py
"""

from __future__ import annotations

import asyncio

from echo_env import EchoEnv  # type: ignore[import-not-found]
from openenv.core.containers.runtime.wandb_sandbox_provider import (  # type: ignore[import-not-found]
    WandbSandboxProvider,
)


ECHO_ENV_IMAGE = "registry.hf.space/openenv-echo-env:latest"
SERVER_CMD = (
    "cd /app/env && /app/.venv/bin/python -m uvicorn "
    "server.app:app --host 0.0.0.0 --port 8000"
)


async def main() -> None:
    WandbSandboxProvider.preflight()
    provider = WandbSandboxProvider(
        ingress_mode="public",
        egress_mode="internet",
        max_lifetime_seconds=3600,
        max_timeout_seconds=300,
        tags=["openenv-wandb-echo-example"],
    )

    env = None
    try:
        env = await EchoEnv.from_docker_image(
            ECHO_ENV_IMAGE,
            provider=provider,
            cmd=SERVER_CMD,
        )
    except Exception:
        provider.stop_container()
        raise

    async with env:
        await env.reset()
        tools = await env.list_tools()
        print(f"tools={[tool.name for tool in tools]}")

        result = await env.call_tool(
            "echo_message",
            message="hello from W&B Sandboxes",
        )
        print(f"echo_message={result}")

    print("cleanup=sandbox stopped and deleted")


if __name__ == "__main__":
    asyncio.run(main())
