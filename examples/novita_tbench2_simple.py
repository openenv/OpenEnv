#!/usr/bin/env python3
"""Minimal example for the TB2 OpenEnv runner (Novita sandbox mode).

Usage:
    PYTHONPATH=src:envs uv run python examples/novita_tbench2_simple.py

Requires:
    NOVITA_API_KEY environment variable (and optionally NOVITA_DOMAIN).
"""

import asyncio
import os

from openenv.core.containers.runtime.novita_provider import NovitaSandboxProvider
from tbench2_env import Tbench2Action, Tbench2Env


async def main() -> int:
    tasks_dir = os.environ.get("TB2_TASKS_DIR")
    if not tasks_dir:
        print("TB2_TASKS_DIR not set. TB2 repo will be downloaded.")

    task_id = os.environ.get("TB2_TASK_ID", "headless-terminal")

    image = NovitaSandboxProvider.image_from_dockerfile(
        "envs/tbench2_env/server/Dockerfile",
    )
    provider = NovitaSandboxProvider()
    # First start builds a Novita template from the Dockerfile, which can take a
    # few minutes; later starts reuse the cached template.
    # Keep wait_for_ready inside try/finally so a readiness timeout still stops
    # the sandbox (otherwise it lives until Novita's hard lifetime).
    base_url = provider.start_container(image=image)
    try:
        provider.wait_for_ready(base_url, timeout_s=300)
        async with Tbench2Env(base_url=base_url, provider=provider) as env:
            result = await env.reset(task_id=task_id)
            print("Instruction head:")
            print(result.observation.instruction[:200])

            result = await env.step(Tbench2Action(action_type="exec", command="ls -la"))
            print("Command output:")
            print(result.observation.output)

            result = await env.step(
                Tbench2Action(
                    action_type="exec", command="curl -v http://127.0.0.1:8000"
                )
            )
            print("Command output:")
            print(result.observation.output)
    finally:
        provider.stop_container()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
