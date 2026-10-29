#!/usr/bin/env python3
"""Hello-world example running the Echo environment on Kubernetes.

Boots the echo-env server as one Pod and one Service via ``KubernetesProvider``,
then talks to it through ``EchoEnv`` (an MCP tool client).

Echo is the smallest environment in the repo, so it is the quickest way to
check that the Kubernetes provider works end to end.

The process must run inside the cluster. ``start_container`` returns
``http://{name}.{namespace}:8000``, and that name only resolves there. A laptop
can create the Pod, but it cannot open that address.

Usage:
    PYTHONPATH=src:envs uv run python examples/kubernetes_echo_env.py

Requires:
    pip install openenv[kubernetes]
    An existing namespace (this example does not create one).
    Permission to create and delete Pods and Services in that namespace.
    An image the cluster can already run. A locally loaded Kind image should
    use a tag other than ``latest``, or Kubernetes will try to pull it again.

    kubectl create namespace openenv
    kind load docker-image echo-env:test --name <cluster>
    # then run this script from a Pod in namespace openenv

Optional environment variables:
    OPENENV_NAMESPACE   existing namespace (default: openenv)
    OPENENV_IMAGE       container image (default: echo-env:test)
    OPENENV_CPU         cpu request and limit, for example 1
    OPENENV_MEMORY      memory request and limit, for example 1Gi
    OPENENV_LABEL       one extra label, key=value
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from echo_env import EchoEnv
from openenv.core.containers.runtime.kubernetes_provider import KubernetesProvider

logger = logging.getLogger(__name__)


def _start_kwargs() -> dict:
    """Read the optional resource and label settings."""
    kwargs: dict = {}
    requests: dict[str, str] = {}
    if cpu := os.environ.get("OPENENV_CPU"):
        requests["cpu"] = cpu
    if memory := os.environ.get("OPENENV_MEMORY"):
        requests["memory"] = memory
    if requests:
        kwargs["resources"] = {"requests": requests, "limits": dict(requests)}

    label = os.environ.get("OPENENV_LABEL")
    if label:
        key, separator, value = label.partition("=")
        if not separator or not key:
            raise SystemExit("OPENENV_LABEL must be key=value.")
        kwargs["labels"] = {key: value}
    return kwargs


async def _interact(base_url: str) -> None:
    """Exercise the Echo MCP tools over the in-cluster Service.

    Args:
        base_url (`str`):
            Service URL returned by ``start_container``.
    """
    async with EchoEnv(base_url=base_url) as env:
        await env.reset()

        tools = await env.list_tools()
        logger.info("Available tools: %s", [t.name for t in tools])

        echoed = await env.call_tool("echo_message", message="Hello, World!")
        logger.info("echo_message -> %s", echoed)
        if echoed != "Hello, World!":
            raise RuntimeError(f"echo_message returned {echoed!r}")

        with_length = await env.call_tool("echo_with_length", message="Hello, World!")
        logger.info("echo_with_length -> %s", with_length)


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    namespace = os.environ.get("OPENENV_NAMESPACE", "openenv")
    image = os.environ.get("OPENENV_IMAGE", "echo-env:test")
    provider = KubernetesProvider(namespace=namespace)

    logger.info("Starting echo environment in namespace %s", namespace)
    base_url = await asyncio.to_thread(
        provider.start_container, image, **_start_kwargs()
    )
    try:
        await asyncio.to_thread(provider.wait_for_ready, base_url, 180)
        logger.info("Server ready.")
        await _interact(base_url)
    finally:
        logger.info("Stopping Pod and Service...")
        await asyncio.to_thread(provider.stop_container)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
