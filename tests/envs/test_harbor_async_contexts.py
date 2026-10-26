# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry points that are reached from inside a running event loop.

`asyncio.run` raises `RuntimeError: asyncio.run() cannot be called from a running event loop`, so
any code path that a server can reach has to use `run_async_safely`. This has now bitten three
separate places (the client's MCP calls, the `run_rollout` tool handler, and registry dataset
resolution), and each time it worked in a script and failed under the server, which is the worst
place to find out.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("openenv.harbor.tasks")


def sources_reachable_from_a_server() -> dict[str, str]:
    """Source of the functions a request can reach, where a loop is already running."""
    from openenv.harbor import environment, tasks
    from openenv.harbor.client import HarborEnv

    return {
        "environment._run_rollout": inspect.getsource(
            environment.HarborEnvironment._run_rollout
        ),
        "tasks._registry_task_dirs": inspect.getsource(tasks._registry_task_dirs),
        "client._call": inspect.getsource(HarborEnv._call),
    }


@pytest.mark.parametrize("name", sorted(sources_reachable_from_a_server()))
def test_no_bare_asyncio_run_on_a_server_reachable_path(name):
    """`asyncio.run` here is a runtime error under ASGI, not a style preference."""
    source = sources_reachable_from_a_server()[name]
    assert "asyncio.run(" not in source, (
        f"{name} calls asyncio.run, which raises when a loop is already running. "
        "Use openenv.core.utils.run_async_safely."
    )


def test_run_async_safely_works_with_a_loop_already_running():
    """The property the helper exists for, exercised the way a server would hit it."""
    from openenv.core.utils import run_async_safely

    async def inner() -> str:
        await asyncio.sleep(0)
        return "ok"

    async def outer() -> str:
        # A loop is running right now, which is exactly when asyncio.run would raise.
        return run_async_safely(inner())

    assert asyncio.run(outer()) == "ok"


def test_bare_asyncio_run_would_have_failed_here():
    """Pins the failure mode, so the test above cannot be mistaken for a tautology."""

    async def inner() -> str:
        return "ok"

    async def outer() -> str:
        return asyncio.run(inner())

    with pytest.raises(
        RuntimeError, match="cannot be called from a running event loop"
    ):
        asyncio.run(outer())


# --- port forwarding --------------------------------------------------------
def test_cloudflare_quick_forward_uses_the_tunnel_subcommand():
    """`cloudflared forward` is an alias for `cloudflared access`, a different feature entirely.

    Invoked that way the process prints `Incorrect Usage. flag provided but not defined: -url` and
    exits without ever emitting a *.trycloudflare.com URL, so `--expose cloudflare` failed at
    startup every time anyone selected it. The quick tunnel is `cloudflared tunnel --url`.
    """
    forwarding = pytest.importorskip("openenv.core.harness.capture.forwarding")

    recorded: list[list[str]] = []

    class _Proc:
        stdout = None

        def poll(self):
            return None

        def terminate(self):
            pass

    forwarder = forwarding.CloudflareForwarder()
    forwarder.preflight = lambda *_a, **_k: None

    def fake_popen(cmd, **_kwargs):
        recorded.append(cmd)
        raise RuntimeError("stop here; the command line is what matters")

    import subprocess

    original = subprocess.Popen
    subprocess.Popen = fake_popen
    try:
        with pytest.raises(Exception):
            forwarder.start(8100)
    finally:
        subprocess.Popen = original

    assert recorded, "start() never built a command"
    cmd = recorded[0]
    assert "forward" not in cmd, "`forward` is cloudflared access, not a tunnel"
    assert cmd[1] == "tunnel" and "--url" in cmd
