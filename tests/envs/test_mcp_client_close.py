# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`close()` on an MCP client must actually close it.

`EnvClient.close` is synchronous and dispatches to `_close_async` via `_dispatch`, which returns an
awaitable in async code and a result in sync code — so one definition serves `client.close()` and
`await client.close()` both. `MCPClientBase` used to override `close` itself with an `async def`, which
silently broke every synchronous caller: the call built a coroutine, dropped it un-awaited, and
returned as if it had worked.

Nothing errored. The websocket stayed open, so the server never reached its `_destroy_session`
cleanup, and sessions accumulated until `max_concurrent_envs` was exhausted. Measured against a live
server with a cap of 16: exactly 16 of 20 connect-and-close cycles succeeded and the rest failed with
`ConnectionClosedOK`, which reads as a network fault rather than a leak. In a training run, where every
rollout takes a session, that is a wall the run hits partway through.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

mcp_client = pytest.importorskip("openenv.core.mcp_client")
env_client = pytest.importorskip("openenv.core.env_client")


def test_close_is_not_an_unawaited_coroutine():
    """The regression itself: a sync `close()` that returns a coroutine never closes anything."""
    assert not inspect.iscoroutinefunction(mcp_client.MCPClientBase.close), (
        "MCPClientBase.close must stay the inherited SYNC dispatcher; overriding it with an "
        "async def makes client.close() a no-op for every synchronous caller"
    )


def test_the_async_work_is_still_there_under_the_right_name():
    """The cleanup did not disappear — it moved to the hook `close` dispatches to."""
    assert inspect.iscoroutinefunction(mcp_client.MCPClientBase._close_async)


def test_close_is_inherited_from_envclient():
    """Which is what makes `_dispatch` — and therefore both call styles — apply."""
    assert mcp_client.MCPClientBase.close is env_client.EnvClient.close


def test_the_mcp_override_runs_and_then_calls_up(monkeypatch):
    """Session teardown must happen AND the parent's socket teardown must still run.

    Dropping either half leaks: skipping the MCP part orphans the server-side session, skipping the
    parent leaves the websocket open, and both present as capacity exhaustion later.
    """
    order: list[str] = []

    class Fake(mcp_client.MCPClientBase):
        def __init__(
            self,
        ):  # deliberately skips real construction; only close is under test
            self._production_session_id = None
            self._http_client = None

        async def _disconnect_async(self):
            order.append("disconnect")

    async def parent_close(self):
        order.append("parent")

    monkeypatch.setattr(
        env_client.EnvClient, "_close_async", parent_close, raising=True
    )
    client = Fake()
    asyncio.run(client._close_async())
    assert order == ["disconnect", "parent"], (
        "the parent's teardown must still be reached"
    )
