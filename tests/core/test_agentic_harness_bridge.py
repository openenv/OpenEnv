# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the loopback MCP bridge (real HTTP server)."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import anyio
import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from mcp.types import CreateMessageResult, TextContent
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessConfig,
    HarnessEnvironment,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HarnessMCPBridge,
)


def make_mcp() -> FastMCP:
    mcp = FastMCP("domain")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return mcp


class RecordingAdapter(AgenticHarnessAdapter):
    BUILTIN_TOOL_NAMES = frozenset({"read_file"})

    def __init__(self):
        super().__init__(HarnessConfig(name="fake", command=["fake"]))
        self.bridge_url: Optional[str] = "unset"
        self.alive = False

    async def start(self, working_directory: str) -> None:
        self.alive = True

    async def stop(self) -> None:
        self.alive = False

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        self.bridge_url = bridge_url

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        yield HarnessEvent(type=HarnessEventType.TURN_COMPLETE, data={"response": "ok"})

    async def is_alive(self) -> bool:
        return self.alive


class TestBridgeStandalone:
    async def test_stop_closes_active_http_stream(self):
        from types import SimpleNamespace

        import httpx
        from starlette.applications import Starlette
        from starlette.responses import StreamingResponse
        from starlette.routing import Route

        finished = threading.Event()

        async def stream(request):
            async def chunks():
                try:
                    yield b"ready\n"
                    await asyncio.Event().wait()
                finally:
                    finished.set()

            return StreamingResponse(chunks())

        app = Starlette(routes=[Route("/mcp", stream)])
        bridge = HarnessMCPBridge(SimpleNamespace(http_app=lambda: app))
        url = bridge.start()
        thread = bridge._thread
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url) as response:
                    assert await anext(response.aiter_lines()) == "ready"
                    started = time.monotonic()
                    await asyncio.to_thread(bridge.stop, 2.0)
                    assert time.monotonic() - started < 3.0
                    assert not thread.is_alive()
                    assert finished.is_set()
                    assert bridge.url is None
            bridge.start()
            assert bridge._thread is not thread
        finally:
            await asyncio.to_thread(bridge.stop)

    def test_failed_stop_keeps_server_until_thread_exits(self):
        from types import SimpleNamespace

        release = threading.Event()
        thread = threading.Thread(target=release.wait, daemon=True)
        server = SimpleNamespace(
            should_exit=False, force_exit=False, config=SimpleNamespace()
        )
        bridge = HarnessMCPBridge(make_mcp())
        bridge._thread = thread
        bridge._uvicorn_server = server
        bridge._url = "http://127.0.0.1:9/mcp"
        thread.start()
        try:
            with pytest.raises(HarnessError, match="stop"):
                bridge.stop(timeout_s=0.05)
            assert bridge._thread is thread
            assert bridge._uvicorn_server is server
            assert server.force_exit
            with pytest.raises(HarnessError, match="stopping"):
                bridge.start()
        finally:
            release.set()
            thread.join(timeout=1.0)
            bridge.stop()
        assert bridge.url is None

    async def test_serves_tools_over_http(self):
        bridge = HarnessMCPBridge(make_mcp())
        url = bridge.start()
        try:
            assert url.startswith("http://127.0.0.1:")
            assert url.endswith("/mcp")
            async with Client(url) as client:
                tools = await client.list_tools()
                assert [tool.name for tool in tools] == ["add"]
                result = await client.call_tool("add", {"a": 2, "b": 3})
                assert result.content[0].text == "5"
        finally:
            bridge.stop()

    def test_two_bridges_get_distinct_ports(self):
        bridge_a = HarnessMCPBridge(make_mcp())
        bridge_b = HarnessMCPBridge(make_mcp())
        try:
            url_a = bridge_a.start()
            url_b = bridge_b.start()
            assert url_a != url_b
        finally:
            bridge_a.stop()
            bridge_b.stop()

    def test_stop_is_idempotent(self):
        bridge = HarnessMCPBridge(make_mcp())
        bridge.start()
        bridge.stop()
        bridge.stop()
        assert bridge.url is None

    def test_stop_before_start_is_noop(self):
        bridge = HarnessMCPBridge(make_mcp())
        bridge.stop()
        assert bridge.url is None

    def test_start_is_idempotent_while_running(self):
        bridge = HarnessMCPBridge(make_mcp())
        try:
            assert bridge.start() == bridge.start()
        finally:
            bridge.stop()


class TestBridgeEnvironmentIntegration:
    @pytest.mark.parametrize("cancellation", ["asyncio", "repeated_asyncio", "anyio"])
    async def test_cancelled_reset_waits_for_bridge_start_before_stopping(
        self, monkeypatch, cancellation
    ):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        stop_entered = asyncio.Event()
        stop_release = asyncio.Event()
        stopped = asyncio.Event()
        bridges = []
        server_threads = []
        original_start = HarnessMCPBridge.start

        def delayed_start(bridge):
            bridges.append(bridge)
            entered.set()
            try:
                assert release.wait(timeout=10.0)
                url = original_start(bridge)
                server_threads.append(bridge._thread)
                return url
            finally:
                finished.set()

        monkeypatch.setattr(HarnessMCPBridge, "start", delayed_start)
        adapter = RecordingAdapter()
        original_stop = adapter.stop
        stop_calls = 0

        async def delayed_stop():
            nonlocal stop_calls
            stop_calls += 1
            if stop_calls == 2:
                stop_entered.set()
                await stop_release.wait()
            await original_stop()
            if stop_calls == 2:
                stopped.set()

        monkeypatch.setattr(adapter, "stop", delayed_stop)
        env = HarnessEnvironment(adapter=adapter, mcp=make_mcp())
        cancel_scope = anyio.CancelScope()

        async def reset():
            with cancel_scope:
                await env.reset_async()

        reset_task = asyncio.create_task(reset())
        try:
            assert await asyncio.to_thread(entered.wait, 10.0)
            if cancellation == "anyio":
                cancel_scope.cancel()
            else:
                reset_task.cancel()
            await asyncio.sleep(0)
            if cancellation == "repeated_asyncio":
                reset_task.cancel()

            done, _ = await asyncio.wait([reset_task], timeout=0.05)
            assert not done, "cancelled reset returned before bridge startup finished"
            release.set()
            await asyncio.wait_for(stop_entered.wait(), timeout=10.0)
            if cancellation == "repeated_asyncio":
                reset_task.cancel()
            done, _ = await asyncio.wait([reset_task], timeout=0.05)
            assert not done, "cancelled reset returned before adapter teardown finished"
            stop_release.set()

            results = await asyncio.wait_for(
                asyncio.gather(reset_task, return_exceptions=True), timeout=15.0
            )
            if cancellation == "anyio":
                assert cancel_scope.cancelled_caught
            else:
                assert isinstance(results[0], asyncio.CancelledError)
            assert stopped.is_set()
            assert finished.is_set()
            assert server_threads
            assert not server_threads[0].is_alive()
            assert env._bridge.url is None
            assert adapter.alive is False
            assert env._episode_active is False
        finally:
            release.set()
            stop_release.set()
            if not reset_task.done():
                reset_task.cancel()
            await asyncio.wait_for(
                asyncio.gather(reset_task, return_exceptions=True), timeout=15.0
            )
            await asyncio.to_thread(finished.wait, 10.0)
            # Keep direct references in case a failed reset/close loses the
            # bridge while its startup worker is still running.
            for bridge in bridges:
                await asyncio.to_thread(bridge.stop)
            env.close()
            assert all(not thread.is_alive() for thread in server_threads)

    async def test_reset_passes_live_bridge_url(self):
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=make_mcp())
        try:
            await env.reset_async()
            assert adapter.bridge_url is not None
            async with Client(adapter.bridge_url) as client:
                tools = await client.list_tools()
                assert [tool.name for tool in tools] == ["add"]
        finally:
            env.close()
        assert env._bridge is None

    async def test_no_tools_no_bridge(self):
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=None)
        try:
            await env.reset_async()
            assert adapter.bridge_url is None
        finally:
            env.close()


class TestRenamedToolsAreServed:
    """Conflict resolution renames tools before injection; the bridge must
    answer to the injected names or the harness calls a name that 404s."""

    @staticmethod
    def make_colliding_mcp() -> FastMCP:
        mcp = FastMCP("domain")

        @mcp.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool
        def read_file(path: str) -> str:
            """Collides with the adapter's built-in read_file."""
            return f"contents of {path}"

        return mcp

    async def test_renamed_tool_is_listed_and_callable(self):
        adapter = RecordingAdapter()  # BUILTIN_TOOL_NAMES == {"read_file"}
        mcp = self.make_colliding_mcp()
        async with Client(mcp) as client:
            source_tools = {tool.name: tool for tool in await client.list_tools()}
        env = HarnessEnvironment(adapter=adapter, mcp=mcp)
        try:
            obs = await env.reset_async()
            assert sorted(obs.metadata["injected_tools"]) == ["add", "env_read_file"]

            async with Client(adapter.bridge_url) as client:
                served = {tool.name: tool for tool in await client.list_tools()}
                assert sorted(served) == ["add", "env_read_file"]
                for name, source_name in {
                    "add": "add",
                    "env_read_file": "read_file",
                }.items():
                    assert (
                        served[name].inputSchema
                        == source_tools[source_name].inputSchema
                    )
                    assert (
                        served[name].outputSchema
                        == source_tools[source_name].outputSchema
                    )

                # The renamed tool must actually resolve, not just be listed.
                result = await client.call_tool("env_read_file", {"path": "a.py"})
                assert result.content[0].text == "contents of a.py"
                with pytest.raises(ToolError, match="Unknown tool"):
                    await client.call_tool("read_file", {"path": "a.py"})

                # The un-renamed one is untouched.
                assert (await client.call_tool("add", {"a": 2, "b": 3})).content[
                    0
                ].text == "5"
        finally:
            env.close()
        async with Client(mcp) as client:
            assert sorted(tool.name for tool in await client.list_tools()) == [
                "add",
                "read_file",
            ]

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [("env_read_file", {"path": "secret"}), ("add", {"a": 2, "b": 3})],
    )
    async def test_renamed_bridge_preserves_source_middleware(self, name, arguments):
        calls = []

        class DenyCalls(Middleware):
            async def on_call_tool(self, context, call_next):
                calls.append(context.message.name)
                raise ToolError("blocked by source middleware")

        mcp = self.make_colliding_mcp()
        mcp.add_middleware(DenyCalls())
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=mcp)
        try:
            await env.reset_async()
            async with Client(adapter.bridge_url) as client:
                with pytest.raises(ToolError, match="blocked by source middleware"):
                    await client.call_tool(name, arguments)
            assert calls == ["read_file" if name == "env_read_file" else name]
        finally:
            env.close()

    async def test_renamed_bridge_preserves_source_lifespan_context(self):
        resources = []

        @asynccontextmanager
        async def lifespan(server):
            resource = {"open": True, "calls": 0}
            resources.append(resource)
            try:
                yield {"resource": resource}
            finally:
                resource["open"] = False

        mcp = FastMCP("domain", lifespan=lifespan)

        @mcp.tool
        def read_file(ctx: Context) -> int:
            resource = ctx.lifespan_context["resource"]
            if not resource["open"]:
                raise RuntimeError("resource already closed")
            resource["calls"] += 1
            return resource["calls"]

        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=mcp)
        try:
            await env.reset_async()
            for expected in (1, 2):
                async with Client(adapter.bridge_url) as client:
                    result = await client.call_tool("env_read_file", {})
                    assert result.content[0].text == str(expected)
                assert resources[-1]["open"]
            assert sum(resource["calls"] for resource in resources) == 2
        finally:
            env.close()
        assert resources
        assert all(not resource["open"] for resource in resources)

    async def test_renamed_bridge_forwards_sampling_and_progress(self):
        sampled = []
        progress = []
        mcp = FastMCP("domain")

        @mcp.tool
        async def read_file(path: str, ctx: Context) -> str:
            response = await ctx.sample(path)
            await ctx.report_progress(1, 1, path)
            return response.text

        async def sample(messages, params, context):
            text = messages[0].content.text
            sampled.append(text)
            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=f"sampled {text}"),
                model="test",
            )

        async def record_progress(completed, total, message):
            progress.append((completed, total, message))

        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=mcp)
        try:
            await env.reset_async()
            async with Client(adapter.bridge_url, sampling_handler=sample) as client:
                for path in ("first", "second"):
                    result = await client.call_tool(
                        "env_read_file",
                        {"path": path},
                        progress_handler=record_progress,
                        timeout=5.0,
                    )
                    assert result.content[0].text == f"sampled {path}"
            assert sampled == ["first", "second"]
            assert progress == [(1, 1, "first"), (1, 1, "second")]
        finally:
            env.close()

    async def test_no_renames_serves_the_source_server_directly(self):
        adapter = RecordingAdapter()
        env = HarnessEnvironment(adapter=adapter, mcp=make_mcp())  # only "add"
        try:
            await env.reset_async()
            async with Client(adapter.bridge_url) as client:
                assert [t.name for t in await client.list_tools()] == ["add"]
        finally:
            env.close()
