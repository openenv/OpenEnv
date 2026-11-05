# SPDX-License-Identifier: BSD-3-Clause

"""Loopback MCP bridge exposing environment tools to a harness (RFC 005)."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from .adapter import HarnessError


async def _source_tools(mcp_server: Any) -> dict[str, Any]:
    """Return {name: FastMCP tool}.

    Async twin of [`~openenv.core.env_server.mcp_environment.get_server_tools`],
    which wraps the same calls in `run_async_safely`; awaiting directly avoids
    spawning a thread when we are already on a loop.
    """
    if hasattr(mcp_server, "get_tools"):
        result = await mcp_server.get_tools()
        if isinstance(result, dict):
            return result
    if hasattr(mcp_server, "list_tools"):
        return {tool.name: tool for tool in await mcp_server.list_tools()}
    return {}


async def build_bridge_server(mcp_server: Any, renames: dict[str, str]) -> Any:
    """
    Return a FastMCP server exposing the env's tools under their injected names.

    Tool-name conflict resolution can rename an environment tool before it is
    injected into a harness (`read_file` -> `env_read_file`). The harness then
    calls the new name, so the bridge has to answer to it: serving the source
    server unchanged would advertise a name that does not resolve.

    Args:
        mcp_server (`FastMCP`):
            The environment's own tool server.
        renames (`dict[str, str]`):
            Mapping of injected name to source name. Empty means no renames.

    Returns:
        `FastMCP`: `mcp_server` itself when there is nothing to rename,
        otherwise a view of it whose tools carry the injected names.
    """
    if not renames:
        return mcp_server

    from fastmcp import Client
    from fastmcp.server.dependencies import get_context
    from fastmcp.server.providers.proxy import FastMCPProxy, StatefulProxyClient
    from fastmcp.server.transforms import ToolTransform
    from fastmcp.tools.tool_transform import ToolTransformConfig
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS

    source_tools = await _source_tools(mcp_server)
    missing = sorted(set(renames.values()) - set(source_tools))
    if missing:
        raise HarnessError(
            "Cannot rename tools that the MCP server does not expose: "
            + ", ".join(missing)
        )

    @asynccontextmanager
    async def lifespan(server):
        # Preserve source resources across harness MCP reconnects until this
        # bridge stops, just as when serving the source server directly.
        async with Client(mcp_server):
            yield {}

    # Keep one backend client per front connection, and mirror its protocol era.
    # StatefulProxyClient defaults to legacy, which cannot carry sampling guards.
    client = StatefulProxyClient(mcp_server)

    def client_factory():
        backend = client.new_stateful()
        if not backend.is_connected():
            version = get_context().request_context.protocol_version
            backend.mode = version if version in MODERN_PROTOCOL_VERSIONS else "legacy"
        return backend

    view = FastMCPProxy(
        name=f"{getattr(mcp_server, 'name', 'openenv')}-harness-view",
        client_factory=client_factory,
        lifespan=lifespan,
    )
    view.add_transform(
        ToolTransform(
            {
                source: ToolTransformConfig(name=served)
                for served, source in renames.items()
            }
        )
    )
    return view


class HarnessMCPBridge:
    """
    Serve an in-process FastMCP server over loopback HTTP for harness use.

    The bridge runs the environment's FastMCP tool server as a
    streamable-HTTP ASGI app on `127.0.0.1` with an ephemeral port, inside a
    daemon thread with its own event loop. It carries only the tool surface:
    the harness subprocess can reach the environment's domain tools but never
    OpenEnv's orchestration API (`reset`/`step`/`state`), which lives on a
    different server entirely. This makes the RFC 005 security boundary
    structural rather than filter-based.

    Args:
        mcp_server (`FastMCP`):
            The environment's FastMCP server to expose.
        host (`str`, *optional*, defaults to `"127.0.0.1"`):
            Interface to bind. Keep this loopback-only.

    Examples:

    ```python
    bridge = HarnessMCPBridge(env.mcp_server)
    url = bridge.start()
    # pass url to the harness adapter's inject_tools()
    bridge.stop()
    ```
    """

    def __init__(self, mcp_server: Any, host: str = "127.0.0.1"):
        self._mcp_server = mcp_server
        self._host = host
        self._url: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[Any] = None
        self._startup_error: Optional[BaseException] = None

    @property
    def url(self) -> Optional[str]:
        """The bridge's MCP endpoint URL, or `None` when not running."""
        return self._url

    def start(self, timeout_s: float = 10.0) -> str:
        """
        Start serving and return the MCP endpoint URL.

        Idempotent: returns the existing URL if already running.

        Args:
            timeout_s (`float`, *optional*, defaults to `10.0`):
                Maximum time to wait for the server to come up.

        Returns:
            `str` URL of the MCP endpoint, e.g. `"http://127.0.0.1:54321/mcp"`.

        Raises:
            [`~openenv.core.harness.adapter.HarnessError`]:
                If the server fails to start within the timeout.
        """
        if self._thread is not None and self._thread.is_alive():
            if self._uvicorn_server.should_exit:
                raise HarnessError("MCP bridge is still stopping")
            assert self._url is not None
            return self._url

        import uvicorn

        app = self._mcp_server.http_app()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, 0))
        port = sock.getsockname()[1]

        config = uvicorn.Config(app, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        self._startup_error = None

        def _serve() -> None:
            try:
                server.run(sockets=[sock])
            except BaseException as exc:  # surfaced to start() below
                self._startup_error = exc
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        self._thread = thread

        deadline = time.monotonic() + timeout_s
        while not server.started:
            if self._startup_error is not None or not thread.is_alive():
                self._teardown()
                raise HarnessError(f"MCP bridge failed to start: {self._startup_error}")
            if time.monotonic() > deadline:
                self.stop()
                raise HarnessError(f"MCP bridge did not start within {timeout_s}s")
            time.sleep(0.01)

        self._url = f"http://{self._host}:{port}/mcp"
        return self._url

    def stop(self, timeout_s: float = 5.0) -> None:
        """
        Stop the bridge server.

        Idempotent: safe to call when the bridge was never started.

        Args:
            timeout_s (`float`, *optional*, defaults to `5.0`):
                Maximum time to wait for the server thread to exit.

        Raises:
            [`~openenv.core.harness.adapter.HarnessError`]:
                If shutdown times out. The server remains tracked so callers
                can retry `stop()`; `start()` rejects it while still stopping.
        """
        server = self._uvicorn_server
        thread = self._thread
        deadline = time.monotonic() + max(timeout_s, 0.0)
        if server is not None:
            # Cancel lingering HTTP streams before the thread join expires.
            # Reserve time for ASGI lifespan cleanup and forced shutdown.
            server.config.timeout_graceful_shutdown = max(timeout_s, 0.0) / 4
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(timeout_s, 0.0) / 2)
            if thread.is_alive() and server is not None:
                server.force_exit = True
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                # Keep ownership so callers can retry, and start() cannot
                # replace a server that still holds resources.
                raise HarnessError(f"MCP bridge did not stop within {timeout_s}s")
        self._teardown()

    def _teardown(self) -> None:
        self._url = None
        self._thread = None
        self._uvicorn_server = None


__all__ = ["build_bridge_server", "HarnessMCPBridge"]
