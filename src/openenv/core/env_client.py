# SPDX-License-Identifier: BSD-3-Clause

"""
Environment client for persistent sessions.

This module provides a WebSocket-based client that maintains a persistent connection
to an environment server, enabling efficient multi-step interactions without
the overhead of HTTP request/response cycles.

The client is async by default. For synchronous usage, use the `.sync()` method
to get a `SyncEnvClient` wrapper.

Examples:

    Async usage:

    ```python
    async with GenericEnvClient(base_url="ws://localhost:8000") as env:
        result = await env.reset()
        result = await env.step({"code": "print('hello')"})
    ```

    Sync usage via `.sync()` wrapper:

    ```python
    env = GenericEnvClient(base_url="ws://localhost:8000").sync()
    with env:
        result = env.reset()
        result = env.step({"code": "print('hello')"})
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, Callable, Dict, Generic, Optional, Type, TYPE_CHECKING, TypeVar
from urllib.parse import urlsplit

from .client_types import StateT, StepResult
from .containers.runtime import LocalDockerProvider, UVProvider
from .utils import convert_to_ws_url, hf_space_subdomain

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from .containers.runtime import ContainerProvider, RuntimeProvider
    from .sync_client import SyncEnvClient

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
EnvClientT = TypeVar("EnvClientT", bound="EnvClient")
ResultT = TypeVar("ResultT")

_VALID_CLIENT_MODES = ("simulation", "production")


class _AutoAsyncResult(Generic[ResultT]):
    """Awaitable result that can also be resolved by synchronous access."""

    def __init__(
        self,
        client: "EnvClient[Any, Any, Any]",
        coro_factory: Callable[[], Any],
    ):
        self._client = client
        self._coro_factory = coro_factory
        self._resolved = False
        self._result: ResultT | None = None

    def __await__(self):
        async def _await_result():
            self._client._claim_execution_mode("async")
            return await self._coro_factory()

        return _await_result().__await__()

    def _sync_result(self) -> ResultT:
        if not self._resolved:
            self._result = self._client._run_sync(self._coro_factory)
            self._resolved = True
        return self._result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sync_result(), name)

    def __bool__(self) -> bool:
        return bool(self._sync_result())

    def __repr__(self) -> str:
        if self._resolved:
            return repr(self._result)
        return f"<{type(self).__name__} pending>"


class _BootstrapResult(Coroutine, Generic[EnvClientT]):
    """Bootstrap handle returned by the client factory methods.

    Returned by [`~openenv.core.EnvClient.from_docker_image`] and
    [`~openenv.core.EnvClient.from_env`]. Resolving the handle is what actually
    starts the container / Space and connects the WebSocket, so a single factory
    call serves both execution modes:

    - `await handle` connects on the running event loop and returns the connected
      async client (unchanged async behavior).
    - `handle.sync()` connects on the sync background loop and returns a
      `SyncEnvClient`, mirroring the instance-level [`~openenv.core.EnvClient.sync`].

    The handle is a full coroutine (it implements `send` / `throw` / `close`), so
    it is a drop-in for the previous `async def` factories: `asyncio.run(...)`,
    `run_async_safely(...)`, and bare `await` all accept it, in addition to the
    new `.sync()` chain. Bootstrap is lazy — the underlying coroutine (and the
    container/Space start) is created only when the handle is driven or `.sync()`
    is called.
    """

    def __init__(self, bootstrap: Callable[[], EnvClientT]):
        self._bootstrap = bootstrap
        self._coro: Optional[Coroutine[Any, Any, EnvClientT]] = None
        self._used = False

    def _consume(self) -> EnvClientT:
        """Run the bootstrap exactly once; a second resolve is a programming error.

        Mirrors native coroutine semantics (a coroutine cannot be awaited twice)
        so that a handle re-driven via a second `.sync()`, or `await` followed by
        `.sync()`, raises instead of silently starting a second container/Space.
        """
        if self._used:
            raise RuntimeError(
                "This bootstrap handle has already been resolved; call the "
                "factory again to start a new environment."
            )
        self._used = True
        return self._bootstrap()

    async def _resolve_async(self) -> EnvClientT:
        client = self._consume()
        try:
            await client.connect()
        except asyncio.CancelledError:
            # _connect_async releases the provider on ordinary failures, but
            # cancellation bypasses its `except Exception`. No client reaches
            # the caller, so release the provider here before propagating the
            # original cancellation. A repeated cancellation or a stop failure
            # during cleanup must not replace it.
            with suppress(Exception, asyncio.CancelledError):
                await client.close()
            raise
        return client

    def _ensure_coro(self) -> "Coroutine[Any, Any, EnvClientT]":
        if self._coro is None:
            self._coro = self._resolve_async()
        return self._coro

    def __await__(self):
        return self._ensure_coro().__await__()

    def send(self, value: Any) -> Any:
        return self._ensure_coro().send(value)

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        return self._ensure_coro().throw(*args, **kwargs)

    def close(self) -> None:
        if self._coro is not None:
            self._coro.close()

    def sync(self) -> "SyncEnvClient":
        client = self._consume()
        try:
            client._run_sync(client._connect_async)
        except Exception:
            # _consume() already started the provider. On the async path
            # _connect_async releases it, but here the failure may be the sync
            # loop setup itself (before _connect_async runs), so stop the
            # provider directly rather than routing through the broken loop.
            client._stop_provider_best_effort()
            raise
        return client.sync()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} pending (await, run, or call .sync())>"


def _normalize_mode(mode: Optional[str]) -> str:
    """Resolve and validate the client communication mode."""
    raw_mode = (
        os.environ.get("OPENENV_CLIENT_MODE", "simulation") if mode is None else mode
    )
    normalized_mode = raw_mode.lower()
    if normalized_mode not in _VALID_CLIENT_MODES:
        raise ValueError(
            f"Invalid mode: '{normalized_mode}'. Must be 'simulation' or 'production'. "
            f"Set via constructor parameter or OPENENV_CLIENT_MODE environment variable."
        )
    return normalized_mode


def _is_localhost_ws_url(ws_url: str) -> bool:
    """Return True when the WebSocket URL targets the local loopback interface.

    The hostname is parsed from the URL so that only the actual host is matched.
    Substring matching is avoided because remote hosts such as
    ``my-localhost-proxy.example.com`` or ``127.0.0.1.example.com`` must not be
    treated as local.
    """
    hostname = urlsplit(ws_url).hostname
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _required_start_container_parameters(provider: Any) -> list[str]:
    """Return required arguments for a bound provider.start_container()."""
    try:
        signature = inspect.signature(provider.start_container)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


async def _best_effort_close(ws: ClientConnection) -> None:
    """Close a socket without letting a slow handshake or failure propagate.

    Scheduled as a background task (never awaited directly) so a dropped
    socket's close handshake can't hold up the caller that triggered the
    drop -- see `EnvClient._receive()`. `EnvClient._close_async()` awaits
    any still-running instance of this task before it returns, so a real
    `close()` call does wait for the handshake; only the caller that
    happened to trigger the drop is spared. `CancelledError` is caught
    alongside `Exception` because that awaiting is exactly what would
    otherwise cancel this mid-handshake if a caller called `close()`
    concurrently.
    """
    try:
        await ws.close()
    except (Exception, asyncio.CancelledError):
        pass  # Best effort


async def _best_effort_graceful_close(ws: ClientConnection) -> None:
    """Notify the server, then close a socket without propagating failures."""
    try:
        await ws.send(json.dumps({"type": "close"}))
    except (Exception, asyncio.CancelledError):
        pass  # Best effort
    await _best_effort_close(ws)


class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    """
    Async environment client for persistent sessions.

    This client maintains a persistent WebSocket connection to an environment
    server, enabling efficient multi-step interactions. Each client instance
    corresponds to a dedicated environment session on the server.

    The client is async by default. For synchronous usage, use the `.sync()`
    method to get a `SyncEnvClient` wrapper.

    Features:
    - Lower latency for sequential interactions
    - Session state is maintained server-side
    - Better suited for long-running episodes
    - Async by default for modern Python async/await patterns

    Examples:

        Async usage:

        ```python
        from envs.coding_env.client import CodingEnv

        async with CodingEnv(base_url="ws://localhost:8000") as env:
            result = await env.reset(seed=42)
            while not result.done:
                action = agent.predict(result.observation)
                result = await env.step(action)
        ```

        Sync usage via `.sync()` wrapper:

        ```python
        env = CodingEnv(base_url="ws://localhost:8000").sync()
        with env:
            result = env.reset(seed=42)
            result = env.step(action)
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        websocket_ping_interval_s: Optional[float] = 20.0,
        websocket_ping_timeout_s: Optional[float] = 20.0,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        mode: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize environment client.

        Args:
            base_url (`str`, *optional*):
                Base URL of the environment server (http:// or ws://). Will be converted to
                ws:// if http:// is provided. May be omitted when the provider
                has enough constructor state to start itself.
            connect_timeout_s (`float`, *optional*, defaults to `10.0`):
                Timeout for establishing WebSocket connection.
            message_timeout_s (`float`, *optional*, defaults to `60.0`):
                Timeout for receiving responses to messages.
            max_message_size_mb (`float`, *optional*, defaults to `100.0`):
                Maximum WebSocket message size in megabytes. Default 100MB to handle large
                observations (screenshots, DOM, etc.).
            websocket_ping_interval_s (`float` or `None`, *optional*, defaults to `20.0`):
                WebSocket keepalive ping interval. Pass `None` to disable.
            websocket_ping_timeout_s (`float` or `None`, *optional*, defaults to `20.0`):
                WebSocket keepalive pong timeout. Pass `None` to disable.
            provider (`ContainerProvider` or `RuntimeProvider`, *optional*):
                Container/runtime provider for lifecycle management.
            mode (`str`, *optional*):
                Communication mode: `'simulation'` for Gym-style API (default) or
                `'production'` for MCP JSON-RPC protocol. Can also be set via the
                `OPENENV_CLIENT_MODE` environment variable. Constructor parameter takes
                precedence over environment variable. Case-insensitive.
            headers (`dict[str, str]`, *optional*):
                Extra HTTP headers sent when opening the connection, e.g.
                `{"Authorization": f"Bearer {hf_token}"}` for a private Hugging Face Space.
        """
        if base_url is None and provider is None:
            raise ValueError("EnvClient requires either base_url or provider.")

        # Store mode (use object.__setattr__ to bypass immutability)
        object.__setattr__(self, "_mode", _normalize_mode(mode))

        self._base_url: Optional[str] = None
        self._ws_url: Optional[str] = None
        self._connect_timeout = connect_timeout_s
        self._message_timeout = message_timeout_s
        self._max_message_size = int(
            max_message_size_mb * 1024 * 1024
        )  # Convert MB to bytes
        self._websocket_ping_interval_s = websocket_ping_interval_s
        self._websocket_ping_timeout_s = websocket_ping_timeout_s
        self._headers = headers
        self._provider = provider
        self._provider_stopped = False
        self._provider_cleanup_pending = False
        self._start_provider_on_connect = base_url is None
        self._child_clients: list[EnvClient[Any, Any, Any]] = []
        self._ws: Optional[ClientConnection] = None
        self._execution_mode: Optional[str] = None
        self._sync_client: Optional["SyncEnvClient[ActT, ObsT, StateT]"] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        # Strong references for fire-and-forget socket closes (see _receive),
        # so the task isn't garbage-collected mid-close. Discarded on done.
        self._pending_close_tasks: set[asyncio.Task] = set()
        if base_url is not None:
            self._set_base_url(base_url)

    @property
    def base_url(self) -> Optional[str]:
        """Public read-only URL of the environment server this client targets.

        Returns the normalized `http(s)`/`ws(s)` base URL (no trailing slash),
        or `None` when the client was created with a provider but has not yet
        started its container (the URL is assigned lazily on `connect()`).

        This is the public counterpart to the private `self._base_url`; the
        previously-public `base_url` attribute was dropped in the `core`
        refactor, leaving sync consumers with no way to read the URL back.
        """
        return self._base_url

    def _set_base_url(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        ws_url = convert_to_ws_url(base_url)
        self._ws_url = f"{ws_url}/ws"

    def _start_provider_if_needed(self) -> None:
        # A missing URL does not mean the previous resource was released.
        # Retry its cleanup before start can overwrite the provider's handle.
        if self._provider_cleanup_pending:
            if not self._start_provider_on_connect:
                raise RuntimeError(
                    "Provider cleanup is pending for this client with an existing "
                    "base URL. Retry close() to finish cleanup, then create a new "
                    "client instead of reconnecting to the old URL."
                )
            self._stop_provider()
        if self._ws_url is not None:
            return
        if self._provider is None:
            raise RuntimeError("EnvClient has no base URL or provider.")
        if hasattr(self._provider, "start_container"):
            required_parameters = _required_start_container_parameters(self._provider)
            if required_parameters:
                required = ", ".join(required_parameters)
                raise ValueError(
                    f"{type(self._provider).__name__} does not support "
                    "provider-owned startup because start_container() requires "
                    f"{required}. Start the provider manually and pass base_url, "
                    "or configure a provider with a constructor-owned image/source."
                )
            self._provider_stopped = False
            base_url = self._provider.start_container()
            self._provider.wait_for_ready(base_url)
        elif hasattr(self._provider, "start"):
            self._provider_stopped = False
            base_url = self._provider.start()
            self._provider.wait_for_ready()
        else:
            raise TypeError("provider must define start_container() or start().")
        self._set_base_url(base_url)

    def _create_session_client(self) -> "EnvClient[Any, Any, Any]":
        # Match _start_provider_if_needed's startup condition. A provider with
        # an existing URL may already be serving the parent or other children.
        starts_provider_here = self._provider is not None and self._ws_url is None
        try:
            self._start_provider_if_needed()
            if self._base_url is None:
                raise RuntimeError("EnvClient has no base URL.")

            signature = inspect.signature(type(self))
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            candidate_kwargs = {
                "base_url": self._base_url,
                "connect_timeout_s": self._connect_timeout,
                "message_timeout_s": self._message_timeout,
                "max_message_size_mb": self._max_message_size / (1024 * 1024),
                "websocket_ping_interval_s": self._websocket_ping_interval_s,
                "websocket_ping_timeout_s": self._websocket_ping_timeout_s,
                "mode": self._mode,
                "headers": self._headers,
            }
            constructor_kwargs = {}
            for name, value in candidate_kwargs.items():
                if accepts_kwargs or name in signature.parameters:
                    constructor_kwargs[name] = value

            return type(self)(**constructor_kwargs)
        except Exception:
            if starts_provider_here and not self._provider_cleanup_pending:
                self._stop_provider_best_effort()
            raise

    async def new_session(self) -> "EnvClient[Any, Any, Any]":
        """
        Create and connect a new session against the same environment server.

        Returns:
            `EnvClient`: A connected child client of the same concrete type.

        The child session is tracked by this parent and closed when the parent
        is closed. Server-side capacity still applies: when the server is at
        `MAX_CONCURRENT_ENVS`, opening the child WebSocket can fail and is
        surfaced as a connection error.
        """
        client = self._create_session_client()
        await client.connect()
        self._child_clients.append(client)
        return client

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of _mode after initialization."""
        if name == "_mode" and hasattr(self, "_mode"):
            raise AttributeError("Cannot modify mode after initialization")
        super().__setattr__(name, value)

    def _claim_execution_mode(self, mode: str) -> None:
        """Lock the client to sync or async execution on first use."""
        if self._execution_mode is None:
            self._execution_mode = mode
        elif self._execution_mode != mode:
            raise RuntimeError(
                f"EnvClient is already being used in {self._execution_mode} mode. "
                "Create a separate client instance when mixing sync and async code."
            )

    def _run_sync(
        self,
        coro_factory: Callable[[], Any],
        *,
        allow_async_handoff: bool = False,
    ) -> Any:
        """Run an async operation through the sync wrapper."""
        if allow_async_handoff and self._execution_mode == "async":
            self._execution_mode = "sync"
        else:
            self._claim_execution_mode("sync")
        if self._sync_client is None:
            self._sync_client = self.sync()
        return self._sync_client._run(coro_factory())

    def _dispatch(self, coro_factory: Callable[[], Any]) -> Any:
        """Return an awaitable in async code and a concrete result in sync code."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if self._execution_mode == "sync":
            sync_loop = (
                self._sync_client._loop if self._sync_client is not None else None
            )
            if running_loop is sync_loop:
                return coro_factory()
            return self._run_sync(coro_factory)
        if running_loop is not None:
            if self._execution_mode == "async":
                return coro_factory()
            return _AutoAsyncResult(self, coro_factory)

        return self._run_sync(coro_factory, allow_async_handoff=True)

    def connect(self) -> Any:
        return self._dispatch(self._connect_async)

    async def _connect_async(self) -> "EnvClient":
        """
        Establish WebSocket connection to the server.

        Returns:
            self for method chaining

        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._ws is not None:
            if self._ws.state in (State.CLOSING, State.CLOSED):
                # Closed by the far end: a keepalive timeout, a tunnel dropping
                # the socket, the server restarting. The object is still not
                # None, so without this check it stays cached and every later
                # call raises `ConnectionClosed` for the rest of the process's
                # life. Only a demonstrably closed socket is dropped, so one
                # still CONNECTING is left alone.
                self._ws = None
                self._ws_loop = None
            elif self._ws_loop is asyncio.get_running_loop():
                return self
            # Connected from a different event loop than the one running
            # now -- e.g. `client = await Client.from_env(...)` inside
            # `asyncio.run(...)`, then `client.sync()` drives every later
            # call on `SyncEnvClient`'s own dedicated background loop. The
            # websocket object is bound to internals of the original loop,
            # which is typically already closed by the time we get here, so
            # it cannot be reused (or even cleanly closed) from this loop.
            # Drop the stale reference and reconnect fresh below rather than
            # silently no-op-ing onto a dead connection.
            self._ws = None
            self._ws_loop = None

        # A timed-out request drops its socket immediately but closes it in the
        # background so the timeout itself remains prompt. Wait for that close
        # before opening a replacement: the old server-side session continues
        # occupying a capacity slot until the close handshake finishes, and
        # many environments allow only one session.
        await self._drain_pending_close_tasks()

        try:
            self._start_provider_if_needed()
        except Exception:
            # A failed cleanup retry must propagate without attempting stop
            # again. For a new startup failure, preserve its original error.
            if not self._provider_cleanup_pending:
                with suppress(Exception):
                    await self.close()
            raise

        assert self._ws_url is not None

        # Disable the proxy for localhost connections via the per-connection
        # `proxy` argument rather than mutating the process-global NO_PROXY
        # env var: concurrent connect() calls (e.g. asyncio.gather over many
        # env clients) would otherwise race on os.environ and leak state.
        connect_kwargs: Dict[str, Any] = {}
        if _is_localhost_ws_url(self._ws_url):
            connect_kwargs["proxy"] = None

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                ping_interval=self._websocket_ping_interval_s,
                ping_timeout=self._websocket_ping_timeout_s,
                additional_headers=self._headers,
                **connect_kwargs,
            )
            self._ws_loop = asyncio.get_running_loop()
        except Exception as e:
            await self.close()
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e

        return self

    def disconnect(self) -> Any:
        return self._dispatch(self._disconnect_async)

    def _schedule_socket_close(
        self, ws: ClientConnection, *, notify_server: bool = False
    ) -> asyncio.Task[None]:
        """Schedule and track a socket close until its handshake finishes."""
        close_coro = (
            _best_effort_graceful_close(ws) if notify_server else _best_effort_close(ws)
        )
        close_task = asyncio.create_task(close_coro)
        self._pending_close_tasks.add(close_task)
        close_task.add_done_callback(self._pending_close_tasks.discard)
        return close_task

    async def _disconnect_async(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            ws = self._ws
            ws_loop = self._ws_loop
            # Detach first so cancellation during the close handshake cannot
            # leave a stale socket cached for a later operation.
            self._ws = None
            self._ws_loop = None
            same_loop = ws_loop is asyncio.get_running_loop()
            if same_loop:
                # Keep ownership of the handshake if this caller is cancelled.
                # A later close/reconnect drains the task before tearing down
                # the loop or opening a replacement server session.
                close_task = self._schedule_socket_close(ws, notify_server=True)
                await asyncio.shield(close_task)

    async def _drain_pending_close_tasks(self) -> None:
        """Wait for background socket closes owned by the current event loop.

        Shielding keeps cancellation of the caller from cancelling the close
        tasks themselves. This matters both before reconnecting, when the old
        server session must release its capacity slot, and during explicit
        client shutdown.
        """
        loop = asyncio.get_running_loop()
        tasks = [
            task
            for task in tuple(self._pending_close_tasks)
            if not task.done() and task.get_loop() is loop
        ]
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    async def _ensure_connected(self) -> None:
        """Ensure WebSocket connection is established on the current loop.

        Always delegates to `_connect_async()` rather than pre-checking
        `self._ws is None`: `_connect_async()` itself is the one that knows
        whether an existing `_ws` is reusable (same event loop) or stale (a
        different one, e.g. from a prior `from_env()` call now being driven
        through `.sync()`'s own loop).
        """
        await self._connect_async()

    async def _send(self, message: Dict[str, Any]) -> None:
        """Send a message over the WebSocket."""
        await self._ensure_connected()
        assert self._ws is not None
        await self._ws.send(json.dumps(message))

    async def _receive(self) -> Dict[str, Any]:
        """Receive the response on the same WebSocket that carried the request.

        Reconnecting here would wait on a fresh socket where the request was
        never sent. The next complete operation may reconnect in `_send()`.
        """
        assert self._ws is not None
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self._message_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # The server may still write the response for this request to the
            # socket after we give up waiting on it. If we left the socket
            # open, the next call's `_receive()` would read that stale frame
            # and silently pair it with an unrelated request -- and every
            # response after that would be shifted by one for the life of
            # the connection. Drop the socket so the next `_send()` opens a
            # fresh one instead of reusing a desynced one.
            #
            # An outer `asyncio.wait_for` around the whole call (e.g. a
            # caller-imposed deadline on step()) cancels this await with
            # CancelledError, not our own message_timeout's TimeoutError --
            # same desync, different exception, so both are caught here.
            ws = self._ws
            self._ws = None
            self._ws_loop = None
            # Fire-and-forget: `close()` waits out the library's close
            # handshake (`close_timeout`, 10s by default) if the server is
            # slow to ack, and *awaiting* that here would hold up this
            # exception -- so an outer `asyncio.wait_for(..., timeout=50ms)`
            # would actually block for up to 10s before its deadline was
            # honored. Scheduling it lets the exception propagate
            # immediately while the close still happens in the background.
            self._schedule_socket_close(ws)
            raise
        return json.loads(raw)

    async def _send_and_receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and wait for response."""
        try:
            await self._send(message)
        except ConnectionClosed:
            # A refused session (e.g. at capacity) sends an error frame then closes. If
            # the close beats our send, read the queued frame (recv() re-raises if none).
            pass
        response = await self._receive()

        # Check for error response
        if response.get("type") == "error":
            error_data = response.get("data", {})
            raise RuntimeError(
                f"Server error: {error_data.get('message', 'Unknown error')} "
                f"(code: {error_data.get('code', 'UNKNOWN')})"
            )

        return response

    @classmethod
    def _bootstrap_container(
        cls: Type[EnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> EnvClientT:
        """Start a Docker container and build an *unconnected* client for it.

        Invoked lazily by the `from_docker_image` bootstrap handle when it is
        awaited or `.sync()`'d: the container start / readiness wait is plain
        blocking code, so the only thing that differs between the async and sync
        resolution paths is how the WebSocket is connected (awaited vs. run sync).
        """
        if provider is None:
            provider = LocalDockerProvider()

        try:
            # Start container, wait for readiness, then build the client.
            base_url = provider.start_container(image, **kwargs)
            provider.wait_for_ready(base_url)
            return cls(base_url=base_url, provider=provider)
        except Exception:
            # No EnvClient exists yet for the caller to close(), so release the
            # container here if start / readiness / construction fails.
            provider.stop_container()
            raise

    @classmethod
    def from_docker_image(
        cls: Type[EnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> "_BootstrapResult[EnvClientT]":
        """
        Create an environment client by spinning up a Docker container.

        Returns a bootstrap handle so the same call works from both async and
        synchronous code: `await` it for the connected async client, or chain
        `.sync()` for a connected `SyncEnvClient` (e.g. a TRL GRPO rollout loop
        that cannot `await`). Bootstrap is lazy — the container starts when the
        handle is resolved.

        Args:
            image (`str`):
                Docker image name to run (e.g., `"coding-env:latest"`).
            provider (`ContainerProvider`, *optional*):
                Container provider to use. Defaults to `LocalDockerProvider`.
            **kwargs:
                Additional arguments to pass to `provider.start_container()`.

        Returns:
            `_BootstrapResult`: `await` for a connected async client, or call
            `.sync()` for a connected `SyncEnvClient`.

        Examples:

        ```python
        # Async
        env = await MyEnv.from_docker_image("coding-env:latest")

        # Sync
        env = MyEnv.from_docker_image("coding-env:latest").sync()
        result = env.reset()
        ```
        """
        return _BootstrapResult(
            lambda: cls._bootstrap_container(image, provider, **kwargs)
        )

    @classmethod
    def _bootstrap_env(
        cls: Type[EnvClientT],
        repo_id: str,
        *,
        use_docker: bool = True,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        **provider_kwargs: Any,
    ) -> EnvClientT:
        """Start a HF Space (docker or uv) and build an *unconnected* client.

        Invoked lazily by the `from_env` bootstrap handle; see
        `_bootstrap_container` for the same split rationale. On a startup
        failure the spawned process (and, for a git+ `project_path`, the temp
        clone directory) is released before re-raising; a later *connection*
        failure is cleaned up by `_connect_async` via `close()`.
        """
        # Extract start args that apply to both providers
        start_args = {}
        for key in ("port", "env_vars", "workers"):
            if key in provider_kwargs:
                start_args[key] = provider_kwargs.pop(key)

        if use_docker:
            # Docker mode: pull from HF registry
            docker_provider = provider or LocalDockerProvider()
            tag = provider_kwargs.pop("tag", "latest")
            image = f"registry.hf.space/{hf_space_subdomain(repo_id)}:{tag}"
            try:
                base_url = docker_provider.start_container(
                    image, **start_args, **provider_kwargs
                )
                docker_provider.wait_for_ready(base_url)
                return cls(base_url=base_url, provider=docker_provider)
            except Exception:
                # No EnvClient exists yet for the caller to close(), so release
                # the container here if start / readiness / construction fails.
                docker_provider.stop_container()
                raise
        else:
            # UV mode: clone and run with uv
            if provider is None:
                uv_kwargs = dict(provider_kwargs)
                project_path = uv_kwargs.pop("project_path", None)
                if project_path is None:
                    project_path = f"git+https://huggingface.co/spaces/{repo_id}"

                provider = UVProvider(project_path=project_path, **uv_kwargs)
            else:
                if provider_kwargs:
                    raise ValueError(
                        "provider_kwargs cannot be used when supplying a provider instance"
                    )

            try:
                context_timeout_s = getattr(provider, "context_timeout_s", None)
                deadline = (
                    time.monotonic() + context_timeout_s
                    if context_timeout_s is not None
                    else None
                )
                base_url = provider.start(**start_args)
                if deadline is None:
                    provider.wait_for_ready()
                else:
                    provider.wait_for_ready(
                        timeout_s=max(0.0, deadline - time.monotonic())
                    )
                return cls(base_url=base_url, provider=provider)
            except Exception:
                # No EnvClient exists yet for the caller to close(), so this is
                # the only chance to release the spawned process and (for a
                # git+ project_path) the temp clone directory. Covers start,
                # readiness, and client construction (e.g. an invalid mode).
                provider.stop()
                raise

    @classmethod
    def from_env(
        cls: Type[EnvClientT],
        repo_id: str,
        *,
        use_docker: bool = True,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        **provider_kwargs: Any,
    ) -> "_BootstrapResult[EnvClientT]":
        """
        Create a client from a Hugging Face Space.

        Returns a bootstrap handle: `await` it for the connected async client, or
        chain `.sync()` for a connected `SyncEnvClient`. Bootstrap is lazy — the
        Space starts when the handle is resolved.

        Args:
            repo_id (`str`):
                Hugging Face space identifier `{org}/{space}`.
            use_docker (`bool`, *optional*, defaults to `True`):
                When `True`, pull from the HF registry and launch via `LocalDockerProvider`.
                When `False`, run the space locally with `UVProvider`.
            provider (`ContainerProvider` or `RuntimeProvider`, *optional*):
                Provider instance to reuse. Must be a `ContainerProvider` when
                `use_docker=True` and a `RuntimeProvider` otherwise.
            **provider_kwargs:
                Additional keyword arguments forwarded to either the container provider's
                `start_container` (docker) or to the `UVProvider` constructor/start (uv).
                When `use_docker=False`, the `project_path` argument can be used to override
                the default git URL (`git+https://huggingface.co/spaces/{repo_id}`).

        Returns:
            `_BootstrapResult`: `await` for a connected async client, or call
            `.sync()` for a connected `SyncEnvClient`.

        Examples:

            ```python
            # Async: pull and run from HF Docker registry
            env = await MyEnv.from_env("openenv/echo-env")

            # Sync: chain .sync()
            env = MyEnv.from_env("openenv/echo-env").sync()

            # Run locally with UV (clones the space)
            env = await MyEnv.from_env("openenv/echo-env", use_docker=False)

            # Run from a local checkout
            env = await MyEnv.from_env(
                "openenv/echo-env",
                use_docker=False,
                project_path="/path/to/local/checkout"
            )
            ```
        """
        return _BootstrapResult(
            lambda: cls._bootstrap_env(
                repo_id,
                use_docker=use_docker,
                provider=provider,
                **provider_kwargs,
            )
        )

    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
        """Convert a JSON response from the env server to StepResult[ObsT]."""
        raise NotImplementedError

    @abstractmethod
    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        """Convert a JSON response from the state endpoint to a State object."""
        raise NotImplementedError

    def reset(self, **kwargs: Any) -> Any:
        return self._dispatch(lambda: self._reset_async(**kwargs))

    async def _reset_async(self, **kwargs: Any) -> StepResult[ObsT]:
        """
        Reset the environment with optional parameters.

        Args:
            **kwargs:
                Optional parameters passed to the environment's reset method.

        Returns:
            StepResult containing initial observation
        """
        message = {
            "type": "reset",
            "data": kwargs,
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    def step(self, action: ActT, **kwargs: Any) -> Any:
        return self._dispatch(lambda: self._step_async(action, **kwargs))

    async def _step_async(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
        """
        Execute an action in the environment.

        Args:
            action:
                The action to execute.
            **kwargs:
                Optional parameters (currently ignored).

        Returns:
            StepResult containing observation, reward, and done status
        """
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    def state(self) -> Any:
        return self._dispatch(self._state_async)

    async def _state_async(self) -> StateT:
        """
        Get the current environment state from the server.

        Returns:
            State object with environment state information
        """
        message = {"type": "state"}
        response = await self._send_and_receive(message)
        return self._parse_state(response.get("data", {}))

    def close(self) -> Any:
        return self._dispatch(self._close_async)

    async def _close_child_clients(self) -> asyncio.CancelledError | None:
        """Close every captured child while deferring caller cancellation."""
        children = list(self._child_clients)
        self._child_clients.clear()
        close_tasks = []
        deferred_cancellation: asyncio.CancelledError | None = None

        for child in children:
            try:
                close_tasks.append(asyncio.ensure_future(child.close()))
            except asyncio.CancelledError as exc:
                if deferred_cancellation is None:
                    deferred_cancellation = exc
            except Exception:
                pass  # Best effort

        if close_tasks:
            close_group = asyncio.gather(*close_tasks, return_exceptions=True)
            while not close_group.done():
                try:
                    await asyncio.shield(close_group)
                except asyncio.CancelledError as exc:
                    if deferred_cancellation is None:
                        deferred_cancellation = exc

            for result in close_group.result():
                if (
                    isinstance(result, asyncio.CancelledError)
                    and deferred_cancellation is None
                ):
                    deferred_cancellation = result

        return deferred_cancellation

    async def _close_async(self) -> None:
        """
        Close the WebSocket connection and clean up resources.

        If this client was created via from_docker_image() or from_env(),
        this will also stop and remove the associated container/process.
        """
        deferred_cancellation: asyncio.CancelledError | None = None
        try:
            try:
                try:
                    child_cancellation = await self._close_child_clients()
                except asyncio.CancelledError as exc:
                    child_cancellation = exc
                if child_cancellation is not None:
                    deferred_cancellation = child_cancellation

                # A real close waits out backgrounded closes, while shielding
                # their socket handshakes from cancellation.
                try:
                    await self._drain_pending_close_tasks()
                except asyncio.CancelledError as exc:
                    if deferred_cancellation is None:
                        deferred_cancellation = exc
            finally:
                # Run even when child or pending-close cleanup is cancelled. A
                # client may already have reconnected, and that current socket
                # must not remain cached or open during teardown.
                try:
                    await self._disconnect_async()
                except asyncio.CancelledError as exc:
                    if deferred_cancellation is None:
                        deferred_cancellation = exc
        finally:
            self._stop_provider()

        if deferred_cancellation is not None:
            raise deferred_cancellation

    def _stop_provider(self) -> None:
        """Stop the provider once, retaining it for retries and later startup."""
        try:
            provider = self._provider
            if provider is None or self._provider_stopped:
                return

            if hasattr(provider, "stop_container"):
                stop = provider.stop_container
            elif hasattr(provider, "stop"):
                stop = provider.stop
            else:
                return

            self._provider_cleanup_pending = True
            stop()
            # Only a successful stop discharges our cleanup responsibility.
            self._provider_cleanup_pending = False
            self._provider_stopped = True
        finally:
            if self._start_provider_on_connect:
                self._base_url = None
                self._ws_url = None

    def _stop_provider_best_effort(self) -> None:
        """Stop the underlying provider directly, ignoring any errors.

        Releases a started container/process when there is no connected client
        to `close()` through the normal path — e.g. sync bootstrap setup fails
        after the provider started but before the connection is established, so
        routing cleanup through the (possibly broken) sync loop is not an option.
        """
        with suppress(Exception):
            self._stop_provider()

    async def __aenter__(self) -> "EnvClient":
        """Enter async context manager, ensuring connection is established."""
        self._claim_execution_mode("async")
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing connection."""
        await self.close()

    def __enter__(self) -> "EnvClient":
        """Enter sync context manager, ensuring connection is established."""
        self._run_sync(self._connect_async)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit sync context manager, closing connection."""
        self._run_sync(self._close_async)

    def sync(self) -> "SyncEnvClient":
        """
        Return a synchronous wrapper around this async client.

        Use this method when you need synchronous access to the environment
        without async/await syntax. This is useful for:
        - Integration with synchronous codebases
        - Interactive/REPL usage
        - Stopping async from "infecting" the call stack

        Returns:
            SyncEnvClient wrapper that provides synchronous methods

        Examples:

            ```python
            async_client = GenericEnvClient(base_url="http://localhost:8000")
            sync_client = async_client.sync()

            with sync_client:
                result = sync_client.reset()
                result = sync_client.step({"code": "print('hello')"})
            ```
        """
        from .sync_client import SyncEnvClient

        if self._sync_client is None:
            self._sync_client = SyncEnvClient(self)
        return self._sync_client
