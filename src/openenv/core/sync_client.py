# SPDX-License-Identifier: BSD-3-Clause

"""
Synchronous wrapper for async EnvClient.

This module provides a SyncEnvClient that wraps an async EnvClient,
allowing synchronous usage while the underlying client uses async I/O.

Examples:

    ```python
    from openenv.core import GenericEnvClient

    # Create async client and get sync wrapper
    async_client = GenericEnvClient(base_url="http://localhost:8000")
    sync_client = async_client.sync()

    # Use synchronous API
    with sync_client:
        result = sync_client.reset()
        result = sync_client.step({"code": "print('hello')"})
    ```
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
from contextlib import suppress
from typing import Any, Dict, Generic, TYPE_CHECKING, TypeVar

from .client_types import StateT, StepResult

if TYPE_CHECKING:
    from .env_client import EnvClient

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")


class SyncEnvClient(Generic[ActT, ObsT, StateT]):
    """
    Synchronous wrapper around an async EnvClient.

    This class provides a synchronous interface to an async EnvClient,
    making it easier to use in synchronous code or to stop async from
    "infecting" the entire call stack.

    The wrapper executes async operations on a dedicated background event loop
    so connection state remains bound to a single loop.

    For guaranteed resource cleanup, use `with SyncEnvClient(...)` or call
    `close()` explicitly. `__del__` is best-effort only and may not run
    reliably (for example, during interpreter shutdown).

    Examples:

        ```python
        # From an async client
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        # Use synchronous context manager
        with sync_client:
            result = sync_client.reset()
            result = sync_client.step({"action": "test"})
        ```

    Attributes:
        _async: The wrapped async EnvClient instance
    """

    def __init__(self, async_client: "EnvClient[ActT, ObsT, StateT]"):
        """
        Initialize sync wrapper around an async client.

        Args:
            async_client (`EnvClient`):
                The async client to wrap.
        """
        self._async = async_client
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._loop_init_lock = threading.Lock()
        self._async_wrapper_cache: Dict[str, Any] = {}
        self._child_clients: list[SyncEnvClient[ActT, ObsT, StateT]] = []

    def _run_loop_forever(self) -> None:
        """Run a dedicated event loop for this sync client."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        loop.run_forever()
        loop.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start background loop thread on first use."""
        if (
            self._loop is not None
            and self._loop_thread
            and self._loop_thread.is_alive()
        ):
            return self._loop

        # Protect loop initialization when multiple threads race on first use.
        with self._loop_init_lock:
            if (
                self._loop is not None
                and self._loop_thread
                and self._loop_thread.is_alive()
            ):
                return self._loop

            self._loop_ready.clear()
            self._loop_thread = threading.Thread(
                target=self._run_loop_forever,
                name="openenv-sync-client-loop",
                daemon=True,
            )
            self._loop_thread.start()
            if not self._loop_ready.wait(timeout=5):
                raise RuntimeError("Timed out starting sync client event loop")
            assert self._loop is not None
            return self._loop

    def _claim_sync_mode(self) -> None:
        if hasattr(self._async, "_claim_execution_mode"):
            self._async._claim_execution_mode("sync")

    def _run(self, coro: Any) -> Any:
        """Run coroutine on dedicated loop and block for result."""
        self._claim_sync_mode()
        loop = self._ensure_loop()
        future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(
            coro, loop
        )
        return future.result()

    def _stop_loop(self) -> None:
        """Stop and join background loop thread."""
        loop = self._loop
        thread = self._loop_thread
        if loop is None:
            return

        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)

        self._loop = None
        self._loop_thread = None

    @property
    def async_client(self) -> "EnvClient[ActT, ObsT, StateT]":
        """Access the underlying async client."""
        return self._async

    def connect(self) -> "SyncEnvClient[ActT, ObsT, StateT]":
        """
        Establish connection to the server.

        Returns:
            self for method chaining
        """
        self._claim_sync_mode()
        self._run(self._async._connect_async())
        return self

    def disconnect(self) -> None:
        """Close the connection."""
        self._claim_sync_mode()
        self._run(self._async._disconnect_async())

    def reset(self, **kwargs: Any) -> StepResult[ObsT]:
        """
        Reset the environment.

        Args:
            **kwargs:
                Optional parameters passed to the environment's reset method.

        Returns:
            StepResult containing initial observation
        """
        self._claim_sync_mode()
        return self._run(self._async._reset_async(**kwargs))

    def step(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
        """
        Execute an action in the environment.

        Args:
            action:
                The action to execute.
            **kwargs:
                Optional parameters.

        Returns:
            StepResult containing observation, reward, and done status
        """
        self._claim_sync_mode()
        return self._run(self._async._step_async(action, **kwargs))

    def state(self) -> StateT:
        """
        Get the current environment state.

        Returns:
            State object with environment state information
        """
        self._claim_sync_mode()
        return self._run(self._async._state_async())

    def close(self) -> None:
        """Close the connection and clean up resources."""
        try:
            for child in list(self._child_clients):
                with suppress(Exception):
                    child.close()
            self._child_clients.clear()
            self._claim_sync_mode()
            self._run(self._async._close_async())
        finally:
            self._stop_loop()

    def new_session(self) -> "SyncEnvClient[ActT, ObsT, StateT]":
        """
        Create a new synchronous session against the same environment server.

        Returns:
            `SyncEnvClient`: A connected child wrapper around a child async
            client of the same concrete type.

        The child session is tracked by this parent and closed when the parent
        is closed. Call this after the parent has connected, because the child
        reuses the parent's current base URL. Server-side capacity still
        applies: when the server is at `MAX_CONCURRENT_ENVS`, opening the child
        WebSocket can fail and is surfaced as a connection error.
        """
        async_client = self._async._create_session_client()
        client = SyncEnvClient(async_client)
        client.connect()
        self._child_clients.append(client)
        return client

    def __enter__(self) -> "SyncEnvClient[ActT, ObsT, StateT]":
        """Enter context manager, establishing connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing connection."""
        self.close()

    def __del__(self) -> None:
        """
        Best-effort cleanup for background loop thread.

        Do not rely on this for deterministic cleanup; prefer context-manager
        usage or an explicit `close()` call.
        """
        try:
            self._stop_loop()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to the async client.

        Async methods are wrapped to run on the sync client's dedicated loop.
        """
        attr = getattr(self._async, name)

        if inspect.iscoroutinefunction(attr):
            cached = self._async_wrapper_cache.get(name)
            if cached is not None:
                return cached

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                method = getattr(self._async, name)
                return self._run(method(*args, **kwargs))

            self._async_wrapper_cache[name] = sync_wrapper
            return sync_wrapper

        return attr

    # Delegate abstract method implementations to the wrapped client
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        """Delegate to async client's _step_payload."""
        return self._async._step_payload(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
        """Delegate to async client's _parse_result."""
        return self._async._parse_result(payload)

    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        """Delegate to async client's _parse_state."""
        return self._async._parse_state(payload)
