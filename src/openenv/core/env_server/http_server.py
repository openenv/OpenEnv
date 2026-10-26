# SPDX-License-Identifier: BSD-3-Clause

"""
HTTP server wrapper for Environment instances.

This module provides utilities to wrap any Environment subclass and expose it
over HTTP and WebSocket endpoints that EnvClient can consume.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    cast,
    Dict,
    Optional,
    Type,
    TypeVar,
)

_MISSING = object()
_MCPResult = TypeVar("_MCPResult")

from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Request,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError

from ._utils import overrides_method
from .interfaces import Environment
from .mcp_environment import get_server_tools
from .mcp_types import (
    CallToolAction,
    CallToolObservation,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    ListToolsAction,
    ListToolsObservation,
    McpMethod,
    WSMCPMessage,
    WSMCPResponse,
)
from .route_config import GetEndpointConfig, register_get_endpoints
from .serialization import deserialize_action, serialize_observation
from .types import (
    Action,
    ConcurrencyConfig,
    EnvironmentMetadata,
    GetTaskRangeRequest,
    GetTaskRequest,
    HealthResponse,
    HealthStatus,
    ListTasksRequest,
    NumTasksRequest,
    Observation,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    ServerCapacityStatus,
    ServerMode,
    SessionInfo,
    State,
    StepRequest,
    StepResponse,
    WSCloseMessage,
    WSErrorCode,
    WSErrorResponse,
    WSObservationResponse,
    WSResetMessage,
    WSStateMessage,
    WSStateResponse,
    WSStepMessage,
)


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable form.

    Handles Pydantic models, dataclasses, and other common types.

    Args:
        obj (`Any`):
            The object to convert.

    Returns:
        `Any`: A JSON-serializable representation of the object.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        # Pydantic model
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        # Object with __dict__
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()}
    # Fallback to string representation
    return str(obj)


async def _maybe_await(value: Any) -> Any:
    """Await values returned by async task APIs while preserving sync APIs."""
    if inspect.isawaitable(value):
        return await value
    return value


from .exceptions import (
    ConcurrencyConfigurationError,
    EnvironmentFactoryError,
    SessionCapacityError,
)


class HTTPEnvServer:
    """
    HTTP server wrapper for Environment instances.

    This class wraps an Environment and exposes its reset(), step(), and state
    methods as HTTP and WebSocket endpoints compatible with EnvClient.

    The server expects:
    - Action deserialization: Converts JSON dict to Action subclass
    - Observation serialization: Converts Observation subclass to JSON dict

    Examples:

        ```python
        from core.env_server import HTTPEnvServer
        from envs.coding_env.server import CodeExecutionEnvironment
        from envs.coding_env.models import CodeAction, CodeObservation

        # Pass environment class (factory pattern)
        server = HTTPEnvServer(
            env=CodeExecutionEnvironment,
            action_cls=CodeAction,
            observation_cls=CodeObservation,
            max_concurrent_envs=4,
        )

        # Register routes with FastAPI
        from fastapi import FastAPI
        app = FastAPI()
        server.register_routes(app)
        ```
    """

    def __init__(
        self,
        env: Callable[[], Environment],
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        max_concurrent_envs: Optional[int] = None,
        concurrency_config: Optional[ConcurrencyConfig] = None,
        env_name: Optional[str] = None,
        state_cls: Type[State] = State,
        *,
        reset_observation_cls: Optional[Type[Observation]] = None,
    ):
        """
        Initialize HTTP server wrapper.

        Args:
            env (`Callable[[], Environment]`):
                Environment factory (callable) that creates new instances. Will be called
                to create a new environment for each WebSocket session.
            action_cls (`Type[Action]`):
                The `Action` subclass this environment expects.
            observation_cls (`Type[Observation]`):
                The `Observation` subclass returned by step.
            reset_observation_cls (`Type[Observation]`, *optional*):
                The reset observation model published in `/schema`. Defaults to
                `observation_cls`; declare a distinct model when reset differs.
            max_concurrent_envs (`int`, *optional*):
                Maximum number of concurrent WebSocket sessions. Mutually exclusive with
                `concurrency_config`.
            concurrency_config (`ConcurrencyConfig`, *optional*):
                Advanced concurrency settings. Mutually exclusive with
                `max_concurrent_envs`.
            env_name (`str`, *optional*):
                Public environment name used by task/split endpoints.
            state_cls (`Type[State]`, *optional*, defaults to `State`):
                The `State` subclass this environment reports. Used for the `/state`
                response model and the `state` entry of `/schema`, so that fields
                declared by the subclass are published and serialized.

        Raises:
            `ValueError`: If both `max_concurrent_envs` and `concurrency_config` are provided.
            `ConcurrencyConfigurationError`: If `max_concurrent_envs` > 1 for an environment
                that is not marked as `SUPPORTS_CONCURRENT_SESSIONS`.
        """
        # Validate that env is callable
        if not callable(env):
            raise TypeError(
                f"env must be a callable (class or factory function), got {type(env)}. "
                f"Pass the environment class (e.g., MyEnvironment) not an instance (e.g., MyEnvironment())."
            )

        self._env_factory: Callable[[], Environment] = env

        # Handle concurrency configuration
        if max_concurrent_envs is not None and concurrency_config is not None:
            raise ValueError(
                "Cannot specify both 'max_concurrent_envs' and 'concurrency_config'. "
                "Please use only one method to configure concurrency."
            )

        if concurrency_config is not None:
            self._concurrency_config = concurrency_config
        elif max_concurrent_envs is not None:
            self._concurrency_config = ConcurrencyConfig(
                max_concurrent_envs=max_concurrent_envs,
                session_timeout=None,
            )
        else:
            # Default configuration
            self._concurrency_config = ConcurrencyConfig(
                max_concurrent_envs=1,
                session_timeout=None,
            )

        self._max_concurrent_envs = self._concurrency_config.max_concurrent_envs

        # Validate concurrency configuration
        self._validate_concurrency_safety()

        self.action_cls = action_cls
        self.observation_cls = observation_cls
        self.reset_observation_cls = reset_observation_cls or observation_cls
        self.state_cls = state_cls
        self.env_name = env_name or self._default_env_name()

        # Session management for WebSocket connections
        self._sessions: Dict[str, Optional[Environment]] = {}
        self._session_executors: Dict[str, ThreadPoolExecutor] = {}
        self._session_stacks: Dict[str, AsyncExitStack] = {}
        self._session_info: Dict[str, SessionInfo] = {}
        self._session_websocket_attachments: set[str] = set()
        self._session_pending_closes: set[str] = set()
        self._session_lock = asyncio.Lock()

        # Create thread pool for running sync code in async context
        # This is needed for environments using sync libraries (e.g., Playwright)
        self._executor = ThreadPoolExecutor(max_workers=32)

        self._requires_single_thread_executor = self._detect_single_thread_requirement()
        self._shared_session_executor: Optional[ThreadPoolExecutor] = None
        if self._requires_single_thread_executor:
            self._shared_session_executor = ThreadPoolExecutor(max_workers=1)

        # Idle session reaper configuration.
        # Timeout is taken from ConcurrencyConfig.session_timeout;
        # None means no timeout (default — reaper is a no-op).
        self._session_idle_timeout_s: Optional[float] = (
            self._concurrency_config.session_timeout
        )
        self._reaper_task: Optional[asyncio.Task[None]] = None

    def _default_env_name(self) -> str:
        factory = self._env_factory
        if inspect.isclass(factory):
            return factory.__name__
        return getattr(factory, "__name__", "environment")

    def _validate_concurrency_safety(self) -> None:
        """
        Validate that the environment supports the configured concurrency level.

        Raises:
            `ConcurrencyConfigurationError`: If `max_concurrent_envs` > 1 for an environment
                that is not marked as `SUPPORTS_CONCURRENT_SESSIONS`.
        """
        import functools

        if self._max_concurrent_envs <= 1:
            return

        factory = self._env_factory
        if isinstance(factory, functools.partial):
            factory = factory.func

        if inspect.isclass(factory):
            env_cls = factory
        else:
            _temp_env = self._env_factory()
            env_cls = type(_temp_env)
            _temp_env.close()
            del _temp_env

        if not getattr(env_cls, "SUPPORTS_CONCURRENT_SESSIONS", False):
            raise ConcurrencyConfigurationError(
                environment_name=env_cls.__name__,
                max_concurrent_envs=self._max_concurrent_envs,
            )

    def _detect_single_thread_requirement(self) -> bool:
        import functools

        factory = self._env_factory
        if isinstance(factory, functools.partial):
            factory = factory.func
        if inspect.isclass(factory):
            return getattr(factory, "REQUIRES_SINGLE_THREAD_EXECUTOR", False)
        return False

    def get_capacity_status(self) -> ServerCapacityStatus:
        """
        Get the current capacity status of the server.

        Returns:
            [`ServerCapacityStatus`] with current session counts and availability.
        """
        return ServerCapacityStatus.from_counts(
            active=len(self._sessions),
            max_sessions=self._max_concurrent_envs,
        )

    async def _run_sync_in_thread_pool(
        self, func: Callable[..., Observation], *args, **kwargs
    ) -> Observation:
        """Run a synchronous function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def _get_valid_kwargs(
        self,
        sig: inspect.Signature,
        kwargs: Dict[str, Any],
        skip_params: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """Filter kwargs to only include parameters accepted by the function signature."""
        if skip_params is None:
            skip_params = set()

        valid_kwargs = {}

        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        for k, v in kwargs.items():
            if k in sig.parameters or has_kwargs:
                if k not in skip_params:
                    valid_kwargs[k] = v

        return valid_kwargs

    async def _create_session(self) -> tuple[str, Environment]:
        """
        Create a new WebSocket session with its own environment instance.

        Returns:
            `tuple[str, Environment]`: Tuple of (session_id, environment).

        Raises:
            `SessionCapacityError`: If max concurrent sessions reached.
            `EnvironmentFactoryError`: If the factory fails to create an environment.
        """
        async with self._session_lock:
            if len(self._sessions) >= self._max_concurrent_envs:
                raise SessionCapacityError(
                    active_sessions=len(self._sessions),
                    max_sessions=self._max_concurrent_envs,
                )

            session_id = str(uuid.uuid4())
            current_time = time.time()

            # Create executor and reserve slot so capacity is not exceeded while
            # we create the env outside the lock (avoids blocking other sessions)
            if self._shared_session_executor is not None:
                executor = self._shared_session_executor
            else:
                executor = ThreadPoolExecutor(max_workers=1)
            self._session_executors[session_id] = executor
            self._sessions[session_id] = None  # placeholder until env is ready

        try:
            # Create environment in the executor thread (outside lock)
            loop = asyncio.get_event_loop()
            env = await loop.run_in_executor(executor, self._env_factory)
        except Exception as e:
            async with self._session_lock:
                if executor is not self._shared_session_executor:
                    executor.shutdown(wait=False)
                self._session_executors.pop(session_id, None)
                self._sessions.pop(session_id, None)
            factory_name = getattr(
                self._env_factory, "__name__", str(self._env_factory)
            )
            raise EnvironmentFactoryError(factory_name) from e

        # Hold the MCP session open for the lifetime of this session,
        # matching the WebSocket path's AsyncExitStack pattern.  This
        # prevents per-request MCP transport teardown/reconnection and
        # preserves FastMCP session state (ctx.set_state / ctx.get_state)
        # across HTTP calls within the same OpenEnv session.
        stack = AsyncExitStack()
        try:
            mcp_session_factory = getattr(env, "mcp_session", None)
            if callable(mcp_session_factory):
                mcp_session_cm = cast(AsyncContextManager[Any], mcp_session_factory())
                await stack.enter_async_context(mcp_session_cm)
        except Exception:
            # MCP transport failed to start — clean up the reserved slot,
            # the env, and the executor so they don't leak permanently
            # against _max_concurrent_envs.
            await stack.aclose()  # best-effort
            async with self._session_lock:
                self._sessions.pop(session_id, None)
                self._session_executors.pop(session_id, None)
                self._session_info.pop(session_id, None)
            await self._cleanup_session_resources(env, executor)
            raise

        async with self._session_lock:
            self._sessions[session_id] = env
            self._session_stacks[session_id] = stack
            now = time.time()
            self._session_info[session_id] = SessionInfo(
                session_id=session_id,
                created_at=current_time,
                last_activity_at=now,
                step_count=0,
                environment_type=type(env).__name__,
            )

        return session_id, env

    async def _destroy_session(self, session_id: str) -> None:
        """
        Destroy a WebSocket session and cleanup resources.

        Args:
            session_id (`str`):
                The session ID to destroy.
        """
        async with self._session_lock:
            env = self._sessions.pop(session_id, None)
            executor = self._session_executors.pop(session_id, None)
            stack = self._session_stacks.pop(session_id, None)
            self._session_info.pop(session_id, None)
            self._session_websocket_attachments.discard(session_id)
            self._session_pending_closes.discard(session_id)

        await self._cleanup_session_resources(env, executor, stack)

    async def _cleanup_session_resources(
        self,
        env: Optional[Environment],
        executor: Optional[ThreadPoolExecutor],
        stack: Optional[AsyncExitStack] = None,
    ) -> None:
        """Close an environment and shut down its executor (best-effort)."""
        # Close the MCP session stack first — this gracefully exits the
        # mcp_session() context (and the underlying FastMCP Client session)
        # before we tear down the environment references.
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                pass  # Best effort cleanup

        # Run close() in the same executor where the env was created
        # This is required for thread-sensitive libraries like Playwright/greenlet
        if env is not None:
            if executor is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(executor, env.close)
                except Exception:
                    # If executor close fails, try direct close as fallback
                    try:
                        env.close()
                    except Exception:
                        pass  # Best effort cleanup
            else:
                try:
                    env.close()
                except Exception:
                    pass  # Best effort cleanup

        if executor is not None and executor is not self._shared_session_executor:
            executor.shutdown(wait=False)

    def _update_session_activity(
        self, session_id: str, increment_step: bool = False
    ) -> None:
        """
        Update session activity timestamp and optionally increment step count.

        Args:
            session_id (`str`):
                The session ID to update.
            increment_step (`bool`, *optional*, defaults to `False`):
                If `True`, increment the step count.
        """
        if session_id in self._session_info:
            self._session_info[session_id].last_activity_at = time.time()
            if increment_step:
                self._session_info[session_id].step_count += 1

    async def _reap_idle_sessions(self) -> None:
        """Background task that periodically destroys sessions idle beyond the timeout."""
        timeout = self._session_idle_timeout_s
        if timeout is None:
            return  # no timeout configured — noop
        interval = max(timeout / 4, 5.0)  # check frequently enough
        while True:
            try:
                await asyncio.sleep(interval)
                now = time.time()
                stale_ids: list[str] = []
                async with self._session_lock:
                    for sid, info in self._session_info.items():
                        if (
                            sid not in self._session_websocket_attachments
                            and now - info.last_activity_at > timeout
                        ):
                            stale_ids.append(sid)
                for sid in stale_ids:
                    # Re-check under lock: activity may have arrived since
                    # the snapshot was taken, making this session active again.
                    # Refresh `now` so slow _destroy_session calls don't cause
                    # subsequent entries to be validated against a stale clock.
                    now = time.time()
                    async with self._session_lock:
                        info = self._session_info.get(sid)
                        if (
                            info is None
                            or sid in self._session_websocket_attachments
                            or (now - info.last_activity_at) <= timeout
                        ):
                            continue
                    await self._destroy_session(sid)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Idle-session reaper encountered an error (will retry): %s",
                    exc,
                )

    def _start_reaper(self) -> None:
        """Start the idle-session reaper if a timeout is configured."""
        if self._session_idle_timeout_s is not None and self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap_idle_sessions())

    def _stop_reaper(self) -> None:
        """Cancel the reaper background task."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get information about a specific session.

        Args:
            session_id (`str`):
                The session ID to query.

        Returns:
            [`SessionInfo`] if the session exists, `None` otherwise.
        """
        return self._session_info.get(session_id)

    async def _run_in_session_executor(
        self, session_id: str, func: Callable[..., Observation], *args, **kwargs
    ) -> Observation:
        """Run a synchronous function in the session's thread pool executor."""
        executor = self._session_executors.get(session_id, self._executor)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    async def _run_mcp_client_operation(
        self,
        mcp_client: Any,
        mcp_session_factory: Any,
        managed_session_id: Optional[str],
        operation: Callable[[], Awaitable[_MCPResult]],
    ) -> _MCPResult:
        """Run an MCP client operation with the appropriate transport lifecycle."""
        if managed_session_id and mcp_client.is_connected():
            return await operation()

        if callable(mcp_session_factory):
            mcp_session_cm = cast(AsyncContextManager[Any], mcp_session_factory())
            async with mcp_session_cm:
                return await operation()

        async with mcp_client:
            return await operation()

    @property
    def active_sessions(self) -> int:
        """Return the number of active WebSocket sessions."""
        return len(self._sessions)

    @property
    def max_concurrent_envs(self) -> int:
        """Return the maximum number of concurrent environments."""
        return self._max_concurrent_envs

    @property
    def is_concurrency_safe(self) -> bool:
        """Return whether the environment is marked as concurrency safe."""
        import inspect

        if inspect.isclass(self._env_factory):
            return getattr(self._env_factory, "SUPPORTS_CONCURRENT_SESSIONS", False)
        else:
            _temp_env = self._env_factory()
            result = getattr(_temp_env, "SUPPORTS_CONCURRENT_SESSIONS", False)
            _temp_env.close()
            del _temp_env
            return result

    @property
    def concurrency_config(self) -> ConcurrencyConfig:
        """Return the concurrency configuration."""
        return self._concurrency_config

    def _factory_produces_harness_env(self) -> bool:
        """Return whether the env factory produces a HarnessEnvironment."""
        import inspect

        # Lazy import: openenv.core.harness imports env_server modules, so a
        # top-level import here would be circular.
        from ..harness.environment import HarnessEnvironment

        if inspect.isclass(self._env_factory):
            return issubclass(self._env_factory, HarnessEnvironment)
        _temp_env = self._env_factory()
        try:
            return isinstance(_temp_env, HarnessEnvironment)
        finally:
            _temp_env.close()

    def _register_harness_route(self, app: FastAPI) -> None:
        """
        Register the production `/harness` WebSocket route (RFC 005).

        Each connection gets its own environment session: connecting resets
        the environment (which starts the harness process and injects tools),
        and each `{"type": "message", "content": ...}` frame runs one
        conversational turn, streamed back as `HarnessEvent` JSON frames
        ending with a `turn_complete` event. Malformed client frames receive a
        recoverable `protocol_error` response without starting a turn; the
        connection remains usable. Terminal failures use `error`.
        """
        # Lazy import to avoid a circular import with openenv.core.harness.
        from ..harness.adapter import HarnessNotRunningError
        from ..harness.events import (
            HarnessClientMessage,
            HarnessEvent,
            HarnessEventType,
            HarnessProtocolError,
        )

        @app.websocket("/harness")
        async def harness_websocket_endpoint(websocket: WebSocket):
            await websocket.accept()

            session_id = None
            session_env = None

            async def send_error(message: str, code: WSErrorCode) -> None:
                error_response = WSErrorResponse(
                    data={"message": message, "code": code}
                )
                await websocket.send_text(error_response.model_dump_json())

            async def send_protocol_error(message: str, code: WSErrorCode) -> None:
                error_response = HarnessProtocolError(
                    data={"message": message, "code": code}
                )
                await websocket.send_text(error_response.model_dump_json())

            async def send_harness_error(message: str) -> None:
                """Emit a terminal ERROR event in the harness event stream."""
                error_event = HarnessEvent(
                    type=HarnessEventType.ERROR,
                    data={"message": message, "recoverable": False},
                )
                await websocket.send_text(error_event.model_dump_json())

            try:
                session_id, session_env = await self._create_session()
                # Protect the live harness from idle reaping and HTTP session
                # close, including startup and turns that emit no events.
                self._session_websocket_attachments.add(session_id)

                async with AsyncExitStack() as stack:
                    mcp_session_factory = getattr(session_env, "mcp_session", None)
                    if callable(mcp_session_factory):
                        mcp_session_cm = cast(
                            AsyncContextManager[Any], mcp_session_factory()
                        )
                        await stack.enter_async_context(mcp_session_cm)

                    # Starts the harness process and injects environment tools
                    await session_env.reset_async()
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "session_started",
                                "data": {
                                    "session_id": session_id,
                                    "harness": session_env.adapter.config.name,
                                },
                            }
                        )
                    )

                    while True:
                        raw_message = await websocket.receive_text()

                        try:
                            message_dict = json.loads(raw_message)
                        except json.JSONDecodeError as e:
                            await send_protocol_error(
                                f"Invalid JSON: {e}", WSErrorCode.INVALID_JSON
                            )
                            continue
                        try:
                            client_message = HarnessClientMessage(**message_dict)
                        except (ValidationError, TypeError) as e:
                            await send_protocol_error(
                                f"Invalid message: {e}",
                                WSErrorCode.VALIDATION_ERROR,
                            )
                            continue

                        self._update_session_activity(session_id, increment_step=True)

                        async def stream_turn(content: str) -> bool:
                            """Stream one turn; True if it ended with TURN_COMPLETE."""
                            saw_terminal = False
                            adapter = session_env.adapter
                            if not await adapter.is_alive():
                                raise HarnessNotRunningError(
                                    "harness process is not running"
                                )
                            async for event in adapter.send_message_streaming(content):
                                await websocket.send_text(event.model_dump_json())
                                # Record progress throughout the turn.
                                self._update_session_activity(session_id)
                                saw_terminal = (
                                    event.type is HarnessEventType.TURN_COMPLETE
                                )
                            return saw_terminal

                        # Bound the turn in wall-clock time, matching what
                        # simulation mode does in HarnessEnvironment._run_turn.
                        # Without this a hung harness holds the session open
                        # forever, and the server sits at capacity.
                        turn_timeout_s = session_env.adapter.config.session_timeout_s
                        try:
                            completed = await asyncio.wait_for(
                                stream_turn(client_message.content),
                                turn_timeout_s,
                            )
                        except asyncio.TimeoutError:
                            await send_harness_error(
                                f"harness turn exceeded {turn_timeout_s} seconds"
                            )
                            break
                        except HarnessNotRunningError:
                            await send_harness_error("harness process is not running")
                            break
                        except Exception:
                            # Harness state after a crash is undefined; end
                            # the session so a reconnect gets a fresh one.
                            # Adapter exceptions can contain credentials or
                            # subprocess output; do not expose them to clients.
                            await send_harness_error("harness turn failed")
                            break

                        if not completed:
                            # send_message() raises HarnessError here; the
                            # socket equivalent is to say so and end the
                            # session, rather than leaving a client that
                            # blocks on the terminal event waiting forever.
                            await send_harness_error(
                                "harness event stream ended without a "
                                "TURN_COMPLETE event"
                            )
                            break

            except WebSocketDisconnect:
                pass
            except SessionCapacityError as e:
                await send_error(str(e), WSErrorCode.CAPACITY_REACHED)
            except EnvironmentFactoryError as e:
                await send_error(str(e), WSErrorCode.FACTORY_ERROR)
            except Exception:
                try:
                    await send_error(
                        "harness session failed", WSErrorCode.SESSION_ERROR
                    )
                except (RuntimeError, WebSocketDisconnect):
                    pass
            finally:
                if session_id:
                    # Release ownership without an await so cancellation cannot
                    # leave a session permanently exempt from idle reaping.
                    self._session_websocket_attachments.discard(session_id)
                    cleanup = asyncio.create_task(self._destroy_session(session_id))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        # ASGI cancellation must not orphan a running harness.
                        # Finish teardown before propagating cancellation, even
                        # when the request's cancel scope cancels us repeatedly.
                        while not cleanup.done():
                            try:
                                await asyncio.shield(cleanup)
                            except asyncio.CancelledError:
                                pass
                        cleanup.result()
                        raise
                try:
                    await websocket.close()
                except (RuntimeError, WebSocketDisconnect):
                    # TestClient raises RuntimeError, real ASGI servers raise
                    # WebSocketDisconnect when the client is already gone.
                    pass

    def register_routes(
        self, app: FastAPI, mode: ServerMode | str = ServerMode.SIMULATION
    ) -> None:
        """
        Register HTTP routes on a FastAPI application.

        Args:
            app (`FastAPI`):
                FastAPI application instance.
            mode (`ServerMode` or `str`, *optional*, defaults to `ServerMode.SIMULATION`):
                Server mode. In production mode, simulation control endpoints (/reset, /step,
                /state) are NOT registered. Only safe endpoints (/health, /schema, /metadata,
                /ws) are available.

        Raises:
            `ValueError`: If `mode` is not a valid `ServerMode` or string equivalent.
        """
        # Convert string to ServerMode enum for backwards compatibility
        if isinstance(mode, str):
            try:
                mode = ServerMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in ServerMode]
                raise ValueError(
                    f"Invalid mode: '{mode}'. Must be one of: {valid_modes}"
                )

        # Wire up idle-session reaper lifecycle via app events
        server_ref = self

        async def _start_session_reaper() -> None:
            server_ref._start_reaper()

        async def _stop_session_reaper() -> None:
            server_ref._stop_reaper()
            if server_ref._shared_session_executor is not None:
                server_ref._shared_session_executor.shutdown(wait=True)

        if not getattr(app.router, "_openenv_reaper_registered", False):
            app.router.on_startup.append(_start_session_reaper)
            app.router.on_shutdown.append(_stop_session_reaper)
            app.router._openenv_reaper_registered = True  # type: ignore[attr-defined]

        # Helper function to handle reset endpoint
        async def reset_handler(
            request: ResetRequest = Body(default_factory=ResetRequest),
        ) -> ResetResponse:
            """Reset endpoint - returns initial observation."""
            _env = self._env_factory()

            try:
                kwargs = request.model_dump(exclude_unset=True)

                is_async = overrides_method(_env.reset_async, Environment.reset_async)

                if is_async:
                    sig = inspect.signature(_env.reset_async)
                else:
                    sig = inspect.signature(_env.reset)
                valid_kwargs = self._get_valid_kwargs(sig, kwargs)

                if is_async:
                    observation = await _env.reset_async(**valid_kwargs)
                else:
                    observation = await self._run_sync_in_thread_pool(
                        _env.reset, **valid_kwargs
                    )
                return ResetResponse(**serialize_observation(observation))
            finally:
                _env.close()

        # Helper function to handle step endpoint
        async def step_handler(request: StepRequest) -> StepResponse:
            """Step endpoint - executes action and returns observation."""
            action_data = request.action

            try:
                action = deserialize_action(action_data, self.action_cls)
            except ValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=e.errors()
                )

            _env = self._env_factory()

            try:
                kwargs = request.model_dump(exclude_unset=True, exclude={"action"})

                is_async = overrides_method(_env.step_async, Environment.step_async)

                if is_async:
                    sig = inspect.signature(_env.step_async)
                else:
                    sig = inspect.signature(_env.step)
                valid_kwargs = self._get_valid_kwargs(
                    sig, kwargs, skip_params={"action"}
                )

                if is_async:
                    observation = await _env.step_async(action, **valid_kwargs)
                else:
                    observation = await self._run_sync_in_thread_pool(
                        _env.step, action, **valid_kwargs
                    )

                return StepResponse(**serialize_observation(observation))
            finally:
                _env.close()

        # Helper function to handle MCP endpoint
        async def mcp_handler(
            request: JsonRpcRequest,
            session_env: Optional[Environment] = None,
            session_id: Optional[str] = None,
        ) -> JsonRpcResponse:
            """
            Handle MCP JSON-RPC requests.

            Supports tools/list and tools/call methods in JSON-RPC 2.0 format,
            plus OpenEnv session lifecycle methods for HTTP MCP (openenv/session/create,
            openenv/session/close).
            """
            method = request.method
            request_id = request.id
            params = request.params
            if not isinstance(params, dict):
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    "Params must be an object",
                    request_id=request_id,
                )

            # OpenEnv extension methods for explicit MCP session management.
            # This enables persistent MCP lifecycles over HTTP /mcp, matching WebSocket semantics.
            if method == "openenv/session/create":
                if session_env is not None and session_id is not None:
                    return JsonRpcResponse.success(
                        result={"session_id": session_id},
                        request_id=request_id,
                    )
                try:
                    created_session_id, _ = await self._create_session()
                except SessionCapacityError as e:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.SERVER_ERROR,
                        str(e),
                        request_id=request_id,
                        data={
                            "active_sessions": e.active_sessions,
                            "max_sessions": e.max_sessions,
                        },
                    )
                except EnvironmentFactoryError as e:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.SERVER_ERROR,
                        str(e),
                        request_id=request_id,
                        data={"factory_name": e.factory_name},
                    )
                return JsonRpcResponse.success(
                    result={"session_id": created_session_id},
                    request_id=request_id,
                )

            if method == "openenv/session/close":
                target_session_id = params.get("session_id")
                if not target_session_id:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        "Invalid params - 'session_id' is required",
                        request_id=request_id,
                    )

                if session_id is not None and target_session_id == session_id:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_REQUEST,
                        "Cannot close active WebSocket-managed session via MCP method",
                        request_id=request_id,
                    )

                async with self._session_lock:
                    if target_session_id in self._session_websocket_attachments:
                        env = _MISSING
                        attached = True
                        self._session_pending_closes.add(target_session_id)
                        executor = None
                        stack = None
                    else:
                        attached = False
                        env = self._sessions.pop(target_session_id, _MISSING)
                    if not attached and env is not _MISSING:
                        executor = self._session_executors.pop(target_session_id, None)
                        stack = self._session_stacks.pop(target_session_id, None)
                        self._session_info.pop(target_session_id, None)
                    elif not attached:
                        executor = None
                        stack = None

                if attached:
                    return JsonRpcResponse.success(
                        result={
                            "session_id": target_session_id,
                            "closed": False,
                            "closing": True,
                        },
                        request_id=request_id,
                    )

                if env is _MISSING:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        f"Unknown session_id: {target_session_id}",
                        request_id=request_id,
                    )

                if env is None:
                    # Session slot reserved but env factory still running;
                    # re-insert the placeholder AND the executor so
                    # _create_session can finish and the executor remains
                    # tracked for eventual shutdown.
                    async with self._session_lock:
                        self._sessions[target_session_id] = None
                        if executor is not None:
                            self._session_executors[target_session_id] = executor
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_REQUEST,
                        f"Session {target_session_id} is still initializing; retry shortly",
                        request_id=request_id,
                    )

                # env/executor/stack cleanup outside the lock
                await self._cleanup_session_resources(env, executor, stack)
                return JsonRpcResponse.success(
                    result={"session_id": target_session_id, "closed": True},
                    request_id=request_id,
                )

            requested_session_id = params.get("session_id")
            managed_session_id = session_id

            # Use provided session environment or create temporary one
            if session_env is not None:
                _env = session_env
                should_close = False
            elif requested_session_id:
                async with self._session_lock:
                    _env = self._sessions.get(requested_session_id, _MISSING)

                if _env is _MISSING:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        f"Unknown session_id: {requested_session_id}",
                        request_id=request_id,
                    )

                if _env is None:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_REQUEST,
                        f"Session {requested_session_id} is still initializing; retry shortly",
                        request_id=request_id,
                    )

                should_close = False
                managed_session_id = requested_session_id
            else:
                _env = self._env_factory()
                should_close = True
            try:
                mcp_client = getattr(_env, "mcp_client", None)
                mcp_server = getattr(_env, "mcp_server", None)
                mcp_session_factory = getattr(_env, "mcp_session", None)

                async def call_mcp_style_step(action: Action) -> Observation:
                    is_async = overrides_method(_env.step_async, Environment.step_async)
                    if is_async:
                        return await _env.step_async(action)
                    if managed_session_id:
                        return await self._run_in_session_executor(
                            managed_session_id,
                            _env.step,
                            action,
                        )
                    return await self._run_sync_in_thread_pool(_env.step, action)

                supports_mcp_style_actions = self.action_cls in {
                    CallToolAction,
                    ListToolsAction,
                }

                if method == McpMethod.TOOLS_LIST:
                    # Check if environment is MCP-enabled
                    if mcp_client is None and mcp_server is None:
                        if supports_mcp_style_actions:
                            observation = await call_mcp_style_step(ListToolsAction())
                            if isinstance(observation, ListToolsObservation):
                                return JsonRpcResponse.success(
                                    result={
                                        "tools": [
                                            tool.model_dump()
                                            for tool in observation.tools
                                        ]
                                    },
                                    request_id=request_id,
                                )
                            return JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INTERNAL_ERROR,
                                "MCP-style tools/list step returned "
                                f"{type(observation).__name__}, expected "
                                "ListToolsObservation",
                                request_id=request_id,
                            )
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INTERNAL_ERROR,
                            "Environment does not support MCP",
                            request_id=request_id,
                        )

                    if mcp_client:
                        tools = await self._run_mcp_client_operation(
                            mcp_client,
                            mcp_session_factory,
                            managed_session_id,
                            mcp_client.list_tools,
                        )

                        return JsonRpcResponse.success(
                            result={
                                "tools": [
                                    t.model_dump()
                                    if hasattr(t, "model_dump")
                                    else dict(t)
                                    for t in tools
                                ]
                            },
                            request_id=request_id,
                        )

                    if mcp_server:
                        tools = []
                        for _tool_name, tool in get_server_tools(mcp_server).items():
                            tools.append(
                                {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "inputSchema": tool.parameters or {},
                                }
                            )
                        return JsonRpcResponse.success(
                            result={"tools": tools},
                            request_id=request_id,
                        )

                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INTERNAL_ERROR,
                        "MCP server not available",
                        request_id=request_id,
                    )

                elif method == McpMethod.TOOLS_CALL:
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})

                    if not tool_name:
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INVALID_PARAMS,
                            "Missing 'name' in params",
                            request_id=request_id,
                        )

                    if mcp_client is None and mcp_server is None:
                        if supports_mcp_style_actions:
                            observation = await call_mcp_style_step(
                                CallToolAction(
                                    tool_name=tool_name,
                                    arguments=arguments,
                                )
                            )
                            if isinstance(observation, CallToolObservation):
                                return JsonRpcResponse.success(
                                    result=_make_json_serializable(observation),
                                    request_id=request_id,
                                )
                            return JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INTERNAL_ERROR,
                                "MCP-style tools/call step returned "
                                f"{type(observation).__name__}, expected "
                                "CallToolObservation",
                                request_id=request_id,
                            )
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INTERNAL_ERROR,
                            "Environment does not support MCP",
                            request_id=request_id,
                        )

                    if mcp_client:
                        result = await self._run_mcp_client_operation(
                            mcp_client,
                            mcp_session_factory,
                            managed_session_id,
                            lambda: mcp_client.call_tool(
                                name=tool_name, arguments=arguments
                            ),
                        )
                    elif mcp_server:
                        server_tools = get_server_tools(mcp_server)
                        if tool_name in server_tools:
                            tool = server_tools[tool_name]
                            if inspect.iscoroutinefunction(tool.fn):
                                result = await tool.fn(**arguments)
                            else:
                                result = tool.fn(**arguments)
                        else:
                            return JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INVALID_PARAMS,
                                f"Tool not found: {tool_name}",
                                request_id=request_id,
                            )
                    else:
                        return JsonRpcResponse.error_response(
                            JsonRpcErrorCode.INTERNAL_ERROR,
                            "MCP server not available",
                            request_id=request_id,
                        )

                    # Ensure result is JSON serializable
                    serializable_result = _make_json_serializable(result)

                    return JsonRpcResponse.success(
                        result=serializable_result,
                        request_id=request_id,
                    )

                else:
                    return JsonRpcResponse.error_response(
                        JsonRpcErrorCode.METHOD_NOT_FOUND,
                        f"Method not found: {method}",
                        request_id=request_id,
                    )

            except Exception as e:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INTERNAL_ERROR,
                    str(e),
                    request_id=request_id,
                )
            finally:
                if managed_session_id:
                    self._update_session_activity(
                        managed_session_id,
                        increment_step=(method == McpMethod.TOOLS_CALL),
                    )
                if should_close:
                    _env.close()

        def _check_env_name(env_name: str) -> None:
            if env_name.lower() != self.env_name.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"unknown environment {env_name!r}",
                )

        def _normalize_split(split: Any) -> Dict[str, Any]:
            if hasattr(split, "model_dump"):
                return cast(Dict[str, Any], split.model_dump())
            if isinstance(split, dict):
                return _make_json_serializable(split)
            if isinstance(split, str):
                split_type = (
                    split if split in {"train", "validation", "test"} else "validation"
                )
                return {"name": split, "type": split_type}
            return {"name": str(split), "type": "validation"}

        async def _call_task_method(method_name: str, *args: Any) -> Any:
            _env = self._env_factory()
            try:
                method = getattr(_env, method_name, None)
                if not callable(method):
                    raise NotImplementedError
                return await _maybe_await(method(*args))
            except NotImplementedError as e:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail=f"{method_name} is not supported for this environment",
                ) from e
            except IndexError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid task index",
                ) from e
            finally:
                _env.close()

        @app.get("/list_environments", tags=["Task API"])
        async def list_environments() -> list[str]:
            return [self.env_name]

        @app.get("/{env_name}/splits", tags=["Task API"])
        async def list_splits(env_name: str) -> list[Dict[str, Any]]:
            _check_env_name(env_name)
            splits = await _call_task_method("list_splits")
            return [_normalize_split(split) for split in splits]

        @app.post("/{env_name}/tasks", tags=["Task API"])
        async def list_tasks(
            env_name: str,
            request: ListTasksRequest,
        ) -> Dict[str, Any]:
            _check_env_name(env_name)
            tasks = await _call_task_method("list_tasks", request.split)
            return {
                "tasks": _make_json_serializable(tasks),
                "env_name": self.env_name,
            }

        @app.post("/{env_name}/num_tasks", tags=["Task API"])
        async def num_tasks(
            env_name: str,
            request: NumTasksRequest,
        ) -> Dict[str, Any]:
            _check_env_name(env_name)
            count = await _call_task_method("num_tasks", request.split)
            return {"num_tasks": count}

        @app.post("/{env_name}/task", tags=["Task API"])
        async def get_task(
            env_name: str,
            request: GetTaskRequest,
        ) -> Dict[str, Any]:
            _check_env_name(env_name)
            task = await _call_task_method("get_task", request.split, request.index)
            return {"task": _make_json_serializable(task)}

        @app.post("/{env_name}/task_range", tags=["Task API"])
        async def get_task_range(
            env_name: str,
            request: GetTaskRangeRequest,
        ) -> Dict[str, Any]:
            _check_env_name(env_name)
            tasks = await _call_task_method(
                "get_task_range", request.split, request.start, request.stop
            )
            return {"tasks": _make_json_serializable(tasks)}

        # Register MCP WebSocket endpoint (available in both production and simulation modes)
        @app.websocket("/mcp")
        async def mcp_websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for MCP JSON-RPC requests.

            Each WebSocket connection gets its own environment instance for MCP operations.
            The client sends JSON-RPC 2.0 requests (tools/list, tools/call) and the server
            responds with JSON-RPC 2.0 responses (result or error).
            """
            await websocket.accept()

            session_id = None
            session_env = None

            try:
                # Create session with dedicated environment
                session_id, session_env = await self._create_session()
                if session_env is None:
                    raise RuntimeError(
                        "Session environment not initialized for MCP websocket"
                    )

                # If environment has an mcp_session context manager, hold it open
                # for the lifetime of the websocket connection

                async with AsyncExitStack() as stack:
                    mcp_session_factory = getattr(session_env, "mcp_session", None)
                    if callable(mcp_session_factory):
                        mcp_session_cm = cast(
                            AsyncContextManager[Any], mcp_session_factory()
                        )
                        await stack.enter_async_context(mcp_session_cm)

                    while True:
                        # Receive message from client
                        raw_message = await websocket.receive_text()

                        try:
                            jsonrpc_dict = json.loads(raw_message)
                            jsonrpc_request = JsonRpcRequest(**jsonrpc_dict)
                        except json.JSONDecodeError as e:
                            error_resp = JsonRpcResponse.error_response(
                                JsonRpcErrorCode.PARSE_ERROR,
                                f"Parse error: {e}",
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            continue
                        except ValidationError as e:
                            error_resp = JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INVALID_REQUEST,
                                f"Invalid request: {e}",
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            continue

                        try:
                            # Call mcp_handler with session environment
                            response = await mcp_handler(
                                jsonrpc_request,
                                session_env=session_env,
                                session_id=session_id,
                            )
                            await websocket.send_text(response.model_dump_json())
                        except Exception as e:
                            error_resp = JsonRpcResponse.error_response(
                                JsonRpcErrorCode.INTERNAL_ERROR,
                                str(e),
                                request_id=jsonrpc_request.id,
                            )
                            await websocket.send_text(error_resp.model_dump_json())

            except WebSocketDisconnect:
                pass
            except SessionCapacityError as e:
                error_resp = JsonRpcResponse.error_response(
                    JsonRpcErrorCode.SERVER_ERROR,
                    str(e),
                    data={
                        "active_sessions": e.active_sessions,
                        "max_sessions": e.max_sessions,
                    },
                )
                await websocket.send_text(error_resp.model_dump_json())
            except EnvironmentFactoryError as e:
                error_resp = JsonRpcResponse.error_response(
                    JsonRpcErrorCode.SERVER_ERROR,
                    str(e),
                    data={"factory_name": e.factory_name},
                )
                await websocket.send_text(error_resp.model_dump_json())
            except Exception as e:
                error_resp = JsonRpcResponse.error_response(
                    JsonRpcErrorCode.SERVER_ERROR,
                    str(e),
                )
                await websocket.send_text(error_resp.model_dump_json())
            finally:
                if session_id:
                    await self._destroy_session(session_id)
                try:
                    await websocket.close()
                except (RuntimeError, WebSocketDisconnect):
                    pass

        # In production mode, a harness environment is exposed directly to
        # clients via a streaming WebSocket (RFC 005).
        if mode == ServerMode.PRODUCTION and self._factory_produces_harness_env():
            self._register_harness_route(app)

        # Register simulation control routes only in simulation mode
        if mode == ServerMode.SIMULATION:

            @app.post(
                "/reset",
                response_model=ResetResponse,
                tags=["Environment Control"],
                summary="Reset the environment",
                description="""
Reset the environment to its initial state and return the first observation.

You can optionally provide a seed for reproducibility and an episode_id for tracking.
                """,
                responses={
                    200: {
                        "description": "Environment reset successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "observation": {"status": "ready", "data": {}},
                                    "reward": None,
                                    "done": False,
                                }
                            }
                        },
                    }
                },
            )
            async def reset(
                request: ResetRequest = Body(default_factory=ResetRequest),
            ) -> ResetResponse:
                return await reset_handler(request)

            @app.post(
                "/step",
                response_model=StepResponse,
                tags=["Environment Control"],
                summary="Execute an action in the environment",
                description="""
Execute an action in the environment and receive the resulting observation.

The action must conform to the environment's action schema, which can be
retrieved from the `/schema` endpoint. If the action is invalid,
the endpoint will return HTTP 422 with detailed validation errors.

The response includes:
- **observation**: The environment's response to the action
- **reward**: Optional reward signal (float or None)
- **done**: Boolean indicating if the episode has terminated
                """,
                responses={
                    200: {
                        "description": "Action executed successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "observation": {"status": "success", "data": {}},
                                    "reward": 1.0,
                                    "done": False,
                                }
                            }
                        },
                    },
                    422: {
                        "description": "Validation error - invalid action format or values",
                        "content": {
                            "application/json": {
                                "example": {
                                    "detail": [
                                        {
                                            "type": "string_too_short",
                                            "loc": ["body", "action", "message"],
                                            "msg": "String should have at least 1 character",
                                            "input": "",
                                        }
                                    ]
                                }
                            }
                        },
                    },
                    500: {
                        "description": "Internal server error during action execution"
                    },
                },
            )
            async def step(request: StepRequest) -> StepResponse:
                return await step_handler(request)

        def get_state_handler() -> State:
            _env = self._env_factory()
            try:
                return _env.state
            finally:
                _env.close()

        def get_metadata_handler() -> EnvironmentMetadata:
            _env = self._env_factory()
            try:
                return _env.get_metadata()
            finally:
                _env.close()

        # Build list of GET endpoints based on mode
        get_endpoints = [
            GetEndpointConfig(
                path="/metadata",
                handler=get_metadata_handler,
                response_model=EnvironmentMetadata,
                tag="Environment Info",
                summary="Get environment metadata",
                description="""
Get metadata about this environment.

Returns information about the environment including name, description,
version, author, and documentation links.
                """,
            ),
            GetEndpointConfig(
                path="/health",
                handler=lambda: HealthResponse(status=HealthStatus.HEALTHY),
                response_model=HealthResponse,
                tag="Health",
                summary="Health check",
                description="Check if the environment server is running and healthy.",
            ),
        ]

        # Only register /state endpoint in simulation mode
        if mode == ServerMode.SIMULATION:
            get_endpoints.insert(
                0,
                GetEndpointConfig(
                    path="/state",
                    handler=get_state_handler,
                    response_model=self.state_cls,
                    tag="State Management",
                    summary="Get current environment state",
                    description="""
Retrieve the current internal state of the environment.

The structure of the state object is defined by the environment's State model.
                    """,
                ),
            )

        register_get_endpoints(app, get_endpoints)

        # Register combined schema endpoint
        @app.get(
            "/schema",
            response_model=SchemaResponse,
            tags=["Schema"],
            summary="Get all JSON schemas",
            description="""
Get JSON schemas for actions, observations, and state in a single response.

Returns a combined schema object containing:
- **action**: JSON schema for actions accepted by this environment
- **observation**: JSON schema for observations returned by step
- **reset_observation**: JSON schema for observations returned by reset
- **state**: JSON schema for environment state objects

This is more efficient than calling individual schema endpoints and provides
all schema information needed to interact with the environment.
            """,
            responses={
                200: {
                    "description": "Combined schemas retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "action": {
                                    "type": "object",
                                    "properties": {"message": {"type": "string"}},
                                },
                                "observation": {
                                    "type": "object",
                                    "properties": {"response": {"type": "string"}},
                                },
                                "state": {
                                    "type": "object",
                                    "properties": {"step_count": {"type": "integer"}},
                                },
                            }
                        }
                    },
                }
            },
        )
        async def get_schemas() -> SchemaResponse:
            """Return all schemas in one response."""
            return SchemaResponse(
                action=self.action_cls.model_json_schema(),
                observation=self.observation_cls.model_json_schema(),
                reset_observation=self.reset_observation_cls.model_json_schema(),
                state=self.state_cls.model_json_schema(),
            )

        # Register MCP endpoint for production mode (direct MCP access)
        @app.post("/mcp")
        async def mcp_endpoint(request_raw: Request) -> Dict[str, Any]:
            """MCP JSON-RPC endpoint for production mode.

            Bypasses step() overhead and provides direct access to MCP tools.
            Supports tools/list and tools/call methods.
            """
            # Parse JSON manually to handle parse errors gracefully
            try:
                body = await request_raw.body()
                request_dict = json.loads(body)
                request = JsonRpcRequest(**request_dict)
            except json.JSONDecodeError:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.PARSE_ERROR
                ).model_dump()
            except ValidationError as e:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_REQUEST,
                    f"Invalid request: {e}",
                ).model_dump()
            except Exception:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.PARSE_ERROR
                ).model_dump()

            response = await mcp_handler(request)
            return response.model_dump()

        # Register WebSocket endpoint for persistent sessions
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for persistent environment sessions.

            By default, each WebSocket connection gets its own environment instance.
            A client can instead attach to an existing HTTP MCP session by passing its
            session ID in the query string. The client sends WSResetMessage,
            WSStepMessage, WSStateMessage, or WSCloseMessage; the server responds with
            WSObservationResponse, WSStateResponse, or WSErrorResponse.
            """
            await websocket.accept()

            session_id = None
            session_env = None
            owns_session = False
            attached_session = False

            try:
                requested_session_id = websocket.query_params.get("session_id")
                if requested_session_id:
                    async with self._session_lock:
                        attached_env = self._sessions.get(
                            requested_session_id, _MISSING
                        )
                        if attached_env is _MISSING:
                            raise RuntimeError(
                                f"Unknown session_id: {requested_session_id}"
                            )
                        if attached_env is None:
                            raise RuntimeError(
                                f"Session {requested_session_id} is still initializing"
                            )
                        if requested_session_id in self._session_websocket_attachments:
                            raise RuntimeError(
                                f"Session {requested_session_id} already has "
                                "an attached WebSocket"
                            )
                        self._session_websocket_attachments.add(requested_session_id)
                        session_id = requested_session_id
                        session_env = attached_env
                        attached_session = True
                    self._update_session_activity(session_id)
                else:
                    session_id, session_env = await self._create_session()
                    owns_session = True

                if session_env is None:
                    raise RuntimeError(
                        "Session environment not initialized for websocket"
                    )

                # Keep MCP session open for entire websocket lifetime
                # (avoids reconnect overhead on every message)

                async with AsyncExitStack() as stack:
                    mcp_session_factory = getattr(session_env, "mcp_session", None)
                    if owns_session and callable(mcp_session_factory):
                        mcp_session_cm = cast(
                            AsyncContextManager[Any], mcp_session_factory()
                        )
                        await stack.enter_async_context(mcp_session_cm)

                    while True:
                        # Receive message from client
                        raw_message = await websocket.receive_text()

                        try:
                            message_dict = json.loads(raw_message)
                        except json.JSONDecodeError as e:
                            error_resp = WSErrorResponse(
                                data={
                                    "message": f"Invalid JSON: {e}",
                                    "code": WSErrorCode.INVALID_JSON,
                                }
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            continue

                        msg_type = message_dict.get("type", "")

                        try:
                            match msg_type:
                                case "reset":
                                    msg = WSResetMessage(**message_dict)

                                    is_async = overrides_method(
                                        session_env.reset_async,
                                        Environment.reset_async,
                                    )

                                    if is_async:
                                        sig = inspect.signature(session_env.reset_async)
                                        valid_kwargs = self._get_valid_kwargs(
                                            sig, msg.data
                                        )
                                        observation = await session_env.reset_async(
                                            **valid_kwargs
                                        )
                                    else:
                                        sig = inspect.signature(session_env.reset)
                                        valid_kwargs = self._get_valid_kwargs(
                                            sig, msg.data
                                        )
                                        observation = (
                                            await self._run_in_session_executor(
                                                session_id,
                                                session_env.reset,
                                                **valid_kwargs,
                                            )
                                        )

                                    self._update_session_activity(session_id)

                                    response = WSObservationResponse(
                                        data=serialize_observation(observation),
                                    )

                                case "step":
                                    msg = WSStepMessage(**message_dict)
                                    action = deserialize_action(
                                        msg.data, self.action_cls
                                    )

                                    is_async = overrides_method(
                                        session_env.step_async,
                                        Environment.step_async,
                                    )

                                    if is_async:
                                        observation = await session_env.step_async(
                                            action
                                        )
                                    else:
                                        observation = (
                                            await self._run_in_session_executor(
                                                session_id, session_env.step, action
                                            )
                                        )

                                    self._update_session_activity(
                                        session_id, increment_step=True
                                    )

                                    response = WSObservationResponse(
                                        data=serialize_observation(observation)
                                    )

                                case "state":
                                    msg = WSStateMessage(**message_dict)
                                    state = session_env.state
                                    if hasattr(state, "model_dump"):
                                        state_data = state.model_dump()
                                    else:
                                        state_data = dict(state) if state else {}

                                    response = WSStateResponse(data=state_data)

                                case "close":
                                    msg = WSCloseMessage(**message_dict)
                                    break

                                case "mcp":
                                    msg = WSMCPMessage(**message_dict)
                                    try:
                                        rpc_request = JsonRpcRequest(**msg.data)
                                    except (ValidationError, Exception) as e:
                                        rpc_response = JsonRpcResponse.error_response(
                                            JsonRpcErrorCode.INVALID_REQUEST,
                                            f"Invalid request: {e}",
                                        )
                                    else:
                                        rpc_response = await mcp_handler(
                                            rpc_request,
                                            session_env=session_env,
                                            session_id=session_id,
                                        )
                                    response = WSMCPResponse(
                                        data=rpc_response.model_dump()
                                    )

                                case _:
                                    response = WSErrorResponse(
                                        data={
                                            "message": f"Unknown message type: {msg_type}",
                                            "code": WSErrorCode.UNKNOWN_TYPE,
                                        }
                                    )

                            await websocket.send_text(response.model_dump_json())

                        except ValidationError as e:
                            error_resp = WSErrorResponse(
                                data={
                                    "message": "Invalid message",
                                    "code": WSErrorCode.VALIDATION_ERROR,
                                    "errors": e.errors(),
                                }
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                        except Exception as e:
                            error_resp = WSErrorResponse(
                                data={
                                    "message": str(e),
                                    "code": WSErrorCode.EXECUTION_ERROR,
                                }
                            )
                            await websocket.send_text(error_resp.model_dump_json())

            except WebSocketDisconnect:
                pass
            except SessionCapacityError as e:
                error_resp = WSErrorResponse(
                    data={
                        "message": str(e),
                        "code": WSErrorCode.CAPACITY_REACHED,
                        "active_sessions": e.active_sessions,
                        "max_sessions": e.max_sessions,
                    }
                )
                await websocket.send_text(error_resp.model_dump_json())
            except EnvironmentFactoryError as e:
                error_resp = WSErrorResponse(
                    data={
                        "message": str(e),
                        "code": WSErrorCode.FACTORY_ERROR,
                        "factory_name": e.factory_name,
                    }
                )
                await websocket.send_text(error_resp.model_dump_json())
            except Exception as e:
                error_resp = WSErrorResponse(
                    data={"message": str(e), "code": WSErrorCode.SESSION_ERROR}
                )
                await websocket.send_text(error_resp.model_dump_json())
            finally:
                if attached_session and session_id:
                    # Do not await the lock before releasing ownership: a task
                    # cancelled while waiting would leave this session
                    # permanently exempt from idle reaping.
                    self._session_websocket_attachments.discard(session_id)
                    if session_id in self._session_pending_closes:
                        await asyncio.shield(self._destroy_session(session_id))
                elif owns_session and session_id:
                    await self._destroy_session(session_id)
                try:
                    await websocket.close()
                except (RuntimeError, WebSocketDisconnect):
                    pass


def create_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
    gradio_builder: Optional[Callable[..., Any]] = None,
    custom_tab_name: str = "Custom",
    custom_tab_primary: bool = False,
    show_default_tab: bool = True,
    title_override: Optional[str] = None,
    state_cls: Type[State] = State,
    *,
    mode: Optional[ServerMode | str] = None,
    reset_observation_cls: Optional[Type[Observation]] = None,
) -> FastAPI:
    """
    Create a FastAPI application with or without web interface.

    This function creates a FastAPI app with the web interface enabled by default,
    including README integration for better user experience.

    Args:
        env (`Callable[[], Environment]`):
            Environment factory (callable) that creates new instances.
        action_cls (`Type[Action]`):
            The Action subclass this environment expects.
        observation_cls (`Type[Observation]`):
            The Observation subclass returned by step.
        reset_observation_cls (`Type[Observation]`, *optional*):
            The reset observation model published in `/schema`. Defaults to
            `observation_cls`; declare a distinct model when reset differs.
        env_name (`str`, *optional*):
            Environment name for README loading.
        max_concurrent_envs (`int`, *optional*):
            Maximum concurrent WebSocket sessions. Mutually exclusive with
            `concurrency_config`.
        concurrency_config (`ConcurrencyConfig`, *optional*):
            Advanced concurrency settings. Mutually exclusive with
            `max_concurrent_envs`.
        gradio_builder (`Callable`, *optional*):
            Callable to build a custom Gradio UI at /web. Signature:
            `(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md) -> gr.Blocks`.
            When `None`, the default Gradio app is used.
        custom_tab_name (`str`, *optional*, defaults to `"Custom"`):
            Label for the env-specific tab when `gradio_builder` is provided.
        custom_tab_primary (`bool`, *optional*, defaults to `False`):
            When `True`, the env-specific tab is active first; the auto-generated
            Playground becomes secondary.
        show_default_tab (`bool`, *optional*, defaults to `True`):
            When `False`, mount the env's `gradio_builder` output alone (no
            auto-generated Playground, no tab chrome). Only meaningful when
            `gradio_builder` is provided.
        title_override (`str`, *optional*):
            If set, used as the Gradio app title instead of the default
            `"OpenEnv Agentic Environment: {name}"`.
        state_cls (`Type[State]`, *optional*, defaults to `State`):
            The `State` subclass this environment reports, used for the `/state`
            response model and the `state` entry of `/schema`.
        mode (`ServerMode` or `str`, *optional*):
            Server mode. When `None`, resolved from the `OPENENV_MODE`
            environment variable, defaulting to simulation.

    Returns:
        `FastAPI` application instance with or without web interface and README integration.
    """
    # Check if web interface should be enabled
    # This can be controlled via environment variable or build argument
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    if enable_web:
        # Gradio-based web UI (gradio is a core dependency)
        from .web_interface import create_web_interface_app

        return create_web_interface_app(
            cast(Any, env),
            action_cls,
            observation_cls,
            env_name,
            max_concurrent_envs,
            concurrency_config,
            state_cls=state_cls,
            reset_observation_cls=reset_observation_cls,
            gradio_builder=gradio_builder,
            custom_tab_name=custom_tab_name,
            custom_tab_primary=custom_tab_primary,
            show_default_tab=show_default_tab,
            title_override=title_override,
            mode=mode,
        )
    else:
        # Use standard FastAPI app without web interface
        return create_fastapi_app(
            env,
            action_cls,
            observation_cls,
            max_concurrent_envs,
            concurrency_config,
            env_name=env_name,
            state_cls=state_cls,
            reset_observation_cls=reset_observation_cls,
            mode=mode,
        )


def create_fastapi_app(
    env: Callable[[], Environment],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    max_concurrent_envs: Optional[int] = None,
    concurrency_config: Optional[ConcurrencyConfig] = None,
    env_name: Optional[str] = None,
    state_cls: Type[State] = State,
    *,
    mode: Optional[ServerMode | str] = None,
    reset_observation_cls: Optional[Type[Observation]] = None,
) -> FastAPI:
    """
    Create a FastAPI application with comprehensive documentation.

    Args:
        env (`Callable[[], Environment]`):
            Environment factory (callable) that creates new instances.
        action_cls (`Type[Action]`):
            The Action subclass this environment expects.
        observation_cls (`Type[Observation]`):
            The Observation subclass returned by step.
        reset_observation_cls (`Type[Observation]`, *optional*):
            The reset observation model published in `/schema`. Defaults to
            `observation_cls`; declare a distinct model when reset differs.
        max_concurrent_envs (`int`, *optional*):
            Maximum concurrent WebSocket sessions. Mutually exclusive with
            `concurrency_config`.
        concurrency_config (`ConcurrencyConfig`, *optional*):
            Advanced concurrency settings. Mutually exclusive with
            `max_concurrent_envs`.
        env_name (`str`, *optional*):
            Optional environment name for task/split endpoints.
        state_cls (`Type[State]`, *optional*, defaults to `State`):
            The `State` subclass this environment reports, used for the `/state`
            response model and the `state` entry of `/schema`.
        mode (`ServerMode` or `str`, *optional*):
            Server mode. When `None`, resolved from the `OPENENV_MODE`
            environment variable, defaulting to simulation.

    Returns:
        `FastAPI` application instance.
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="OpenEnv Environment HTTP API",
        version="1.0.0",
        description="""
# OpenEnv Environment HTTP API

HTTP API for interacting with OpenEnv environments through a standardized interface.

## Features

* **Environment Reset**: Initialize or restart episodes
* **Action Execution**: Send actions and receive observations
* **State Inspection**: Query current environment state
* **Schema Access**: Retrieve JSON schemas for actions and observations

## Workflow

1. Call `/reset` to start a new episode and get initial observation
2. Call `/step` repeatedly with actions to interact with environment
3. Episode ends when observation returns `done: true`
4. Call `/state` anytime to inspect current environment state

## Documentation

* **Swagger UI**: Available at `/docs`
* **ReDoc**: Available at `/redoc`
* **OpenAPI Schema**: Available at `/openapi.json`
        """,
        openapi_tags=[
            {
                "name": "Environment Control",
                "description": "Core operations for environment interaction (reset, step)",
            },
            {
                "name": "State Management",
                "description": "Operations for inspecting environment state",
            },
            {
                "name": "Environment Info",
                "description": "Information about the environment",
            },
            {
                "name": "Schema",
                "description": "JSON Schema endpoints for actions, observations, and state",
            },
            {"name": "Health", "description": "Service health and status checks"},
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "OpenEnv Team",
            "url": "https://github.com/huggingface/OpenEnv",
        },
        license_info={
            "name": "BSD-3-Clause",
            "url": "https://github.com/huggingface/OpenEnv/blob/main/LICENSE",
        },
    )

    server = HTTPEnvServer(
        env,
        action_cls,
        observation_cls,
        max_concurrent_envs,
        concurrency_config=concurrency_config,
        env_name=env_name,
        state_cls=state_cls,
        reset_observation_cls=reset_observation_cls,
    )
    if mode is None:
        mode = os.environ.get("OPENENV_MODE", ServerMode.SIMULATION.value)
    server.register_routes(app, mode=mode)
    return app
