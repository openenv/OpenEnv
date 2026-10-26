# SPDX-License-Identifier: BSD-3-Clause

"""Environment wrapper for turn-based external agentic harnesses (RFC 005)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

from ..env_server.mcp_environment import MCPEnvironment
from ..env_server.mcp_types import CallToolAction, ListToolsAction, Tool
from ..env_server.types import Action, Observation, State
from ..utils import run_async_safely
from .adapter import (
    AgenticHarnessAdapter,
    HarnessError,
    HarnessNotRunningError,
    HarnessTurnTimeoutError,
)
from .bridge import build_bridge_server, HarnessMCPBridge
from .events import events_to_metadata, HarnessEvent, HarnessEventType
from .tools import resolve_tool_conflicts

try:
    from fastmcp import FastMCP
except ModuleNotFoundError:  # pragma: no cover - fastmcp is a core dependency
    FastMCP = None

logger = logging.getLogger(__name__)


class HarnessAction(Action):
    """
    Action carrying one user message for a conversational harness turn.

    Args:
        message (`str`):
            The user message for this turn.
    """

    message: str


class HarnessEnvironment(MCPEnvironment):
    """
    Environment that wraps an external turn-based agentic harness.

    In simulation mode, `reset()` starts a fresh harness process and
    conversation, and each `step()` with a [`~openenv.core.harness.environment.HarnessAction`]
    is one conversational turn: the harness runs its internal ReAct loop and
    the response is returned as an observation. The harness keeps conversation
    context across turns; the training loop controls episode boundaries.
    MCP actions (`ListToolsAction`, `CallToolAction`) keep their standard
    routing so orchestrators can inspect and invoke domain tools directly.

    Environment MCP tools are conflict-resolved against the adapter's built-in
    tool names and injected into the harness before it starts. Rubrics run
    after each turn completes, outside the harness's control loop.

    Factories used with `HTTPEnvServer` must construct a fresh adapter per
    environment instance — adapters own a single harness process and must not
    be shared. Constructing the environment starts nothing; the harness
    process and any tool bridge start on `reset()`.

    Args:
        adapter ([`~openenv.core.harness.adapter.AgenticHarnessAdapter`]):
            Adapter owning the harness process lifecycle.
        mcp (`FastMCP`, *optional*):
            Environment-specific MCP tools to inject into the harness. When
            `None`, an empty internal server is used and nothing is injected.
        rubric ([`~openenv.core.rubrics.Rubric`], *optional*):
            Reward rubric applied to each turn's observation.
        transform (`Transform`, *optional*):
            Optional observation transform (inherited from `Environment`).
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        adapter: AgenticHarnessAdapter,
        mcp: Optional[Any] = None,
        rubric: Optional[Any] = None,
        transform: Optional[Any] = None,
    ):
        if mcp is None:
            if FastMCP is None:  # pragma: no cover - fastmcp is a core dependency
                raise ModuleNotFoundError(
                    "fastmcp is required to construct a HarnessEnvironment"
                )
            mcp = FastMCP("harness-env-tools")
        super().__init__(mcp, transform=transform)
        self.rubric = rubric
        self.adapter = adapter
        self._state = State(episode_id=None, step_count=0)
        self._trajectory: list[HarnessEvent] = []
        self._episode_active = False
        self._closed = False
        self._bridge: Optional[HarnessMCPBridge] = None

    @property
    def state(self) -> State:
        """Current episode state."""
        return self._state

    @property
    def trajectory(self) -> list[HarnessEvent]:
        """Full event trajectory across all turns in the current episode."""
        return list(self._trajectory)

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Start a fresh harness process and conversation.

        Stops any running harness, injects the environment's conflict-resolved
        MCP tools, and starts the harness in its configured working directory.

        Args:
            seed (`int`, *optional*):
                Unused; accepted for interface compatibility.
            episode_id (`str`, *optional*):
                Episode identifier. A UUID is generated when `None`.

        Returns:
            `Observation` with `done=False` and metadata listing the names of
            the injected tools.
        """
        self._episode_active = False
        # Unconditional: stop() is contractually idempotent, and a harness that
        # died on its own still holds reapable resources (pipes, reader threads,
        # an unwaited process) that is_alive() reports nothing about.
        bridge_start: Optional[asyncio.Task[str]] = None
        bridge_stop_failed = False
        try:
            await self.adapter.stop()
            if self._bridge is not None:
                # A failed shutdown must not be replaced by another live bridge.
                try:
                    await asyncio.to_thread(self._bridge.stop)
                except Exception:
                    bridge_stop_failed = True
                    raise

            tools = await self._collect_injectable_tools()
            resolved = resolve_tool_conflicts(tools, self.adapter.BUILTIN_TOOL_NAMES)

            bridge_url: Optional[str] = None
            if resolved:
                # The bridge must serve the tools under the names we inject, so a
                # tool renamed by conflict resolution stays callable.
                renames = {
                    new.name: old.name
                    for new, old in zip(resolved, tools)
                    if new.name != old.name
                }
                served = await build_bridge_server(self._require_mcp_server(), renames)
                self._bridge = HarnessMCPBridge(served)
                bridge_start = asyncio.create_task(
                    asyncio.to_thread(self._bridge.start)
                )
                bridge_url = await asyncio.shield(bridge_start)

            await self.adapter.inject_tools(resolved, bridge_url)
            await self.adapter.start(self.adapter.config.working_directory)

            self._state = State(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
            )
            self._trajectory = []
            if self.rubric is not None:
                await self._reset_rubric_async()
            self._episode_active = True

            observation = Observation(
                done=False,
                reward=0.0,
                metadata={
                    "episode_id": self._state.episode_id,
                    "injected_tools": [tool.name for tool in resolved],
                },
            )
            return self._apply_transform(observation)
        except BaseException:
            # The adapter is already stopped when bridge shutdown fails. Keep
            # the bridge for a later retry without spending its timeout twice.
            if not bridge_stop_failed:
                await self._cleanup_episode(bridge_start)
            raise

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Sync facade over `reset_async` for non-async callers."""
        return run_async_safely(
            self.reset_async(seed=seed, episode_id=episode_id, **kwargs)
        )

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute one action: a conversational turn or an MCP action.

        Args:
            action (`Action`):
                A [`~openenv.core.harness.environment.HarnessAction`] runs one
                conversational turn; `ListToolsAction`/`CallToolAction` keep
                their standard MCP routing.
            timeout_s (`float`, *optional*):
                Wall-clock budget for this turn. Defaults to the adapter
                config's `session_timeout_s`.

        Returns:
            `Observation` with the harness response and turn events in
            metadata.
        """
        if isinstance(action, (ListToolsAction, CallToolAction)):
            return await super().step_async(action, timeout_s=timeout_s, **kwargs)
        if isinstance(action, HarnessAction):
            return await self._run_turn(action, timeout_s=timeout_s)
        raise TypeError(
            "HarnessEnvironment only accepts HarnessAction, ListToolsAction, "
            f"or CallToolAction; got {type(action).__name__}"
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Sync facade for non-MCP actions; runs one conversational turn."""
        if isinstance(action, HarnessAction):
            return run_async_safely(self._run_turn(action, timeout_s=timeout_s))
        raise TypeError(
            "HarnessEnvironment only accepts HarnessAction, ListToolsAction, "
            f"or CallToolAction; got {type(action).__name__}"
        )

    async def _run_turn(
        self,
        action: HarnessAction,
        timeout_s: Optional[float] = None,
    ) -> Observation:
        """Send one message to the harness and build the turn's observation."""
        if not self._episode_active:
            raise HarnessNotRunningError(
                "No active episode; call reset() before step()"
            )
        try:
            if not await self.adapter.is_alive():
                return await self._terminal_error_observation(
                    "harness process is not running",
                    error_type="harness_crashed",
                )

            timeout = (
                timeout_s
                if timeout_s is not None
                else (self.adapter.config.session_timeout_s)
            )
            try:
                harness_response = await asyncio.wait_for(
                    self.adapter.send_message(action.message), timeout
                )
            except asyncio.TimeoutError:
                return await self._terminal_error_observation(
                    f"harness turn exceeded {timeout} seconds",
                    error_type="turn_timeout",
                )
            except HarnessTurnTimeoutError as exc:
                # Must precede HarnessError: an adapter that raises the dedicated
                # timeout exception means a timeout, not a crash.
                return await self._terminal_error_observation(
                    str(exc),
                    error_type="turn_timeout",
                )
            except HarnessError as exc:
                return await self._terminal_error_observation(
                    str(exc),
                    error_type="harness_crashed",
                )

            self._trajectory.extend(harness_response.events)
            self._state.step_count += 1

            observation = Observation(
                done=harness_response.done,
                reward=0.0,
                metadata={
                    "response": harness_response.response,
                    "turn_events": events_to_metadata(harness_response.events),
                    "turn_number": self._state.step_count,
                },
            )
            if self.rubric is not None:
                observation.reward = await self._apply_rubric_async(action, observation)
            return self._apply_transform(observation)
        except BaseException:
            # A failed or interrupted turn cannot safely resume. Preserve the
            # original failure for the caller after releasing resources.
            await self._cleanup_episode()
            raise

    async def _cleanup_episode(
        self, bridge_start: Optional[asyncio.Task[str]] = None
    ) -> None:
        """Finish teardown even when the caller is cancelled repeatedly."""
        self._episode_active = False

        async def cleanup() -> None:
            try:
                if bridge_start is not None:
                    # Cancelling to_thread does not stop its worker. Startup
                    # must finish before teardown can safely stop the bridge.
                    try:
                        await bridge_start
                    except Exception:
                        pass
            finally:
                try:
                    await self.adapter.stop()
                except Exception:
                    pass
                finally:
                    await self._stop_bridge()

        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            # ASGI cancel scopes can cancel every await. Keep the cleanup in
            # its own task and join it before allowing cancellation to escape.
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    pass
            cleanup_task.result()
            raise

    async def _terminal_error_observation(
        self, message: str, error_type: str
    ) -> Observation:
        """Stop the harness and return a terminal observation for a failed turn."""
        await self._cleanup_episode()
        error_event = HarnessEvent(
            type=HarnessEventType.ERROR,
            data={"message": message, "recoverable": False},
        )
        self._trajectory.append(error_event)
        observation = Observation(
            done=True,
            reward=0.0,
            metadata={
                "error": message,
                "error_type": error_type,
                "turn_events": events_to_metadata([error_event]),
            },
        )
        return self._apply_transform(observation)

    async def _collect_injectable_tools(self) -> list[Tool]:
        """
        Enumerate the environment's MCP tools for injection into the harness.

        Mode-specific tools registered with `MCPEnvironment.tool(mode=...)` are
        excluded: they are tracked by the environment rather than registered on
        the FastMCP server, so the bridge cannot serve them. Advertising a tool
        the harness then cannot call is worse than not advertising it, so they
        are dropped with a warning. Supporting them needs a design decision
        about mode semantics inside a harness turn (follow-up).
        """
        list_tools_observation = await self._async_handle_list_tools()
        error = list_tools_observation.metadata.get("error")
        if error:
            raise HarnessError(f"Failed to enumerate environment tools: {error}")

        tools = list(list_tools_observation.tools)
        mode_specific = [t for t in tools if t.name in self._mode_tool_schemas]
        if mode_specific:
            logger.warning(
                "Not injecting mode-specific tools into harness %r (the tool "
                "bridge cannot serve them): %s",
                self.adapter.config.name,
                ", ".join(sorted(t.name for t in mode_specific)),
            )
            tools = [t for t in tools if t.name not in self._mode_tool_schemas]
        return tools

    async def _stop_bridge(self) -> None:
        """Best-effort stop of the tool bridge, off the event loop."""
        if self._bridge is None:
            return
        try:
            await asyncio.to_thread(self._bridge.stop)
        except Exception:
            pass

    def close(self) -> None:
        """Stop the harness process and release environment resources."""
        if self._closed and self._bridge is None:
            return
        self._closed = True
        self._episode_active = False
        try:
            run_async_safely(self.adapter.stop())
        except Exception:
            pass
        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception:
                pass
            else:
                self._bridge = None
        super().close()


__all__ = ["HarnessAction", "HarnessEnvironment"]
