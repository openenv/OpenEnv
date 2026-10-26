# SPDX-License-Identifier: BSD-3-Clause

"""Tests for HarnessEnvironment (RFC 005 turn-based wrapper)."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

import pytest
from fastmcp import FastMCP
from openenv.core.env_server._utils import overrides_method
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import ListToolsAction, Tool
from openenv.core.env_server.serialization import serialize_observation
from openenv.core.env_server.types import Action
from openenv.core.harness import (
    AgenticHarnessAdapter,
    HarnessAction,
    HarnessConfig,
    HarnessEnvironment,
    HarnessError,
    HarnessEvent,
    HarnessEventType,
    HarnessNotRunningError,
    HarnessTurnTimeoutError,
)
from openenv.core.rubrics import Rubric


def turn_events(response: str, done: bool = False) -> list[HarnessEvent]:
    return [
        HarnessEvent(type=HarnessEventType.TOOL_CALL, data={"tool_name": "shell"}),
        HarnessEvent(
            type=HarnessEventType.TURN_COMPLETE,
            data={"response": response, "done": done},
        ),
    ]


class FakeAdapter(AgenticHarnessAdapter):
    """Recording fake with failure knobs."""

    BUILTIN_TOOL_NAMES = frozenset({"read_file"})

    def __init__(self, config: Optional[HarnessConfig] = None):
        super().__init__(config or HarnessConfig(name="fake", command=["fake"]))
        self.calls: list[str] = []
        self.injected_tools: Optional[list[Tool]] = None
        self.injected_bridge_url: Optional[str] = "unset"
        self.alive = False
        self.fail_on_start = False
        self.fail_on_send = False
        self.raise_turn_timeout = False
        self.send_delay_s = 0.0
        self.scripted_turns: list[list[HarnessEvent]] = []

    async def start(self, working_directory: str) -> None:
        self.calls.append("start")
        if self.fail_on_start:
            raise HarnessError("boom on start")
        self.alive = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self.alive = False

    async def inject_tools(self, tools, bridge_url: Optional[str] = None) -> None:
        self.calls.append("inject_tools")
        self.injected_tools = list(tools)
        self.injected_bridge_url = bridge_url

    async def send_message_streaming(self, message: str) -> AsyncIterator[HarnessEvent]:
        self.calls.append("send")
        if self.send_delay_s:
            await asyncio.sleep(self.send_delay_s)
        if self.raise_turn_timeout:
            raise HarnessTurnTimeoutError("adapter timed the turn out itself")
        if self.fail_on_send:
            raise HarnessError("harness exploded mid-turn")
        events = (
            self.scripted_turns.pop(0)
            if self.scripted_turns
            else turn_events("default response")
        )
        for event in events:
            yield event

    async def is_alive(self) -> bool:
        return self.alive


class FakeBridge:
    """In-process stand-in for HarnessMCPBridge (real one tested separately)."""

    instances: list["FakeBridge"] = []

    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        self.started = 0
        self.stopped = 0
        self.fail_on_stop = False
        FakeBridge.instances.append(self)

    def start(self, timeout_s: float = 10.0) -> str:
        self.started += 1
        return "http://127.0.0.1:9/mcp"

    def stop(self, timeout_s: float = 5.0) -> None:
        self.stopped += 1
        if self.fail_on_stop:
            raise HarnessError("bridge shutdown timed out")


@pytest.fixture(autouse=True)
def fake_bridge(monkeypatch):
    FakeBridge.instances = []
    monkeypatch.setattr("openenv.core.harness.environment.HarnessMCPBridge", FakeBridge)
    return FakeBridge


class SpyRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.seen: list[tuple] = []
        self.resets = 0

    def forward(self, action, observation) -> float:
        self.seen.append((action, observation))
        return 0.75

    def reset(self) -> None:
        self.resets += 1


def make_env(**kwargs) -> tuple[HarnessEnvironment, FakeAdapter]:
    adapter = FakeAdapter()
    mcp = kwargs.pop("mcp", "default")
    if mcp == "default":
        mcp = FastMCP("domain")

        @mcp.tool
        def query_db(sql: str) -> str:
            """Run a SQL query."""
            return sql

        @mcp.tool
        def read_file(path: str) -> str:
            """Read a task file."""
            return path

    env = HarnessEnvironment(adapter=adapter, mcp=mcp, **kwargs)
    return env, adapter


class TestReset:
    async def test_inject_before_start_with_conflict_resolution(self):
        env, adapter = make_env()
        obs = await env.reset_async()

        assert adapter.calls == ["stop", "inject_tools", "start"]
        names = sorted(t.name for t in adapter.injected_tools)
        assert names == ["env_read_file", "query_db"]
        assert sorted(obs.metadata["injected_tools"]) == names
        assert obs.done is False
        assert env.state.step_count == 0
        assert env.state.episode_id
        assert adapter.injected_bridge_url == "http://127.0.0.1:9/mcp"
        assert len(FakeBridge.instances) == 1
        assert FakeBridge.instances[0].started == 1

    async def test_reset_stops_live_adapter_first(self):
        env, adapter = make_env()
        await env.reset_async()
        adapter.calls.clear()
        await env.reset_async()
        assert adapter.calls == ["stop", "inject_tools", "start"]

    async def test_reset_does_not_replace_bridge_after_shutdown_failure(self):
        env, adapter = make_env()
        await env.reset_async()
        bridge = env._bridge
        bridge.fail_on_stop = True
        stop_count = bridge.stopped
        adapter.calls.clear()

        with pytest.raises(HarnessError, match="bridge shutdown timed out"):
            await env.reset_async()

        assert env._bridge is bridge
        assert bridge.stopped == stop_count + 1
        assert FakeBridge.instances == [bridge]
        assert "inject_tools" not in adapter.calls
        assert "start" not in adapter.calls
        assert adapter.alive is False
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="go"))

        bridge.fail_on_stop = False
        await env.reset_async()
        assert env._bridge is not bridge
        assert adapter.alive is True
        env.close()

    async def test_cancelled_bridge_stop_still_cleans_up(self, monkeypatch):
        env, adapter = make_env()
        await env.reset_async()
        bridge = env._bridge
        stop = bridge.stop

        def cancel_once():
            stop()
            if bridge.stopped == 1:
                raise asyncio.CancelledError()

        monkeypatch.setattr(bridge, "stop", cancel_once)
        with pytest.raises(asyncio.CancelledError):
            await env.reset_async()

        assert bridge.stopped == 2
        assert env._bridge is bridge
        assert FakeBridge.instances == [bridge]
        assert adapter.alive is False
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="go"))
        env.close()

    async def test_reset_clears_trajectory_and_resets_rubric(self):
        rubric = SpyRubric()
        env, adapter = make_env(rubric=rubric)
        await env.reset_async()
        await env.step_async(HarnessAction(message="turn 1"))
        assert env.trajectory
        await env.reset_async(episode_id="ep-2")
        assert env.trajectory == []
        assert env.state.episode_id == "ep-2"
        assert env.state.step_count == 0
        assert rubric.resets == 2

    async def test_no_mcp_tools_injects_empty_list_and_no_bridge(self):
        env, adapter = make_env(mcp=None)
        await env.reset_async()
        assert adapter.injected_tools == []
        assert adapter.injected_bridge_url is None
        assert FakeBridge.instances == []

    @pytest.mark.parametrize("async_rubric", [False, True])
    async def test_rubric_reset_failure_stops_resources_and_allows_retry(
        self, async_rubric
    ):
        failure = RuntimeError("rubric reset failed")

        class FailingRubric(SpyRubric):
            fail = True

            def reset(self):
                assert adapter.alive
                assert FakeBridge.instances[-1].started == 1
                if self.fail:
                    raise failure
                super().reset()

        class AsyncFailingRubric(FailingRubric):
            async def reset_async(self):
                self.reset()

        rubric = AsyncFailingRubric() if async_rubric else FailingRubric()
        env, adapter = make_env(rubric=rubric)
        with pytest.raises(RuntimeError) as caught:
            await env.reset_async()

        assert caught.value is failure
        assert adapter.alive is False
        assert adapter.calls[-1] == "stop"
        assert FakeBridge.instances[0].stopped >= 1
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="after failed reset"))

        rubric.fail = False
        try:
            obs = await env.reset_async()
            assert obs.done is False
            assert adapter.alive
            assert rubric.resets == 1
            assert len(FakeBridge.instances) == 2
            obs = await env.step_async(HarnessAction(message="retry"))
            assert obs.metadata["response"] == "default response"
        finally:
            env.close()

    async def test_start_failure_leaves_episode_inactive(self):
        env, adapter = make_env()
        adapter.fail_on_start = True
        with pytest.raises(HarnessError):
            await env.reset_async()
        assert FakeBridge.instances[0].stopped >= 1
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="hi"))


class TestStep:
    async def test_turn_updates_state_and_trajectory(self):
        env, adapter = make_env()
        adapter.scripted_turns = [turn_events("first"), turn_events("second", True)]
        await env.reset_async()

        obs1 = await env.step_async(HarnessAction(message="fix the bug"))
        assert obs1.done is False
        assert obs1.metadata["response"] == "first"
        assert obs1.metadata["turn_number"] == 1
        assert env.state.step_count == 1

        obs2 = await env.step_async(HarnessAction(message="tests still fail"))
        assert obs2.done is True
        assert obs2.metadata["response"] == "second"
        assert env.state.step_count == 2
        assert len(env.trajectory) == 4

    async def test_observation_serializes_to_json(self):
        env, adapter = make_env()
        await env.reset_async()
        obs = await env.step_async(HarnessAction(message="go"))
        payload = serialize_observation(obs)
        assert json.loads(json.dumps(payload))["metadata"]["turn_events"]

    async def test_rubric_applied_after_turn(self):
        rubric = SpyRubric()
        env, adapter = make_env(rubric=rubric)
        await env.reset_async()
        action = HarnessAction(message="go")
        obs = await env.step_async(action)
        assert obs.reward == 0.75
        seen_action, seen_obs = rubric.seen[0]
        assert seen_action is action
        assert seen_obs.metadata["response"] == "default response"

    async def test_step_before_reset_raises(self):
        env, _ = make_env()
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="hi"))

    async def test_dead_adapter_returns_terminal_error_observation(self):
        env, adapter = make_env()
        await env.reset_async()
        adapter.alive = False
        obs = await env.step_async(HarnessAction(message="hi"))
        assert obs.done is True
        assert obs.metadata["error_type"] == "harness_crashed"

    async def test_mid_turn_crash_stops_adapter(self):
        env, adapter = make_env()
        await env.reset_async()
        adapter.fail_on_send = True
        obs = await env.step_async(HarnessAction(message="hi"))
        assert obs.done is True
        assert obs.metadata["error_type"] == "harness_crashed"
        assert adapter.calls[-1] == "stop"
        with pytest.raises(HarnessNotRunningError):
            await env.step_async(HarnessAction(message="again"))

    async def test_turn_timeout_returns_terminal_error_observation(self):
        env, adapter = make_env()
        adapter.config.session_timeout_s = 0.05
        adapter.send_delay_s = 5.0
        await env.reset_async()
        obs = await env.step_async(HarnessAction(message="hi"))
        assert obs.done is True
        assert obs.metadata["error_type"] == "turn_timeout"
        assert adapter.calls[-1] == "stop"

    async def test_unknown_action_type_raises(self):
        env, _ = make_env()
        await env.reset_async()
        with pytest.raises(TypeError):
            await env.step_async(Action())

    async def test_list_tools_routing_preserved(self):
        env, _ = make_env()
        obs = await env.step_async(ListToolsAction())
        assert sorted(t.name for t in obs.tools) == ["query_db", "read_file"]


class TestSyncFacadeAndClose:
    def test_sync_reset_and_step(self):
        env, adapter = make_env()
        adapter.scripted_turns = [turn_events("sync response", True)]
        obs = env.reset()
        assert obs.done is False
        obs = env.step(HarnessAction(message="go"))
        assert obs.metadata["response"] == "sync response"
        assert obs.done is True
        assert env.state.step_count == 1

    def test_close_is_idempotent_and_stops_adapter_and_bridge(self):
        env, adapter = make_env()
        env.reset()
        env.close()
        assert "stop" in adapter.calls
        assert FakeBridge.instances[0].stopped >= 1
        stop_count = adapter.calls.count("stop")
        env.close()
        assert adapter.calls.count("stop") == stop_count

    def test_close_retries_bridge_shutdown_after_failure(self):
        env, _ = make_env()
        env.reset()
        bridge = env._bridge
        bridge.fail_on_stop = True

        env.close()

        assert env._bridge is bridge
        stop_count = bridge.stopped
        bridge.fail_on_stop = False
        env.close()
        assert bridge.stopped == stop_count + 1
        assert env._bridge is None
        env.close()
        assert bridge.stopped == stop_count + 1

    def test_constructor_starts_nothing(self):
        env, adapter = make_env()
        assert adapter.calls == []

    def test_server_selects_async_paths(self):
        env, _ = make_env()
        assert overrides_method(env.reset_async, Environment.reset_async)
        assert overrides_method(env.step_async, Environment.step_async)
        assert env.SUPPORTS_CONCURRENT_SESSIONS is False


class TestTransforms:
    @pytest.mark.parametrize("sync", [False, True])
    async def test_reset_and_turn_apply_transform_once_after_rubric(self, sync):
        seen = []

        def transform(observation):
            seen.append(observation)
            return observation.model_copy(
                update={
                    "reward": observation.reward + 1.0,
                    "metadata": {**observation.metadata, "transformed": True},
                }
            )

        rubric = SpyRubric()
        env, adapter = make_env(rubric=rubric, transform=transform)
        adapter.scripted_turns = [turn_events("first"), turn_events("last", True)]
        try:
            reset_obs = env.reset() if sync else await env.reset_async()
            assert reset_obs.metadata["transformed"] is True
            assert reset_obs.reward == 1.0
            assert len(seen) == 1

            for message, done in [("first", False), ("last", True)]:
                action = HarnessAction(message=message)
                obs = env.step(action) if sync else await env.step_async(action)
                assert obs.metadata["transformed"] is True
                assert obs.metadata["response"] == message
                assert obs.done is done
                assert obs.reward == 1.75
                assert rubric.seen[-1][1] is seen[-1]
                assert "transformed" not in rubric.seen[-1][1].metadata
            assert len(seen) == 3
        finally:
            env.close()

    @pytest.mark.parametrize(
        "failure, value, error_type",
        [
            ("alive", False, "harness_crashed"),
            ("fail_on_send", True, "harness_crashed"),
            ("raise_turn_timeout", True, "turn_timeout"),
            ("send_delay_s", 5.0, "turn_timeout"),
        ],
    )
    async def test_terminal_errors_apply_transform(self, failure, value, error_type):
        seen = []

        def transform(observation):
            seen.append(observation)
            return observation.model_copy(
                update={"metadata": {**observation.metadata, "transformed": True}}
            )

        env, adapter = make_env(transform=transform)
        try:
            await env.reset_async()
            seen.clear()
            setattr(adapter, failure, value)
            obs = await env.step_async(HarnessAction(message="go"), timeout_s=0.05)
            assert obs.done is True
            assert obs.metadata["error_type"] == error_type
            assert obs.metadata["transformed"] is True
            assert len(seen) == 1
            assert seen[0].metadata["turn_events"] == obs.metadata["turn_events"]
        finally:
            env.close()


class TestCleanupAndErrorClassification:
    @pytest.mark.parametrize(
        "stage, failure",
        [
            ("is_alive", RuntimeError("adapter liveness check failed")),
            ("send_message", RuntimeError("unexpected adapter failure")),
            ("send_message", json.JSONDecodeError("invalid harness JSON", "{", 1)),
            ("rubric", RuntimeError("turn evaluation failed")),
        ],
    )
    @pytest.mark.parametrize("stop_fails", [False, True])
    async def test_unexpected_turn_failure_stops_resources_and_allows_retry(
        self, stage, failure, stop_fails, monkeypatch
    ):
        env, adapter = make_env(rubric=SpyRubric())
        await env.reset_async()
        target = env if stage == "rubric" else adapter
        method = "_apply_rubric_async" if stage == "rubric" else stage
        original = getattr(target, method)
        original_stop = adapter.stop

        async def fail(*args, **kwargs):
            raise failure

        async def failing_stop():
            await original_stop()
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(target, method, fail)
        if stop_fails:
            monkeypatch.setattr(adapter, "stop", failing_stop)
        try:
            with pytest.raises(type(failure)) as caught:
                await env.step_async(HarnessAction(message="go"))

            assert caught.value is failure
            assert adapter.alive is False
            assert adapter.calls[-1] == "stop"
            assert FakeBridge.instances[0].stopped >= 1
            with pytest.raises(HarnessNotRunningError, match="No active episode"):
                await env.step_async(HarnessAction(message="again"))

            monkeypatch.setattr(target, method, original)
            monkeypatch.setattr(adapter, "stop", original_stop)
            await env.reset_async()
            assert adapter.alive
            assert len(FakeBridge.instances) == 2
            obs = await env.step_async(HarnessAction(message="fresh turn"))
            assert obs.metadata["response"] == "default response"
        finally:
            env.close()

    @pytest.mark.parametrize("stage", ["is_alive", "send_message", "rubric"])
    @pytest.mark.parametrize("stop_fails", [False, True])
    async def test_cancelled_turn_cleans_up_resources(
        self, stage, stop_fails, monkeypatch
    ):
        env, adapter = make_env(rubric=SpyRubric())
        await env.reset_async()
        entered = asyncio.Event()
        target = env if stage == "rubric" else adapter
        method = "_apply_rubric_async" if stage == "rubric" else stage
        original = getattr(target, method)
        original_stop = adapter.stop

        async def stall(*args, **kwargs):
            await original(*args, **kwargs)
            entered.set()
            await asyncio.Event().wait()

        async def failing_stop():
            await original_stop()
            raise HarnessError("cleanup failed")

        monkeypatch.setattr(target, method, stall)
        if stop_fails:
            monkeypatch.setattr(adapter, "stop", failing_stop)
        turn_task = asyncio.create_task(env.step_async(HarnessAction(message="go")))
        try:
            await asyncio.wait_for(entered.wait(), timeout=10.0)
            turn_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await turn_task

            assert adapter.alive is False
            assert adapter.calls[-1] == "stop"
            assert FakeBridge.instances[0].stopped >= 1
            with pytest.raises(HarnessNotRunningError):
                await env.step_async(HarnessAction(message="again"))

            monkeypatch.setattr(target, method, original)
            monkeypatch.setattr(adapter, "stop", original_stop)
            await env.reset_async()
            obs = await env.step_async(HarnessAction(message="fresh turn"))
            assert obs.metadata["response"] == "default response"
        finally:
            if not turn_task.done():
                turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
            env.close()

    @pytest.mark.parametrize("stage", ["inject_tools", "start", "rubric"])
    async def test_cancelled_reset_cleans_up_resources(self, stage, monkeypatch):
        env, adapter = make_env(rubric=SpyRubric())
        entered = asyncio.Event()
        target = env if stage == "rubric" else adapter
        method = "_reset_rubric_async" if stage == "rubric" else stage
        original = getattr(target, method)

        async def stall(*args, **kwargs):
            await original(*args, **kwargs)
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(target, method, stall)
        reset_task = asyncio.create_task(env.reset_async())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10.0)
            reset_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reset_task

            assert adapter.alive is False
            assert adapter.calls[-1] == "stop"
            assert FakeBridge.instances[0].stopped >= 1
            with pytest.raises(HarnessNotRunningError):
                await env.step_async(HarnessAction(message="go"))
        finally:
            if not reset_task.done():
                reset_task.cancel()
            await asyncio.gather(reset_task, return_exceptions=True)
            env.close()

    async def test_reset_stops_adapter_even_when_not_alive(self):
        # A harness that died on its own reports is_alive() False while still
        # holding reapable resources; skipping stop() leaked them.
        env, adapter = make_env()
        await env.reset_async()
        adapter.alive = False
        adapter.calls.clear()

        await env.reset_async()

        assert adapter.calls[0] == "stop"

    async def test_start_failure_stops_the_adapter(self):
        env, adapter = make_env()
        adapter.fail_on_start = True
        with pytest.raises(HarnessError):
            await env.reset_async()
        # inject -> start (raises) -> stop
        assert adapter.calls[-1] == "stop"

    async def test_adapter_raised_timeout_is_not_reported_as_a_crash(self):
        env, adapter = make_env()
        await env.reset_async()
        adapter.raise_turn_timeout = True

        obs = await env.step_async(HarnessAction(message="hi"))

        assert obs.done is True
        assert obs.metadata["error_type"] == "turn_timeout"


class TestModeSpecificTools:
    async def test_mode_tools_are_not_injected(self, caplog):
        # Tools registered with tool(mode=...) are tracked by the environment
        # rather than the FastMCP server, so the bridge cannot serve them.
        # Advertising them would promise the harness a tool it cannot call.
        env, adapter = make_env()

        @env.tool(mode="production")
        def prod_only(x: int) -> int:
            """Production-only tool."""
            return x

        # A mode tool is only advertised when the env is in that mode; without
        # this the tool never reaches the injectable list and the test would
        # pass vacuously.
        env._mode = "production"
        listed = await env._async_handle_list_tools()
        assert "prod_only" in [t.name for t in listed.tools]

        with caplog.at_level("WARNING"):
            obs = await env.reset_async()

        assert "prod_only" not in obs.metadata["injected_tools"]
        assert all(t.name != "prod_only" for t in adapter.injected_tools)
        assert "prod_only" in caplog.text
