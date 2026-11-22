# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle coverage for MCP tool calls exposed by the environment server."""

from typing import Any, Callable

from fastapi.testclient import TestClient
from fastmcp import FastMCP
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class LifecycleMCPEnvironment(MCPEnvironment):
    """Small MCP environment with injectable episode limits."""

    def __init__(
        self,
        clock: Callable[[], float],
        max_steps: int = 2,
        time_limit_s: float = 10.0,
    ) -> None:
        self.clock = clock
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self.started_at = clock()
        self.done = False
        self.tool_calls = 0
        self._state = State(episode_id="lifecycle", step_count=0)

        server = FastMCP("lifecycle-test")

        @server.tool
        def record(value: int = 1) -> int:
            self.tool_calls += 1
            return value

        @server.tool
        def fail() -> None:
            raise RuntimeError("tool failed")

        super().__init__(server)

    def reset(self, **_: Any) -> Observation:
        self.started_at = self.clock()
        self.done = False
        self.tool_calls = 0
        self._state = State(episode_id="lifecycle", step_count=0)
        return Observation(done=False, reward=0.0)

    def _step_impl(self, action: Action, **_: Any) -> Observation:
        return Observation(metadata={"action": str(action)})

    def _finish(self, observation: Observation) -> Observation:
        if (
            self._state.step_count >= self.max_steps
            or self.clock() - self.started_at >= self.time_limit_s
        ):
            self.done = True
            observation.done = True
        return observation

    def step(self, action: Action, **kwargs: Any) -> Observation:
        if self.done:
            return Observation(done=True)
        self._state.step_count += 1
        return self._finish(super().step(action, **kwargs))

    async def step_async(self, action: Action, **kwargs: Any) -> Observation:
        if self.done:
            return Observation(done=True)
        self._state.step_count += 1
        return self._finish(await super().step_async(action, **kwargs))

    @property
    def state(self) -> State:
        return self._state


class DirectMCPEnvironment(MCPEnvironment):
    """MCP environment that retains the base class direct dispatch."""

    def __init__(self) -> None:
        self.tool_calls = 0
        self._state = State(episode_id="direct", step_count=0)
        server = FastMCP("direct-test")

        @server.tool
        def record() -> str:
            self.tool_calls += 1
            return "direct"

        super().__init__(server)

    def reset(self, **_: Any) -> Observation:
        return Observation()

    def _step_impl(self, action: Action, **_: Any) -> Observation:
        return Observation(metadata={"action": str(action)})

    @property
    def state(self) -> State:
        return self._state


class SyncOnlyMCPEnvironment(DirectMCPEnvironment):
    """MCP environment that overrides only the synchronous step method."""

    def __init__(self) -> None:
        self.step_calls = 0
        super().__init__()

    def step(self, action: Action, **kwargs: Any) -> Observation:
        self.step_calls += 1
        return super().step(action, **kwargs)


def make_app(factory: Callable[[], MCPEnvironment]):
    return create_app(
        env=factory,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
        mode="production",
    )


def create_session(client: TestClient) -> str:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "openenv/session/create",
            "params": {},
            "id": 1,
        },
    )
    return response.json()["result"]["session_id"]


def call_tool(client: TestClient, session_id: str, request_id: int = 2) -> dict:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "record",
                "arguments": {"value": request_id},
                "session_id": session_id,
            },
            "id": request_id,
        },
    )
    return response.json()


def test_http_mcp_calls_use_step_and_stop_at_limit() -> None:
    clock = FakeClock()
    instances: list[LifecycleMCPEnvironment] = []

    def factory() -> LifecycleMCPEnvironment:
        env = LifecycleMCPEnvironment(clock)
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        session_id = create_session(client)
        env = instances[-1]

        assert "result" in call_tool(client, session_id, 2)
        second = call_tool(client, session_id, 3)
        after_end = call_tool(client, session_id, 4)

        assert "result" in second
        assert after_end["result"] is None
        assert env.state.step_count == 2
        assert env.tool_calls == 2
        assert env.done is True


def test_mcp_websocket_calls_use_step_once() -> None:
    clock = FakeClock()
    instances: list[LifecycleMCPEnvironment] = []

    def factory() -> LifecycleMCPEnvironment:
        env = LifecycleMCPEnvironment(clock)
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        with client.websocket_connect("/mcp") as websocket:
            for request_id in (1, 2, 3):
                websocket.send_json(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "record",
                            "arguments": {"value": request_id},
                        },
                        "id": request_id,
                    }
                )
                response = websocket.receive_json()

        env = instances[-1]
        assert response["result"] is None
        assert env.state.step_count == 2
        assert env.tool_calls == 2


def test_http_mcp_and_websocket_step_share_one_episode_without_double_counting() -> (
    None
):
    clock = FakeClock()
    instances: list[LifecycleMCPEnvironment] = []

    def factory() -> LifecycleMCPEnvironment:
        env = LifecycleMCPEnvironment(clock)
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        session_id = create_session(client)
        env = instances[-1]
        call_tool(client, session_id)

        with client.websocket_connect(f"/ws?session_id={session_id}") as websocket:
            websocket.send_json(
                {
                    "type": "step",
                    "data": {
                        "type": "call_tool",
                        "tool_name": "record",
                        "arguments": {"value": 3},
                    },
                }
            )
            response = websocket.receive_json()

        assert response["type"] == "observation"
        assert response["data"]["done"] is True
        assert env.state.step_count == 2
        assert env.tool_calls == 2


def test_time_limit_and_calls_after_episode_end() -> None:
    clock = FakeClock()
    instances: list[LifecycleMCPEnvironment] = []

    def factory() -> LifecycleMCPEnvironment:
        env = LifecycleMCPEnvironment(clock)
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        session_id = create_session(client)
        env = instances[-1]
        clock.advance(11)

        first = call_tool(client, session_id)
        after_end = call_tool(client, session_id, 3)

        assert "result" in first
        assert after_end["result"] is None
        assert env.state.step_count == 1
        assert env.tool_calls == 1
        assert env.done is True


def test_tools_list_does_not_count_as_a_step() -> None:
    clock = FakeClock()
    instances: list[LifecycleMCPEnvironment] = []

    def factory() -> LifecycleMCPEnvironment:
        env = LifecycleMCPEnvironment(clock)
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        session_id = create_session(client)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )

        assert "tools" in response.json()["result"]
        assert instances[-1].state.step_count == 0


def test_tool_errors_keep_existing_json_rpc_mapping() -> None:
    clock = FakeClock()

    with TestClient(make_app(lambda: LifecycleMCPEnvironment(clock))) as client:
        session_id = create_session(client)
        missing = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "missing",
                    "session_id": session_id,
                },
                "id": 2,
            },
        ).json()
        failed = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "fail", "session_id": session_id},
                "id": 3,
            },
        ).json()

        assert missing["error"]["code"] == -32602
        assert failed["error"]["code"] == -32603


def test_environment_without_step_override_keeps_direct_dispatch() -> None:
    instances: list[DirectMCPEnvironment] = []

    def factory() -> DirectMCPEnvironment:
        env = DirectMCPEnvironment()
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        session_id = create_session(client)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "record", "session_id": session_id},
                "id": 2,
            },
        ).json()

        assert "result" in response
        assert instances[-1].tool_calls == 1
        assert instances[-1].state.step_count == 0


def test_environment_with_only_sync_step_override_keeps_direct_dispatch() -> None:
    instances: list[SyncOnlyMCPEnvironment] = []

    def factory() -> SyncOnlyMCPEnvironment:
        env = SyncOnlyMCPEnvironment()
        instances.append(env)
        return env

    with TestClient(make_app(factory)) as client:
        session_id = create_session(client)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "record", "session_id": session_id},
                "id": 2,
            },
        ).json()

        # This matches the current WebSocket step dispatch for a sync-only override.
        assert "result" in response
        assert instances[-1].tool_calls == 1
        assert instances[-1].step_calls == 0
