# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the harness/session runtime layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import Tool
from openenv.core.env_server.types import State
from openenv.core.harness import (
    build_harness_rollout_func,
    CLIHarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    RESERVED_TOOL_NAMES,
    RolloutEvent,
    SessionMCPBridge,
    StepEnvSessionAdapter,
    ToolResult,
    VerifyResult,
)
from openenv.core.llm_client import LLMResponse, ToolCall


class FakeSyncEnv:
    """Simple sync environment used to test session adapters."""

    def __init__(self):
        self.closed = False
        self.step_count = 0
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[str] = []

    def reset(self, **kwargs: Any) -> StepResult[dict[str, Any]]:
        self.reset_calls.append(kwargs)
        self.step_count = 0
        return StepResult(
            observation={"text": "Ready", "task": kwargs.get("task")},
            reward=0.0,
            done=False,
        )

    def step(self, action: str) -> StepResult[dict[str, Any]]:
        self.step_calls.append(action)
        self.step_count += 1
        done = action == "finish"
        reward = 1.0 if done else 0.25
        return StepResult(
            observation={
                "text": f"ran:{action}",
                "done": done,
                "reward": reward,
            },
            reward=reward,
            done=done,
        )

    def state(self) -> State:
        return State(episode_id="fake-episode", step_count=self.step_count)

    def close(self) -> None:
        self.closed = True


@dataclass
class RecordingSessionFactory:
    """Factory used to test rollout helper ordering."""

    session: Any
    created_tasks: list[Any] = field(default_factory=list)

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> Any:
        self.created_tasks.append((task, seed, episode_id))
        return self.session


class TestStepEnvSessionAdapter:
    """Tests for the generic step-based resource session adapter."""

    def test_session_exposes_tools_and_lifecycle(self):
        env = FakeSyncEnv()
        tools = [
            Tool(
                name="advance",
                description="Advance the fake environment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                    "required": ["value"],
                },
            ),
        ]

        session = StepEnvSessionAdapter(
            client=env,
            task="test-task",
            tool_specs=tools,
            action_builder=lambda name, arguments: arguments["value"],
            initial_messages_builder=lambda result, task: [
                {"role": "user", "content": f"{task}:{result.observation['text']}"}
            ],
        )

        assert session.initial_messages() == [
            {"role": "user", "content": "test-task:Ready"}
        ]
        assert session.list_tools() == tools

        tool_result = session.call_tool("advance", {"value": "next"})

        assert tool_result.data["observation"]["text"] == "ran:next"
        assert tool_result.done is False
        assert tool_result.metadata["reward"] == 0.25

        verify_result = session.verify(
            transcript=[{"role": "assistant", "content": ""}]
        )
        assert verify_result.env_reward == 0.25
        assert verify_result.done is False
        assert verify_result.metrics["step_count"] == 1

        session.close()
        assert env.closed is True

    def test_session_uses_custom_verify_builder(self):
        env = FakeSyncEnv()
        seen: dict[str, Any] = {}

        session = StepEnvSessionAdapter(
            client=env,
            task="verify-me",
            tool_specs=[],
            action_builder=lambda name, arguments: None,
            initial_messages_builder=lambda result, task: [],
            verify_builder=lambda transcript, final_state, last_result, state: (
                seen.update(
                    {
                        "transcript": transcript,
                        "final_state": final_state,
                        "last_result": last_result,
                        "state": state,
                    }
                )
                or VerifyResult(env_reward=9.0, done=True, metrics={"kind": "custom"})
            ),
        )

        result = session.verify(
            transcript=[{"role": "assistant", "content": "done"}],
            final_state={"terminal": True},
        )

        assert result.env_reward == 9.0
        assert result.done is True
        assert result.metrics == {"kind": "custom"}
        assert seen["final_state"] == {"terminal": True}

    def test_session_rejects_reserved_tool_names(self):
        env = FakeSyncEnv()

        with pytest.raises(ValueError, match="reserved"):
            StepEnvSessionAdapter(
                client=env,
                task="reserved-tool",
                tool_specs=[
                    Tool(
                        name="reset",
                        description="Must remain an orchestration control",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                action_builder=lambda name, arguments: name,
                initial_messages_builder=lambda result, task: [],
            )

    def test_session_closes_client_when_reset_fails(self):
        class FailingResetEnv(FakeSyncEnv):
            def reset(self, **kwargs: Any) -> StepResult[dict[str, Any]]:
                raise RuntimeError("reset failed")

        env = FailingResetEnv()

        with pytest.raises(RuntimeError, match="reset failed"):
            StepEnvSessionAdapter(
                client=env,
                task="bad-reset",
                tool_specs=[],
                action_builder=lambda name, arguments: None,
                initial_messages_builder=lambda result, task: [],
            )

        assert env.closed is True


class TestSessionMCPBridge:
    """Tests for exposing sessions through an MCP-style JSON-RPC bridge."""

    def test_tools_list_and_tools_call(self):
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="bridge-task",
            tool_specs=[
                Tool(
                    name="finish",
                    description="Finish the fake task",
                    input_schema={
                        "type": "object",
                        "properties": {},
                    },
                )
            ],
            action_builder=lambda name, arguments: name,
            initial_messages_builder=lambda result, task: [],
        )
        bridge = SessionMCPBridge(session)

        list_response = bridge.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        assert list_response["result"]["tools"][0]["name"] == "finish"

        call_response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "finish", "arguments": {}},
            }
        )
        assert call_response["result"]["data"]["reward"] == 1.0
        assert call_response["result"]["done"] is True

    def test_unknown_method_returns_jsonrpc_error(self):
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="bridge-task",
            tool_specs=[],
            action_builder=lambda name, arguments: None,
            initial_messages_builder=lambda result, task: [],
        )
        bridge = SessionMCPBridge(session)

        response = bridge.handle_request(
            {"jsonrpc": "2.0", "id": 7, "method": "resources/list", "params": {}}
        )

        assert response["error"]["message"] == "Method not found"
        assert response["id"] == 7

    def test_unknown_tool_returns_jsonrpc_error(self):
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="bridge-task",
            tool_specs=[],
            action_builder=lambda name, arguments: None,
            initial_messages_builder=lambda result, task: [],
        )
        bridge = SessionMCPBridge(session)

        response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "missing", "arguments": {}},
            }
        )

        assert response["error"]["message"] == "Unknown tool: missing"
        assert response["id"] == 9

    def test_reserved_tool_returns_jsonrpc_error(self):
        class ReservedSession:
            def list_tools(self):
                return []

            def call_tool(self, name, arguments):
                raise AssertionError("reserved names must be blocked by the bridge")

        bridge = SessionMCPBridge(ReservedSession())

        for name in RESERVED_TOOL_NAMES:
            response = bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                }
            )

            assert (
                response["error"]["message"] == f"Reserved orchestration tool: {name}"
            )
            assert response["id"] == 12

    def test_value_error_returns_invalid_params_error(self):
        class ValueErrorSession:
            def list_tools(self):
                return []

            def call_tool(self, name, arguments):
                raise ValueError("bad arguments")

        bridge = SessionMCPBridge(ValueErrorSession())

        response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "finish", "arguments": {}},
            }
        )

        assert response["error"]["message"] == "bad arguments"
        assert response["id"] == 10

    def test_unexpected_tool_error_returns_internal_error(self):
        class FailingSession:
            def list_tools(self):
                return []

            def call_tool(self, name, arguments):
                raise RuntimeError("bridge exploded")

        bridge = SessionMCPBridge(FailingSession())

        response = bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"name": "finish", "arguments": {}},
            }
        )

        assert response["error"]["message"] == "bridge exploded"
        assert response["id"] == 11


class TestMCPHarnessAdapter:
    """Tests for the white-box MCP-first harness adapter."""

    def test_run_white_box_accumulates_model_tokens_and_tool_trace(self):
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="white-box-task",
            tool_specs=[
                Tool(
                    name="finish",
                    description="Finish the task",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            action_builder=lambda name, arguments: name,
            initial_messages_builder=lambda result, task: [
                {"role": "user", "content": "Solve the task"}
            ],
        )

        adapter = MCPHarnessAdapter()

        model_calls: list[tuple[list[dict[str, Any]], list[Tool], dict[str, Any]]] = []

        def model_step(messages, tools, sampling):
            model_calls.append((messages, tools, sampling))
            return ModelStepResult(
                response=LLMResponse(
                    content="Calling finish",
                    tool_calls=[ToolCall(id="tool-1", name="finish", args={})],
                ),
                prompt_ids=[11, 12],
                completion_ids=[21, 22],
                logprobs=[-0.1, -0.2],
            )

        result = adapter.run_white_box(
            model_step=model_step,
            session=session,
            limits=HarnessRunLimits(max_turns=2),
        )

        assert result.done is True
        assert result.prompt_ids == [11, 12]
        assert result.completion_ids == [21, 22]
        assert result.logprobs == [-0.1, -0.2]
        assert len(result.tool_trace) == 1
        assert result.tool_trace[0].tool_name == "finish"
        assert result.tool_trace[0].result.done is True
        assert model_calls[0][2] == {}

    def test_run_white_box_does_not_verify_session(self):
        class VerifyingSession:
            def __init__(self):
                self.verified = False

            def initial_messages(self):
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                return []

            def call_tool(self, name, arguments):
                raise AssertionError("No tools expected")

            def verify(self, transcript, final_state=None):
                self.verified = True
                return VerifyResult(env_reward=1.0)

            def close(self):
                pass

        session = VerifyingSession()
        adapter = MCPHarnessAdapter()

        adapter.run_white_box(
            model_step=lambda messages, tools, sampling: ModelStepResult(
                response=LLMResponse(content="Final answer", tool_calls=[]),
                prompt_ids=[1],
                completion_ids=[2],
                logprobs=[-0.5],
            ),
            session=session,
            limits=HarnessRunLimits(max_turns=1),
        )

        assert session.verified is False


class TestBuildHarnessRolloutFunc:
    """Tests for the TRL rollout helper built on sessions/harnesses."""

    def test_rollout_func_verifies_after_harness_completes(self):
        order: list[str] = []

        class RecordingSession:
            def __init__(self):
                self.closed = False

            def initial_messages(self):
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                return [
                    Tool(
                        name="finish",
                        description="Finish the task",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, name, arguments):
                order.append("tool")
                return ToolResult(data={"reward": 1.0}, done=True)

            def verify(self, transcript, final_state=None):
                order.append("verify")
                return VerifyResult(
                    env_reward=1.0,
                    done=True,
                    metrics={"verified": True},
                )

            def close(self):
                order.append("close")
                self.closed = True

        session = RecordingSession()
        factory = RecordingSessionFactory(session=session)
        adapter = MCPHarnessAdapter()

        def model_step_builder(trainer, session):
            return lambda messages, tools, sampling: (
                order.append("model")
                or ModelStepResult(
                    response=LLMResponse(
                        content="calling tool",
                        tool_calls=[ToolCall(id="call-1", name="finish", args={})],
                    ),
                    prompt_ids=[101],
                    completion_ids=[202],
                    logprobs=[-0.25],
                )
            )

        rollout_func = build_harness_rollout_func(
            session_factory=factory,
            harness_adapter=adapter,
            model_step_builder=model_step_builder,
            limits=HarnessRunLimits(max_turns=2),
        )

        result = rollout_func(["task-a"], trainer=object())

        assert result["prompt_ids"] == [[101]]
        assert result["completion_ids"] == [[202]]
        assert result["logprobs"] == [[-0.25]]
        assert result["env_reward"] == [1.0]
        assert result["verify_metrics"] == [{"verified": True}]
        assert factory.created_tasks == [("task-a", None, None)]
        assert order == ["model", "tool", "verify", "close"]

    def test_rollout_func_rejects_verify_reward_mismatch(self):
        class RecordingSession:
            def initial_messages(self):
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                return [
                    Tool(
                        name="finish",
                        description="Finish the task",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, name, arguments):
                return ToolResult(
                    data={"reward": 1.0},
                    done=True,
                    metadata={"reward": 1.0},
                )

            def verify(self, transcript, final_state=None):
                return VerifyResult(env_reward=0.5, done=True)

            def close(self):
                pass

        factory = RecordingSessionFactory(session=RecordingSession())
        rollout_func = build_harness_rollout_func(
            session_factory=factory,
            harness_adapter=MCPHarnessAdapter(),
            model_step_builder=lambda trainer, session: (
                lambda messages, tools, sampling: ModelStepResult(
                    response=LLMResponse(
                        content="calling tool",
                        tool_calls=[ToolCall(id="call-1", name="finish", args={})],
                    ),
                )
            ),
            limits=HarnessRunLimits(max_turns=1),
        )

        with pytest.raises(
            ValueError,
            match="verify.env_reward must forward the environment reward",
        ):
            rollout_func(["task-a"], trainer=object())

    def test_rollout_func_accepts_float_rounding_in_reward_check(self):
        class RecordingSession:
            def initial_messages(self):
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                return [
                    Tool(
                        name="finish",
                        description="Finish the task",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]

            def call_tool(self, name, arguments):
                return ToolResult(
                    data={"reward": 0.3},
                    done=True,
                    metadata={"reward": 0.3},
                )

            def verify(self, transcript, final_state=None):
                return VerifyResult(env_reward=0.1 + 0.2, done=True)

            def close(self):
                pass

        rollout_func = build_harness_rollout_func(
            session_factory=RecordingSessionFactory(session=RecordingSession()),
            harness_adapter=MCPHarnessAdapter(),
            model_step_builder=lambda trainer, session: (
                lambda messages, tools, sampling: ModelStepResult(
                    response=LLMResponse(
                        content="calling tool",
                        tool_calls=[ToolCall(id="call-1", name="finish", args={})],
                    ),
                )
            ),
            limits=HarnessRunLimits(max_turns=1),
        )

        result = rollout_func(["task-a"], trainer=object())

        assert result["env_reward"] == [0.3]

    def test_rollout_func_rejects_missing_reward(self):
        class RecordingSession:
            def initial_messages(self):
                return [{"role": "user", "content": "task"}]

            def list_tools(self):
                return []

            def call_tool(self, name, arguments):
                raise AssertionError("No tools expected")

            def verify(self, transcript, final_state=None):
                return VerifyResult(env_reward=None, done=True)

            def close(self):
                pass

        rollout_func = build_harness_rollout_func(
            session_factory=RecordingSessionFactory(session=RecordingSession()),
            harness_adapter=MCPHarnessAdapter(),
            model_step_builder=lambda trainer, session: (
                lambda messages, tools, sampling: ModelStepResult(
                    response=LLMResponse(content="done", tool_calls=[]),
                )
            ),
            limits=HarnessRunLimits(max_turns=1),
        )

        with pytest.raises(ValueError, match="did not produce an environment reward"):
            rollout_func(["task-a"], trainer=object())


class TestCLIHarnessAdapter:
    """Tests for black-box CLI-style evaluation harnesses."""

    def test_black_box_runner_can_use_session_mcp_bridge(self):
        env = FakeSyncEnv()
        session = StepEnvSessionAdapter(
            client=env,
            task="cli-task",
            tool_specs=[
                Tool(
                    name="finish",
                    description="Finish the task",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            action_builder=lambda name, arguments: name,
            initial_messages_builder=lambda result, task: [],
        )

        def runner(bridge, current_session, limits):
            list_response = bridge.handle_request(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            )
            call_response = bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "finish", "arguments": {}},
                }
            )
            return HarnessRolloutResult(
                messages=[{"role": "assistant", "content": "done"}],
                tool_trace=[],
                events=[
                    RolloutEvent(
                        type="cli_runner",
                        payload={
                            "tools": list_response["result"]["tools"],
                            "result": call_response["result"],
                        },
                    )
                ],
                done=True,
                metrics={"mode": "black_box"},
            )

        adapter = CLIHarnessAdapter(runner=runner)
        result = adapter.run_black_box(session=session, limits=HarnessRunLimits())

        assert result.done is True
        assert result.metrics["mode"] == "black_box"
        assert result.events[0].payload["result"]["done"] is True
