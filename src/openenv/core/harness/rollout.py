# SPDX-License-Identifier: BSD-3-Clause

"""Trainer-side rollout helpers for driving environments with a harness loop.

This module hosts the whole-rollout API: a harness drives an entire episode in
one ``run_white_box``/``run_black_box`` call against a [`~openenv.core.harness.rollout.ResourceSession`].
It complements the turn-based agentic harness layer (RFC 005) defined in the
sibling modules of ``openenv.core.harness``. Import these names from
``openenv.core.harness``.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    runtime_checkable,
    TypedDict,
    TypeVar,
)

from ..client_types import StepResult
from ..env_server.mcp_types import JsonRpcErrorCode, JsonRpcResponse, Tool
from ..env_server.types import State
from ..llm_client import LLMResponse
from .training import TrainingTrace

Message = dict[str, Any]
RESERVED_TOOL_NAMES = frozenset({"reset", "step", "state", "close"})


@dataclass
class ToolResult:
    """Normalized result from a resource session tool invocation."""

    data: Any = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class VerifyResult:
    """Final rollout data produced after a rollout completes.

    ``env_reward`` must forward reward already produced inside the environment.
    It must not synthesize a new reward in the orchestration layer.
    """

    env_reward: float | None = None
    done: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutEvent:
    """Event emitted while a harness drives a rollout."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolTraceEntry:
    """Record of a tool call issued by the harness."""

    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult


@dataclass
class ModelStepResult:
    """Structured output from one white-box model sampling step."""

    response: LLMResponse
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class HarnessRolloutResult:
    """Complete result of a harness-driven rollout."""

    messages: list[Message] = field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)
    events: list[RolloutEvent] = field(default_factory=list)
    done: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class HarnessRunLimits:
    """Execution limits for harness-driven rollouts."""

    max_turns: int = 10
    max_tool_calls_per_turn: int | None = None
    max_total_tool_calls: int | None = None
    sampling: dict[str, Any] = field(default_factory=dict)


class ModelStep(Protocol):
    """Callable used by white-box harnesses to sample the next model turn."""

    def __call__(
        self,
        messages: list[Message],
        tools: list[Tool],
        sampling: dict[str, Any],
    ) -> ModelStepResult: ...


class TraceEntry(TypedDict, total=False):
    """One captured model call from a loop-owning rollout: the request, the reply, and the tokens.

    Defined HERE, not in the trainer. A loop-owning harness (the agent runs its own tool loop and
    we read back what it did) is an OpenEnv concern, and OpenEnv is what produces this record --
    `core.harness.capture.contract.to_trace_entries` emits exactly these keys. TRL carried a local
    copy with a `TODO(@openenv)` asking for this, which left the trainer owning the schema for a
    shape it neither produces nor can validate; the two could drift and nothing would notice until
    the token fields came back empty.

    `completion_token_ids` and `per_token_logps` must be equal length when both are present: they
    are the sampled ids and their generator logprobs, in order. `completion_tokens` is a fallback
    for engines that return token STRINGS (`"token_id:{id}"`) rather than ids.

    `prompt_token_ids` IS THE POINT OF THIS RECORD. It is the engine's own tokenization of
    everything the model saw before it generated, and a consumer that has it MUST NOT re-derive the
    prompt. This field did not exist until 2026-09; the docstring here used to say there was
    "deliberately no prompt field" and that consumers should re-derive from `request`. That
    instruction was wrong, and it was expensive:

      * a local re-render with `apply_chat_template` matched the engine on 0 of 28 measured turns on
        Qwen3.5-4B -- off by two tokens at the generation boundary, every turn;
      * Qwen3.5-4B and -2B ship INVERTED `enable_thinking` defaults, so the same re-render matched
        the engine for one and diverged for the other: 100% of turn transitions forked,
        drift_tokens_mean 189 against 2.2, KL 0.067 against 0.016;
      * training on those misaligned positions collapsed a run permanently at its FIRST weight
        update, the model emitting `<|im_start|>bash` where `<function=bash>` belongs.

    Both producers already held these ids and dropped them here: capture has `node.prompt_ids`,
    harbor has `HarborTurn.prompt_token_ids`. `to_turn_records` kept them but had no HTTP endpoint,
    so no remote consumer could reach it.

    `loss_mask` marks which positions are trainable (1) versus context (0), covering the whole
    sequence `prompt_token_ids + completion_token_ids`. Without it a consumer has to guess which
    calls were real agent turns and which were framework bookkeeping -- a heuristic where the
    producer has structural knowledge. It also carries the case where a turn's logprobs were
    rejected on ingest: the tokens stay as context and the mask goes to 0, which a consumer cannot
    infer and would otherwise train against a logprob of 0.0, i.e. p=1.0.

    `reward` is a MIRROR of what `verify()` already returned, present so a persisted trace is
    self-describing. `verify()` remains the source of truth. `None` means UNSCORED and must never be
    coerced to 0.0: a rollout that failed to grade is excluded from the group baseline, not punished.

    Only an ENVIRONMENT can set `reward`. A capture proxy sees model calls, not task outcomes, so
    `openenv.core.harness.capture.to_trace_entries` omits the key entirely rather than guessing. This
    is a `total=False` TypedDict and every key is optional for exactly that reason: read `reward` with
    `.get()`, because indexing it raises on any entry a proxy produced.
    """

    request: dict[
        str, Any
    ]  # forwarded chat body: {"messages": [...], "tools": [...] | None}
    response: dict[str, Any]  # upstream reply: {"choices": [{"message": {...}, ...}]}
    prompt_token_ids: list[
        int
    ]  # the ENGINE's tokenization of everything before this turn
    completion_token_ids: list[int]  # generated token ids for this turn
    completion_tokens: list[str]  # fallback token strings when ids are absent
    per_token_logps: list[float]  # generator logprobs, aligned with the ids above
    loss_mask: list[int]  # 1 = train this position, over prompt + completion
    reward: float | None  # mirror of verify(); None is UNSCORED, never 0.0
    metadata: dict[str, Any]  # task id, difficulty tier, harness, sandbox, session id


@runtime_checkable
class LoopOwningSession(Protocol):
    """What a session must offer BEYOND `ResourceSession` when the agent owns its own loop.

    In the loop-owning path nothing calls `step()` per turn -- an external agent (opencode, codex,
    claude-code, ...) drives itself to completion and the captured trace is read back afterwards.
    So a factory used in that mode must return sessions that can be waited on and asked for their
    trace. Neither method belongs on the base `ResourceSession`, which models the step-per-turn
    contract.

    `wait_for_completion` returns the agent's exit code. `fetch_proxy_trace` returns the captured
    turns; where they come from is the session's business -- a file inside the sandbox, or an HTTP
    call to a capture server that multiplexes many rollouts at once. That freedom is the point:
    the consumer asks for `TraceEntry`s and does not learn how they were obtained.

    `@runtime_checkable` so a factory can assert what it is about to return actually satisfies this
    -- the alternative is discovering a missing `fetch_proxy_trace` several minutes into a paid
    rollout. Note that only method PRESENCE is checked, never signatures, which is the right
    strictness here: the two implementations legitimately differ in how they wait.
    """

    def wait_for_completion(self, timeout_s: float | None = ...) -> int: ...

    def fetch_proxy_trace(self) -> list[TraceEntry]: ...


@runtime_checkable
class TrainableSession(Protocol):
    """A loop-owning session that supplies validated, masked engine captures."""

    def wait_for_completion(self, timeout_s: float | None = ...) -> int: ...

    def fetch_training_trace(self) -> TrainingTrace: ...


class ResourceSession(ABC):
    """Per-rollout environment/resource session exposed to harnesses."""

    @abstractmethod
    def initial_messages(self) -> list[Message]:
        """Return initial prompt messages for the harness."""

    @abstractmethod
    def list_tools(self) -> list[Tool]:
        """Return the MCP-style tool manifest exposed by the session."""

    @abstractmethod
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a session tool."""

    @abstractmethod
    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        """Finalize a rollout after the harness has stopped.

        This hook may add metrics or artifacts and may forward the final reward
        already produced by the environment.
        """

    @abstractmethod
    def close(self) -> None:
        """Release session resources."""


SessionT = TypeVar("SessionT", bound=ResourceSession)


class ResourceSessionFactory(ABC, Generic[SessionT]):
    """Factory for producing isolated per-rollout sessions.

    Generic over the concrete session type it creates, so a subclass such as
    ``OpenCodeSessionFactory(ResourceSessionFactory[OpenCodeSession])`` types
    ``create`` as returning ``OpenCodeSession``. Subclassing without a type
    argument stays valid and behaves as before.
    """

    @abstractmethod
    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> SessionT:
        """Create one isolated resource session for a rollout."""


class HarnessAdapter(ABC):
    """Interface implemented by harness drivers."""

    @abstractmethod
    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        """Run a rollout while the trainer owns model sampling."""

    @abstractmethod
    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        """Run a rollout with an opaque harness (evaluation-only path)."""


def _serialize_for_message(value: Any) -> str:
    """Convert structured tool data to a stable text payload."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _state_to_data(state: Any) -> Any:
    """Convert state objects to plain data for metrics and artifacts."""

    if state is None:
        return None
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state


def _tool_result_reward(tool_result: ToolResult) -> float | None:
    """Extract the environment reward already emitted by a tool result."""

    reward = tool_result.metadata.get("reward")
    if reward is None and isinstance(tool_result.data, dict):
        reward = tool_result.data.get("reward")
    if reward is None:
        return None
    return float(reward)


def _resolve_env_reward(
    rollout: HarnessRolloutResult,
    verify: VerifyResult,
) -> float:
    """Resolve the final environment reward without allowing external synthesis."""

    trace_reward: float | None = None
    for entry in reversed(rollout.tool_trace):
        trace_reward = _tool_result_reward(entry.result)
        if trace_reward is not None:
            break

    verify_reward = None if verify.env_reward is None else float(verify.env_reward)
    if (
        trace_reward is not None
        and verify_reward is not None
        and not math.isclose(
            verify_reward,
            trace_reward,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(
            "verify.env_reward must forward the environment reward from the rollout"
        )

    if trace_reward is not None:
        return trace_reward
    if verify_reward is not None:
        return verify_reward
    raise ValueError("rollout did not produce an environment reward")


class StepEnvSessionAdapter(ResourceSession):
    """Expose an existing step/reset/state client as a resource session."""

    def __init__(
        self,
        client: Any,
        *,
        task: Any = None,
        seed: int | None = None,
        episode_id: str | None = None,
        tool_specs: list[Tool],
        action_builder: Callable[[str, dict[str, Any]], Any],
        initial_messages_builder: Callable[[StepResult[Any], Any], list[Message]],
        tool_result_builder: Callable[
            [str, dict[str, Any], StepResult[Any], Any],
            ToolResult,
        ]
        | None = None,
        verify_builder: Callable[
            [list[Message], Any | None, StepResult[Any] | None, Any],
            VerifyResult,
        ]
        | None = None,
        reset_kwargs: dict[str, Any] | None = None,
    ):
        if hasattr(client, "sync") and callable(client.sync):
            self._client = client.sync()
        else:
            self._client = client

        self._task = task
        self._tool_specs = list(tool_specs)
        self._action_builder = action_builder
        self._initial_messages_builder = initial_messages_builder
        self._tool_result_builder = tool_result_builder or self._default_tool_result
        self._verify_builder = verify_builder or self._default_verify
        self._closed = False
        self._tools_by_name = {tool.name: tool for tool in self._tool_specs}
        reserved = sorted(set(self._tools_by_name) & RESERVED_TOOL_NAMES)
        if reserved:
            raise ValueError(
                "Tool names are reserved for orchestration controls: "
                + ", ".join(reserved)
            )

        reset_payload = dict(reset_kwargs or {})
        if seed is not None:
            reset_payload.setdefault("seed", seed)
        if episode_id is not None:
            reset_payload.setdefault("episode_id", episode_id)

        self._last_result: StepResult[Any] | None = None
        self._last_state = None
        try:
            self._initial_result: StepResult[Any] = self._client.reset(**reset_payload)
            self._last_state = self._read_state()
        except Exception:
            self.close()
            raise

    def _read_state(self) -> Any:
        if hasattr(self._client, "state") and callable(self._client.state):
            return self._client.state()
        return None

    def _default_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: StepResult[Any],
        state: Any,
    ) -> ToolResult:
        return ToolResult(
            data={
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "observation": result.observation,
                "reward": result.reward,
                "done": result.done,
            },
            done=bool(result.done),
            metadata={
                "reward": result.reward,
                "state": _state_to_data(state),
            },
        )

    def _default_verify(
        self,
        transcript: list[Message],
        final_state: Any | None,
        last_result: StepResult[Any] | None,
        state: Any,
    ) -> VerifyResult:
        reward = None if last_result is None else last_result.reward
        done = False if last_result is None else bool(last_result.done)
        metrics = {}
        if isinstance(state, State):
            metrics["step_count"] = state.step_count
        elif isinstance(state, dict) and "step_count" in state:
            metrics["step_count"] = state["step_count"]

        return VerifyResult(
            env_reward=reward,
            done=done,
            metrics=metrics,
            artifacts={
                "final_state": _state_to_data(state),
                "transcript_length": len(transcript),
            },
        )

    def initial_messages(self) -> list[Message]:
        return list(self._initial_messages_builder(self._initial_result, self._task))

    def list_tools(self) -> list[Tool]:
        return list(self._tool_specs)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in RESERVED_TOOL_NAMES:
            raise KeyError(f"Reserved orchestration tool: {name}")
        if name not in self._tools_by_name:
            raise KeyError(f"Unknown tool: {name}")

        action = self._action_builder(name, dict(arguments))
        result = self._client.step(action)
        state = self._read_state()
        self._last_result = result
        self._last_state = state
        return self._tool_result_builder(name, dict(arguments), result, state)

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        return self._verify_builder(
            list(transcript),
            final_state,
            self._last_result,
            self._last_state,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self._client, "close") and callable(self._client.close):
            self._client.close()


class SessionMCPBridge:
    """Expose a resource session through an in-process MCP JSON-RPC bridge."""

    def __init__(self, session: ResourceSession):
        self.session = session

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        if request.get("jsonrpc") != "2.0":
            return JsonRpcResponse.error_response(
                JsonRpcErrorCode.INVALID_REQUEST,
                message="Invalid Request",
                request_id=request_id,
            ).model_dump()

        if method == "tools/list":
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in self.session.list_tools()
            ]
            return JsonRpcResponse.success(
                {"tools": tools},
                request_id=request_id,
            ).model_dump()

        if method == "tools/call":
            if "name" not in params:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    message="Missing tool name",
                    request_id=request_id,
                ).model_dump()

            tool_name = params["name"]
            if tool_name in RESERVED_TOOL_NAMES:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    message=f"Reserved orchestration tool: {tool_name}",
                    request_id=request_id,
                ).model_dump()

            try:
                result = self.session.call_tool(
                    tool_name,
                    dict(params.get("arguments", {})),
                )
            except KeyError as exc:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.METHOD_NOT_FOUND,
                    message=str(exc.args[0]) if exc.args else "Method not found",
                    request_id=request_id,
                ).model_dump()
            except ValueError as exc:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INVALID_PARAMS,
                    message=str(exc),
                    request_id=request_id,
                ).model_dump()
            except Exception as exc:
                return JsonRpcResponse.error_response(
                    JsonRpcErrorCode.INTERNAL_ERROR,
                    message=str(exc),
                    request_id=request_id,
                ).model_dump()
            return JsonRpcResponse.success(
                {
                    "data": result.data,
                    "done": result.done,
                    "metadata": dict(result.metadata),
                    "error": result.error,
                },
                request_id=request_id,
            ).model_dump()

        return JsonRpcResponse.error_response(
            JsonRpcErrorCode.METHOD_NOT_FOUND,
            message="Method not found",
            request_id=request_id,
        ).model_dump()


class MCPHarnessAdapter(HarnessAdapter):
    """White-box harness that follows an MCP tool-calling loop."""

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        run_limits = limits or HarnessRunLimits()
        messages = list(session.initial_messages())
        tools = session.list_tools()
        result = HarnessRolloutResult(messages=list(messages))
        total_tool_calls = 0

        for turn_index in range(run_limits.max_turns):
            step_result = model_step(messages, tools, dict(run_limits.sampling))
            assistant_message = step_result.response.to_message_dict()
            messages.append(assistant_message)

            result.prompt_ids.extend(step_result.prompt_ids)
            result.completion_ids.extend(step_result.completion_ids)
            result.logprobs.extend(step_result.logprobs)
            result.events.append(
                RolloutEvent(
                    type="model_response",
                    payload={
                        "turn": turn_index,
                        "content": step_result.response.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": dict(tool_call.args),
                            }
                            for tool_call in step_result.response.tool_calls
                        ],
                    },
                )
            )

            if not step_result.response.tool_calls:
                result.done = True
                break

            tool_calls = step_result.response.tool_calls
            if run_limits.max_tool_calls_per_turn is not None:
                tool_calls = tool_calls[: run_limits.max_tool_calls_per_turn]

            for tool_call in tool_calls:
                if (
                    run_limits.max_total_tool_calls is not None
                    and total_tool_calls >= run_limits.max_total_tool_calls
                ):
                    result.metrics["truncated"] = True
                    result.messages = list(messages)
                    return result

                tool_result = session.call_tool(tool_call.name, dict(tool_call.args))
                total_tool_calls += 1

                trace_entry = ToolTraceEntry(
                    tool_name=tool_call.name,
                    arguments=dict(tool_call.args),
                    result=tool_result,
                )
                result.tool_trace.append(trace_entry)
                result.events.append(
                    RolloutEvent(
                        type="tool_call",
                        payload={
                            "tool_name": tool_call.name,
                            "arguments": dict(tool_call.args),
                            "done": tool_result.done,
                        },
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": _serialize_for_message(tool_result.data),
                    }
                )

                if tool_result.done:
                    result.done = True
                    break

            if result.done:
                break

        result.messages = list(messages)
        result.metrics.setdefault(
            "turns",
            len([event for event in result.events if event.type == "model_response"]),
        )
        result.metrics.setdefault("tool_calls", len(result.tool_trace))
        return result

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError(
            "MCPHarnessAdapter is a white-box harness. "
            "Use CLIHarnessAdapter for opaque evaluation harnesses."
        )


class CLIHarnessAdapter(HarnessAdapter):
    """Thin black-box adapter for opaque CLI-style harnesses."""

    def __init__(
        self,
        runner: Callable[
            [SessionMCPBridge, ResourceSession, HarnessRunLimits],
            HarnessRolloutResult,
        ],
    ):
        self._runner = runner

    def run_white_box(
        self,
        model_step: ModelStep,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        raise NotImplementedError(
            "CLIHarnessAdapter only supports black-box evaluation rollouts."
        )

    def run_black_box(
        self,
        session: ResourceSession,
        limits: HarnessRunLimits | None = None,
    ) -> HarnessRolloutResult:
        run_limits = limits or HarnessRunLimits()
        bridge = SessionMCPBridge(session)
        return self._runner(bridge, session, run_limits)


def build_harness_rollout_func(
    *,
    session_factory: Any,
    harness_adapter: HarnessAdapter,
    model_step_builder: Callable[[Any, ResourceSession], ModelStep],
    limits: HarnessRunLimits | None = None,
    reward_key: str = "env_reward",
) -> Callable[[list[Any], Any], dict[str, list[Any]]]:
    """Build a TRL-compatible rollout function from sessions and harnesses."""

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, list[Any]]:
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        rewards: list[float] = []
        verify_metrics: list[dict[str, Any]] = []

        for prompt in prompts:
            session = session_factory.create(task=prompt)
            try:
                model_step = model_step_builder(trainer, session)
                rollout = harness_adapter.run_white_box(
                    model_step=model_step,
                    session=session,
                    limits=limits,
                )
                verify = session.verify(
                    transcript=rollout.messages,
                    final_state={
                        "done": rollout.done,
                        "metrics": dict(rollout.metrics),
                        "events": [
                            {
                                "type": event.type,
                                "payload": dict(event.payload),
                            }
                            for event in rollout.events
                        ],
                        "tool_trace": [
                            {
                                "tool_name": entry.tool_name,
                                "arguments": dict(entry.arguments),
                                "result": {
                                    "data": entry.result.data,
                                    "done": entry.result.done,
                                    "metadata": dict(entry.result.metadata),
                                    "error": entry.result.error,
                                },
                            }
                            for entry in rollout.tool_trace
                        ],
                    },
                )

                all_prompt_ids.append(list(rollout.prompt_ids))
                all_completion_ids.append(list(rollout.completion_ids))
                all_logprobs.append(list(rollout.logprobs))
                rewards.append(_resolve_env_reward(rollout, verify))
                verify_metrics.append(dict(verify.metrics))
            finally:
                session.close()

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            reward_key: rewards,
            "verify_metrics": verify_metrics,
        }

    return rollout_func


__all__ = [
    "CLIHarnessAdapter",
    "HarnessAdapter",
    "HarnessRolloutResult",
    "HarnessRunLimits",
    "LoopOwningSession",
    "MCPHarnessAdapter",
    "Message",
    "ModelStep",
    "ModelStepResult",
    "RESERVED_TOOL_NAMES",
    "ResourceSession",
    "ResourceSessionFactory",
    "RolloutEvent",
    "SessionMCPBridge",
    "StepEnvSessionAdapter",
    "ToolResult",
    "ToolTraceEntry",
    "TraceEntry",
    "TrainableSession",
    "VerifyResult",
    "build_harness_rollout_func",
]
