# SPDX-License-Identifier: BSD-3-Clause

"""Tests for EnvEvalHarness."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import openai
import pydantic
import pytest
from openenv.core.env_client import SessionLostError
from openenv.core.env_server.mcp_types import Tool
from openenv.core.evals import EnvEvalHarness, EvalConfig, EvalResult
from openenv.core.harness import (
    MCPHarnessAdapter,
    ModelStepResult,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.llm_client import LLMResponse, ToolCall
from websockets.exceptions import ConnectionClosed

ANSWER_TOOL = Tool(
    name="answer",
    description="Submit the final answer.",
    input_schema={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
)

FORWARD = object()  # verify forwards the last step's reward


class FakeSession(ResourceSession):
    """
    A single-question environment whose rewards are computed inside it.

    A task spec may carry, besides `id`:
    - `answer`: the correct answer (reward 1.0, else 0.0, episode ends);
    - `steps`: a list of `(reward, done)` returned step by step instead;
    - `step_error` / `verify_error`: an exception raised by that call;
    - `create_error`: an exception the factory raises for an episode;
    - `verify_reward`: what `verify` reports instead of the last step's reward.
    """

    def __init__(self, factory: FakeFactory, task: Any, seed: int | None) -> None:
        self.factory = factory
        self.task = task
        self.seed = seed
        self.rewards: list[float | None] = []
        self.closed = 0

    def initial_messages(self) -> list[dict[str, Any]]:
        content = f"question {self.task['id']} seed {self.seed}"
        return [{"role": "user", "content": content}]

    def list_tools(self) -> list[Tool]:
        return [ANSWER_TOOL]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if "step_error" in self.task:
            raise self.task["step_error"]
        if name != "answer":
            raise KeyError(f"Unknown tool: {name}")
        if "steps" in self.task:
            reward, done = self.task["steps"][len(self.rewards)]
        else:
            reward = 1.0 if arguments["answer"] == self.task["answer"] else 0.0
            done = True
        self.rewards.append(reward)
        return ToolResult(
            data={"reward": reward}, done=done, metadata={"reward": reward}
        )

    def verify(
        self, transcript: list[dict[str, Any]], final_state: Any | None = None
    ) -> VerifyResult:
        if "verify_error" in self.task:
            raise self.task["verify_error"]
        reward = self.task.get("verify_reward", FORWARD)
        if reward is FORWARD:
            reward = self.rewards[-1] if self.rewards else None
        return VerifyResult(env_reward=reward, done=True)

    def close(self) -> None:
        self.closed += 1
        if self.factory.close_error is not None:
            raise self.factory.close_error


class FakeFactory(ResourceSessionFactory):
    def __init__(
        self,
        *,
        close_error: Exception | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.close_error = close_error
        self.create_error = create_error
        self.calls: list[tuple[str, int | None, str | None]] = []
        self.sessions: list[FakeSession] = []

    def create(
        self, task: Any, seed: int | None = None, episode_id: str | None = None
    ) -> FakeSession:
        self.calls.append((task["id"], seed, episode_id))
        if self.create_error is not None:
            raise self.create_error
        if "create_error" in task and episode_id is not None:
            raise task["create_error"]
        session = FakeSession(self, task, seed)
        self.sessions.append(session)
        return session

    @property
    def episode_calls(self) -> list[tuple[str, int | None, str | None]]:
        return [call for call in self.calls if call[2] is not None]


class Policy:
    """
    A model step that answers by task id.

    `answers[task_id]` is one reply or a list consumed one model call at a
    time. A reply is a string (the `answer` tool call), `None` (plain text, no
    tool call), `("tool", name)` (a call to another tool) or an exception.
    """

    def __init__(
        self, answers: dict[str, Any], *, completion_ids: list[int] | None = None
    ) -> None:
        self.answers = {
            key: list(value) if isinstance(value, list) else value
            for key, value in answers.items()
        }
        self.completion_ids = completion_ids or []
        self.calls = 0
        self.sampling: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        sampling: dict[str, Any],
    ) -> ModelStepResult:
        self.calls += 1
        self.sampling.append(dict(sampling))
        task_id = messages[0]["content"].split()[1]
        reply = self.answers[task_id]
        if isinstance(reply, list):
            reply = reply.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if reply is None:
            response = LLMResponse(content="I think it is 4.")
        elif isinstance(reply, tuple):
            response = LLMResponse(
                content="", tool_calls=[ToolCall(id="c", name=reply[1], args={})]
            )
        else:
            response = LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c", name="answer", args={"answer": reply})],
            )
        return ModelStepResult(
            response=response, completion_ids=list(self.completion_ids)
        )


def task(task_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": task_id, "answer": "4", **fields}


def harness(
    tasks: list[dict[str, Any]],
    policy: Policy,
    factory: FakeFactory | None = None,
) -> tuple[EnvEvalHarness, FakeFactory]:
    factory = factory or FakeFactory()
    return (
        EnvEvalHarness(
            factory,
            MCPHarnessAdapter(),
            tasks,
            model_step=policy,
        ),
        factory,
    )


def run(evaluator: EnvEvalHarness, **params: Any) -> dict[str, Any]:
    return evaluator.run("0.1", {}, "fake", params)


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://llm/v1/chat/completions")


def _status(cls: type, code: int) -> Exception:
    return cls("m", response=httpx.Response(code, request=_request()), body=None)


def _validation_error() -> Exception:
    try:
        pydantic.TypeAdapter(int).validate_python("x")
    except pydantic.ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


INFRA_ERRORS = [
    ConnectionError("refused"),
    SessionLostError("lost"),
    ConnectionClosed(None, None),
    openai.APIConnectionError(request=_request()),
    openai.APITimeoutError(request=_request()),
    _status(openai.AuthenticationError, 401),
    _status(openai.PermissionDeniedError, 403),
    _status(openai.NotFoundError, 404),
    _status(openai.RateLimitError, 429),
    _status(openai.InternalServerError, 503),
]

POLICY_ERRORS = [
    json.JSONDecodeError("Expecting value", "{", 1),
    KeyError("Unknown tool: x"),
    ValueError("bad action"),
    _validation_error(),
    _status(openai.BadRequestError, 400),
    _status(openai.UnprocessableEntityError, 422),
    _status(openai.ConflictError, 409),
    RuntimeError("Server error: step failed"),
    TimeoutError(),
    NotImplementedError(),
]


class TestTasks:
    @pytest.mark.parametrize(
        "tasks, message",
        [
            ([], "empty"),
            ([1], "position 0"),
            ([{"split": "test"}], "position 0"),
            ([{"id": 3}], "position 0"),
            ([{"id": "a"}, {"id": "a"}], "'a'"),
        ],
    )
    def test_constructor_rejects_bad_task_lists(
        self, tasks: list[Any], message: str
    ) -> None:
        factory = FakeFactory()
        with pytest.raises(ValueError, match=message):
            EnvEvalHarness(factory, MCPHarnessAdapter(), tasks, model_step=Policy({}))
        assert factory.calls == []


class TestParameters:
    def test_defaults_reach_factory_and_model_step(self) -> None:
        policy = Policy({"t0": "4", "t1": "4"})
        evaluator, factory = harness([task("t0"), task("t1")], policy)
        run(evaluator)
        assert factory.episode_calls == [
            ("t0", 0, "ep-000000"),
            ("t1", 0, "ep-000001"),
        ]
        assert policy.sampling[0] == {"temperature": 0.0, "max_tokens": 200}

    def test_default_turn_limit_is_nine(self) -> None:
        never_done = task("t0", steps=[(0.0, False)] * 20)
        policy = Policy({"t0": "4"})
        evaluator, _ = harness([never_done], policy)
        run(evaluator)
        assert policy.calls == 9

    @pytest.mark.parametrize(
        "params, key",
        [
            ({"limit": 0}, "limit"),
            ({"limit": True}, "limit"),
            ({"limit": 1.5}, "limit"),
            ({"seeds": []}, "seeds"),
            ({"seeds": [0, 0]}, "seeds"),
            ({"seeds": [None]}, "seeds"),
            ({"seeds": 0}, "seeds"),
            ({"seeds": ["0"]}, "seeds"),
            ({"samples_per_task": 0}, "samples_per_task"),
            ({"samples_per_task": True}, "samples_per_task"),
            ({"temperature": -0.1}, "temperature"),
            ({"temperature": float("nan")}, "temperature"),
            ({"temperature": "0"}, "temperature"),
            ({"temperature": 0.0, "samples_per_task": 2}, "temperature"),
            ({"success_threshold": float("inf")}, "success_threshold"),
            ({"success_threshold": None}, "success_threshold"),
            ({"max_turns": 0}, "max_turns"),
            ({"max_tokens": True}, "max_tokens"),
        ],
    )
    def test_invalid_values_raise_before_anything(
        self, params: dict[str, Any], key: str
    ) -> None:
        evaluator, factory = harness([task("t0")], Policy({"t0": "4"}))
        with pytest.raises(ValueError, match=key):
            evaluator.run("0.1", {}, "fake", params)
        assert factory.calls == []

    def test_unknown_keys_are_ignored_and_recorded(self) -> None:
        evaluator, _ = harness([task("t0")], Policy({"t0": "4"}))
        params = {"model": "m", "llm_endpoint": "http://llm/v1", "split": "test"}
        config = EvalConfig(
            harness_name=evaluator.name,
            harness_version="0.1",
            library_versions={},
            dataset="fake",
            eval_parameters=params,
        )
        result = evaluator.run_from_config(config)
        assert result.scores["success_rate"] == 1.0
        assert result.config.eval_parameters == params

    def test_limit_and_episode_order(self) -> None:
        tasks = [task("t0"), task("t1"), task("t2")]
        evaluator, factory = harness(tasks, Policy({"t0": "4", "t1": "4"}))
        run(evaluator, limit=2, seeds=[3, 5], samples_per_task=2, temperature=0.7)
        assert factory.episode_calls == [
            ("t0", 3, "ep-000000"),
            ("t0", 3, "ep-000001"),
            ("t0", 5, "ep-000002"),
            ("t0", 5, "ep-000003"),
            ("t1", 3, "ep-000004"),
            ("t1", 3, "ep-000005"),
            ("t1", 5, "ep-000006"),
            ("t1", 5, "ep-000007"),
        ]

    def test_limit_above_the_task_count_keeps_all_tasks(self) -> None:
        tasks = [task("t0"), task("t1"), task("t2")]
        evaluator, _ = harness(tasks, Policy({"t0": "4", "t1": "4", "t2": "4"}))
        assert run(evaluator, limit=10)["num_tasks"] == 3


class TestOutcomes:
    def test_scored_partial_credit_and_threshold(self) -> None:
        partial = task("t0", steps=[(0.5, True)])
        evaluator, _ = harness([partial], Policy({"t0": "4"}))
        scores = run(evaluator)
        assert scores["success_rate"] == 0.0
        assert scores["mean_reward"] == 0.5
        assert run(evaluator, success_threshold=0.5)["success_rate"] == 1.0

    def test_reward_is_never_rounded(self) -> None:
        reward = 0.8333333333333334
        evaluator, _ = harness(
            [task("t0", steps=[(reward, True)])], Policy({"t0": "4"})
        )
        assert run(evaluator)["mean_reward"] == reward

    def test_env_ended_with_none_is_unscored(self) -> None:
        tasks = [task("t0"), task("t1", steps=[(None, True)])]
        evaluator, _ = harness(tasks, Policy({"t0": "4", "t1": "4"}))
        scores = run(evaluator)
        assert scores["unscored_rate"] == 0.5
        assert scores["success_rate"] == 1.0
        assert scores["mean_reward"] == 1.0
        assert scores["pass@1"] == 1.0
        assert scores["pass@1_num_tasks"] == 1

    def test_final_none_after_earlier_zero_is_unscored(self) -> None:
        late_failure = task("t0", steps=[(0.0, False), (None, True)])
        evaluator, _ = harness([late_failure], Policy({"t0": "4"}))
        scores = run(evaluator)
        assert scores["unscored_rate"] == 1.0
        assert scores["mean_reward"] is None

    def test_final_none_with_verify_reward_is_scored(self) -> None:
        graded_later = task("t0", steps=[(None, True)], verify_reward=0.75)
        evaluator, _ = harness([graded_later], Policy({"t0": "4"}))
        scores = run(evaluator)
        assert scores["mean_reward"] == 0.75
        assert scores["unscored_rate"] == 0.0

    def test_reply_without_tool_call_is_policy_error(self) -> None:
        tasks = [task("t0"), task("t1")]
        evaluator, _ = harness(tasks, Policy({"t0": "4", "t1": None}))
        scores = run(evaluator)
        assert scores["error_rate"] == 0.5
        assert scores["success_rate"] == 0.5
        assert scores["pass@1"] == 0.5
        assert scores["mean_reward"] == 1.0

    def test_turn_limit_after_rewarded_step_is_policy_error(self) -> None:
        multi_turn = task("t0", steps=[(0.5, False), (1.0, True)])
        evaluator, _ = harness([multi_turn], Policy({"t0": "4"}))
        scores = run(evaluator, max_turns=1)
        assert scores["error_rate"] == 1.0
        assert scores["success_rate"] == 0.0
        assert scores["mean_reward"] is None

    def test_verify_done_is_ignored(self) -> None:
        not_ended = task("t0", steps=[(0.0, False)] * 2, verify_reward=1.0)
        evaluator, _ = harness([not_ended], Policy({"t0": "4"}))
        scores = run(evaluator, max_turns=2)
        assert scores["error_rate"] == 1.0
        assert scores["success_rate"] == 0.0

    def test_call_to_unknown_tool_is_policy_error(self) -> None:
        evaluator, _ = harness([task("t0")], Policy({"t0": ("tool", "search")}))
        scores = run(evaluator)
        assert scores["error_rate"] == 1.0
        assert scores["success_rate"] == 0.0

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("Server error: capacity"), ValueError("bad index")]
        + INFRA_ERRORS[:1],
        ids=lambda e: type(e).__name__,
    )
    def test_create_failures_are_infra(self, error: Exception) -> None:
        tasks = [task("t0"), task("t1"), task("t2", create_error=error)]
        evaluator, _ = harness(tasks, Policy({"t0": "4", "t1": "x"}))
        scores = run(evaluator)
        assert scores["infra_error_rate"] == pytest.approx(1 / 3)
        assert scores["success_rate"] == 0.5

    @pytest.mark.parametrize("where", ["model_step", "call_tool", "verify"])
    @pytest.mark.parametrize("error", INFRA_ERRORS, ids=lambda e: type(e).__name__)
    def test_infra_exceptions(self, where: str, error: Exception) -> None:
        scores = self._run_raising(where, error)
        assert scores["infra_error_rate"] == 0.5
        assert scores["error_rate"] == 0.0
        assert scores["success_rate"] == 1.0

    @pytest.mark.parametrize("where", ["model_step", "call_tool", "verify"])
    @pytest.mark.parametrize("error", POLICY_ERRORS, ids=lambda e: type(e).__name__)
    def test_policy_exceptions(self, where: str, error: Exception) -> None:
        scores = self._run_raising(where, error)
        assert scores["error_rate"] == 0.5
        assert scores["infra_error_rate"] == 0.0
        assert scores["success_rate"] == 0.5

    def _run_raising(self, where: str, error: Exception) -> dict[str, Any]:
        failing = task("t1")
        answer: Any = "4"
        if where == "model_step":
            answer = error
        elif where == "call_tool":
            failing["step_error"] = error
        else:
            failing["verify_error"] = error
        evaluator, _ = harness([task("t0"), failing], Policy({"t0": "4", "t1": answer}))
        return run(evaluator)

    def test_reward_mismatch_is_infra(self) -> None:
        mismatch = task("t0", verify_reward=0.0)
        evaluator, _ = harness([mismatch], Policy({"t0": "4"}))
        assert run(evaluator)["infra_error_rate"] == 1.0

    def test_mismatch_tolerance_is_upstreams(self) -> None:
        close = task("t0", verify_reward=1.0 + 1e-12)
        evaluator, _ = harness([close], Policy({"t0": "4"}))
        assert run(evaluator)["mean_reward"] == 1.0

    def test_keyboard_interrupt_propagates_and_session_closes(self) -> None:
        policy = Policy({"t0": KeyboardInterrupt()})
        evaluator, factory = harness([task("t0")], policy)
        with pytest.raises(KeyboardInterrupt):
            run(evaluator)
        assert all(session.closed == 1 for session in factory.sessions)

    def test_every_session_is_closed_once(self) -> None:
        tasks = [
            task("t0"),
            task("t1"),
            task("t2", steps=[(None, True)]),
            task("t3", step_error=ConnectionError()),
            task("t4", verify_error=ValueError()),
        ]
        policy = Policy({"t0": "4", "t1": None, "t2": "4", "t3": "4", "t4": "4"})
        evaluator, factory = harness(tasks, policy)
        run(evaluator)
        assert [session.closed for session in factory.sessions] == [1] * len(
            factory.sessions
        )

    def test_close_errors_propagate(self) -> None:
        factory = FakeFactory(close_error=RuntimeError("close failed"))
        evaluator, _ = harness([task("t0")], Policy({"t0": "4"}), factory)
        with pytest.raises(RuntimeError, match="close failed"):
            run(evaluator)


class TestScores:
    def test_rates_and_denominators(self) -> None:
        tasks = [
            task("solved"),
            task("wrong"),
            task("silent"),
            task("unscored", steps=[(None, True)]),
            task("lost", create_error=ConnectionError()),
        ]
        policy = Policy({"solved": "4", "wrong": "5", "silent": None, "unscored": "4"})
        evaluator, _ = harness(tasks, policy)
        scores = run(evaluator)
        p = 1 / 3
        assert scores["num_tasks"] == 5
        assert scores["num_episodes"] == 5
        assert scores["success_rate"] == pytest.approx(p)
        assert scores["success_rate_stderr"] == pytest.approx(
            math.sqrt(p * (1 - p) / 3)
        )
        assert scores["mean_reward"] == 0.5
        assert scores["error_rate"] == 0.2
        assert scores["unscored_rate"] == 0.2
        assert scores["infra_error_rate"] == 0.2

    def test_nothing_to_average_gives_none_not_nan(self) -> None:
        tasks = [task("t0", create_error=ConnectionError())]
        evaluator, _ = harness(tasks, Policy({}))
        scores = run(evaluator)
        for key in (
            "success_rate",
            "success_rate_stderr",
            "mean_reward",
            "mean_turns",
            "mean_completion_tokens",
            "pass@1",
        ):
            assert scores[key] is None, key
        assert scores["pass@1_num_tasks"] == 0
        assert scores["per_task"] == {"t0": "left_out"}
        json.dumps(scores, allow_nan=False)

    def test_pass_at_k_matches_unbiased_estimator(self) -> None:
        a = task("a")
        b = task("b", steps=[(None, True)])
        c = task("c", steps=[(None, True)])
        # b: two solved, one policy error, one unscored (the env's own step list
        # is consumed per session, so b's sessions get different scripts below).
        policy = Policy(
            {
                "a": ["4", "5", "5", "5"],
                "b": ["4", "4", None, "4"],
                "c": ["4"] * 4,
            }
        )
        b_steps = iter([[(1.0, True)], [(1.0, True)], [(0.0, True)], [(None, True)]])

        class Factory(FakeFactory):
            def create(
                self, task: Any, seed: int | None = None, episode_id: str | None = None
            ) -> FakeSession:
                session = super().create(task, seed, episode_id)
                if task["id"] == "b" and episode_id is not None:
                    session.task = {**task, "steps": next(b_steps)}
                return session

        evaluator, _ = harness([a, b, c], policy, Factory())
        scores = run(evaluator, samples_per_task=4, temperature=0.7)

        def estimate(n: int, solved: int, k: int) -> float:
            return 1 - math.comb(n - solved, k) / math.comb(n, k)

        for k in range(1, 5):
            values = [estimate(4, 1, k)]
            if k <= 3:
                values.append(estimate(3, 2, k))
            assert scores[f"pass@{k}"] == pytest.approx(sum(values) / len(values))
            assert scores[f"pass@{k}_num_tasks"] == len(values)
        assert scores["per_task"] == {
            "a": "not_solved",
            "b": "not_solved",
            "c": "left_out",
        }

    def test_breakdowns(self) -> None:
        tasks = [
            task("t0", difficulty=2, group="x"),
            task("t1", difficulty=1, group="y"),
            task("t2", difficulty=2),
            task("t3", difficulty=3, steps=[(None, True)]),
            task("t4"),
        ]
        policy = Policy({"t0": "4", "t1": "4", "t2": "5", "t3": "4", "t4": "4"})
        evaluator, _ = harness(tasks, policy)
        scores = run(evaluator)
        assert list(scores["by_difficulty"]) == ["2", "1", "3"]
        assert scores["by_difficulty"] == {"2": 0.5, "1": 1.0, "3": None}
        assert scores["by_group"] == {"x": 1.0, "y": 1.0}

    def test_breakdowns_are_empty_without_keys(self) -> None:
        evaluator, _ = harness([task("t0")], Policy({"t0": "4"}))
        scores = run(evaluator)
        assert scores["by_difficulty"] == {}
        assert scores["by_group"] == {}

    def test_per_task_rule_over_several_episodes(self) -> None:
        tasks = [
            task("all_solved"),
            task("one_silent"),
            task("half_unscored"),
            task("all_left_out", create_error=ConnectionError()),
        ]
        unscored_first = iter([[(None, True)], [(1.0, True)]])

        class Factory(FakeFactory):
            def create(
                self, task: Any, seed: int | None = None, episode_id: str | None = None
            ) -> FakeSession:
                session = super().create(task, seed, episode_id)
                if task["id"] == "half_unscored" and episode_id is not None:
                    session.task = {**task, "steps": next(unscored_first)}
                return session

        policy = Policy(
            {"all_solved": "4", "one_silent": ["4", None], "half_unscored": "4"}
        )
        evaluator, _ = harness(tasks, policy, Factory())
        scores = run(evaluator, seeds=[0, 1])
        assert scores["per_task"] == {
            "all_solved": "solved",
            "one_silent": "not_solved",
            "half_unscored": "solved",
            "all_left_out": "left_out",
        }

    def test_turns_and_tokens_average_over_finished_rollouts(self) -> None:
        tasks = [
            task("t0", steps=[(0.0, False), (1.0, True)]),
            task("t1", steps=[(None, True)]),
            task("t2"),
            task("t3", step_error=ConnectionError()),
        ]
        policy = Policy(
            {"t0": "4", "t1": "4", "t2": None, "t3": "4"}, completion_ids=[1, 2, 3, 4]
        )
        evaluator, _ = harness(tasks, policy)
        scores = run(evaluator)
        assert scores["mean_turns"] == pytest.approx(4 / 3)
        assert scores["mean_completion_tokens"] == pytest.approx(16 / 3)

    def test_tokens_are_none_when_one_rollout_has_no_ids(self) -> None:
        class SomeIds(Policy):
            def __call__(
                self,
                messages: list[dict[str, Any]],
                tools: list[Tool],
                sampling: dict[str, Any],
            ) -> ModelStepResult:
                result = super().__call__(messages, tools, sampling)
                if "t1" in messages[0]["content"]:
                    result.completion_ids = []
                return result

        policy = SomeIds({"t0": "4", "t1": "4"}, completion_ids=[1, 2])
        evaluator, _ = harness([task("t0"), task("t1")], policy)
        assert run(evaluator)["mean_completion_tokens"] is None

    def test_scores_round_trip_as_eval_result(self) -> None:
        tasks = [task("t0", difficulty=0), task("t1", difficulty=1)]
        evaluator, _ = harness(tasks, Policy({"t0": "4", "t1": None}))
        config = EvalConfig(
            harness_name=evaluator.name,
            harness_version="0.1",
            library_versions={},
            dataset="fake",
            eval_parameters={"seeds": [0]},
        )
        result = evaluator.run_from_config(config)
        assert EvalResult.model_validate_json(result.model_dump_json()) == result


class TestExports:
    def test_package_exports(self) -> None:
        from openenv.core.evals import EnvEvalHarness, InspectAIHarness  # noqa: F401
