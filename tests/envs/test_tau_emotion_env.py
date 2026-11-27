# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline tests for the τ-emotion-bench environment.

The simulated user is scripted by replacing τ-emotion-bench's LLM call with
litellm's `mock_response`, so these run without a network or an API key. The
tools, the airline database and τ-emotion-bench's evaluator are the real ones.
"""

import pytest

pytest.importorskip("tau_emotion", reason="τ-emotion-bench is not installed")

import litellm
import tau_emotion.utils.llm_utils as llm_utils
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from tau_emotion_env.server.tau_emotion_environment import (
    llm_config,
    TauEmotionEnvironment,
)

TASK = "airline_test_0000"


@pytest.fixture
def scripted_user(monkeypatch):
    """Make the simulated user say each of `replies` in turn."""
    replies: list[str] = []

    def completion(**kwargs):
        return litellm.completion(
            model="openai/scripted",
            messages=kwargs["messages"],
            mock_response=replies.pop(0),
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    return replies


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return TauEmotionEnvironment(domain="airline", split="test")


def test_tools_are_the_domain_tools_plus_user_and_done(env):
    tools = {t.name: t for t in env.step(ListToolsAction()).tools}
    assert {"get_user_details", "book_reservation", "respond_to_user", "done"} <= set(
        tools
    )
    assert tools["get_user_details"].input_schema["required"] == ["user_id"]
    assert tools["respond_to_user"].input_schema["required"] == ["message"]


def test_episode_runs_until_the_user_stops(env, scripted_user):
    scripted_user += [
        "Hi, I need help with a reservation.",
        "That's all, thanks. ###STOP###",
    ]

    observation = env.reset(task_id=TASK)
    assert observation.metadata["user_message"] == "Hi, I need help with a reservation."
    assert "airline" in observation.metadata["policy"].lower()
    assert observation.metadata["task_id"] == TASK

    details = env.step(
        CallToolAction(
            tool_name="get_user_details", arguments={"user_id": "ethan_nguyen_6045"}
        )
    )
    assert '"user_id": "ethan_nguyen_6045"' in details.result.content[0].text
    assert not details.done

    final = env.step(
        CallToolAction(
            tool_name="respond_to_user", arguments={"message": "Anything else?"}
        )
    )
    assert isinstance(final, CallToolObservation)
    assert final.result.content[0].text == "That's all, thanks. ###STOP###"
    assert final.done and env.state.done
    assert final.metadata["termination_reason"] == "user_stop"
    assert final.reward == env.state.reward == 0.0  # nothing was done for the user
    assert final.metadata["reward_info"]["db_check"] is not None
    assert env.state.step_count == 2


def test_an_unsatisfied_user_scores_minus_one(env, scripted_user):
    scripted_user += [
        "Hi, I need help.",
        "Forget it, you're useless. ###UNSATISFIED###",
    ]
    env.reset(task_id=TASK)
    final = env.step(
        CallToolAction(tool_name="respond_to_user", arguments={"message": "No."})
    )
    assert final.done and final.reward == -1.0
    assert final.metadata["termination_reason"] == "user_unsatisfied"


def test_a_failed_simulation_ends_the_episode_unscored(env, scripted_user):
    scripted_user += ["Hi, I need help."]  # the user's model fails on the next turn
    env.reset(task_id=TASK)
    final = env.step(
        CallToolAction(tool_name="respond_to_user", arguments={"message": "Hello."})
    )
    assert "simulation failed" in final.error.message
    assert final.done and env.state.done
    assert final.reward is None and env.state.reward is None
    assert final.metadata["termination_reason"] == "infrastructure_error"
    assert final.metadata["reward_info"] == {}


def test_done_ends_the_episode_from_the_agents_side(env, scripted_user):
    scripted_user += ["Hi, I need help."]
    env.reset(task_id=TASK)
    final = env.step(CallToolAction(tool_name="done", arguments={}))
    assert final.done
    assert final.metadata["termination_reason"] == "agent_stop"
    assert (
        env.step(
            CallToolAction(tool_name="respond_to_user", arguments={"message": "Hi"})
        )
        .result.content[0]
        .text
        == "The conversation has ended."
    )


def test_train_tasks_run_against_the_train_database(monkeypatch, scripted_user):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    env = TauEmotionEnvironment(domain="airline", split="train")
    task_id = sorted(env.tasks)[0]
    scripted_user += ["Hi.", "Bye. ###STOP###"]
    env.reset(task_id=task_id)
    assert env._simulation.env_kwargs == {"db_split": "train"}
    final = env.step(
        CallToolAction(tool_name="respond_to_user", arguments={"message": "Bye."})
    )
    assert final.done and "reward_info" in final.metadata


def test_tools_need_a_reset_first(env):
    observation = env.step(
        CallToolAction(tool_name="get_user_details", arguments={"user_id": "x"})
    )
    assert observation.error is not None


def test_unknown_domain_split_and_task(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with pytest.raises(ValueError, match="domain"):
        TauEmotionEnvironment(domain="mock")
    with pytest.raises(ValueError, match="split"):
        TauEmotionEnvironment(domain="airline", split="base")
    with pytest.raises(ValueError, match="unknown task"):
        TauEmotionEnvironment(domain="airline").reset(task_id="airline_test_9999")


def test_providers(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    model, args = llm_config("hf")
    assert model == "openai/Qwen/Qwen3-235B-A22B-Instruct-2507"
    assert args["api_base"] == "https://router.huggingface.co/v1"
    assert args["api_key"] == "hf_test"
    assert llm_config("openai", "gpt-4.1")[0] == "openai/gpt-4.1"
    model, args = llm_config(
        "openai", "qwen3-235b-a22b-instruct-2507", api_base="http://vllm:8000/v1"
    )
    assert model == "openai/qwen3-235b-a22b-instruct-2507"
    assert args["api_base"] == "http://vllm:8000/v1"
    assert llm_config("anthropic")[0].startswith("anthropic/")
    with pytest.raises(ValueError):
        llm_config("azure")


def test_reset_needs_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = TauEmotionEnvironment(domain="airline")  # starts without it
    with pytest.raises(ValueError, match="hf_token"):
        env.reset(task_id=TASK)


def test_reset_takes_the_clients_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    keys = []

    def completion(**kwargs):
        keys.append(kwargs["api_key"])
        return litellm.completion(
            model="openai/scripted", messages=kwargs["messages"], mock_response="Hi."
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    env = TauEmotionEnvironment(domain="airline")
    env.reset(task_id=TASK, hf_token="hf_client")
    env.reset(task_id="airline_test_0001")  # kept for the session
    assert keys == ["hf_client", "hf_client"]
    assert "hf_client" not in env.state.model_dump_json()
    env.close()


def test_a_failed_reset_keeps_the_conversation(env, scripted_user, monkeypatch):
    scripted_user += ["Hi, I need help.", "Thanks. ###STOP###"]
    env.reset(task_id=TASK)

    def completion(**kwargs):
        raise litellm.AuthenticationError("bad token", "openai", "scripted")

    scripted_completion = llm_utils.completion
    monkeypatch.setattr(llm_utils, "completion", completion)
    with pytest.raises(RuntimeError, match="could not open the conversation"):
        env.reset(task_id="airline_test_0001")
    assert env.state.task_id == TASK

    monkeypatch.setattr(llm_utils, "completion", scripted_completion)
    final = env.step(
        CallToolAction(tool_name="respond_to_user", arguments={"message": "Bye."})
    )
    assert final.done and final.metadata["termination_reason"] == "user_stop"


def test_every_domain_builds(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    from tau_emotion_env.server.tau_emotion_environment import DOMAINS

    assert len(DOMAINS) == 10
    for domain in DOMAINS:
        env = TauEmotionEnvironment(domain=domain)
        assert env.tasks
        assert "respond_to_user" in {t.name for t in env.step(ListToolsAction()).tools}
