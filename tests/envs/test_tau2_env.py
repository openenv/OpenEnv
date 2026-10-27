# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline tests for the τ²-bench environment.

The simulated user is scripted by replacing τ²-bench's LLM call with litellm's
`mock_response`, so these run without a network or an API key. The tools, the
airline database and τ²-bench's evaluator are the real ones.
"""

import importlib
import threading
import time

import pytest

pytest.importorskip("tau2", reason="τ²-bench is not installed")

import litellm
import tau2.evaluator.evaluator_nl_assertions as nl_assertions
import tau2.utils.llm_utils as llm_utils
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from tau2_env.server.tau2_environment import llm_config, Tau2Environment


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
    return Tau2Environment(domain="airline", split="test")


def test_tools_are_the_domain_tools_plus_user_and_done(env):
    tools = {t.name: t for t in env.step(ListToolsAction()).tools}
    assert {"get_user_details", "book_reservation", "respond_to_user", "done"} <= set(
        tools
    )
    assert tools["get_user_details"].input_schema["required"] == ["user_id"]
    assert tools["respond_to_user"].input_schema["required"] == ["message"]


def test_episode_runs_until_the_user_stops(env, scripted_user):
    scripted_user += [
        "Hi, my user id is noah_muller_9847.",
        "That's all, thanks. ###STOP###",
    ]

    observation = env.reset(task_id="2")
    assert observation.metadata["user_message"] == "Hi, my user id is noah_muller_9847."
    assert "airline" in observation.metadata["policy"].lower()

    details = env.step(
        CallToolAction(
            tool_name="get_user_details", arguments={"user_id": "noah_muller_9847"}
        )
    )
    assert '"user_id": "noah_muller_9847"' in details.result.content[0].text
    assert not details.done

    final = env.step(
        CallToolAction(
            tool_name="respond_to_user", arguments={"message": "Anything else?"}
        )
    )
    assert isinstance(final, CallToolObservation)
    assert final.result.content[0].text == "That's all, thanks. ###STOP###"
    assert final.done and env.state.done
    assert final.reward == env.state.reward
    assert final.metadata["reward_info"]["reward_basis"] == ["DB", "COMMUNICATE"]
    assert env.state.step_count == 2


def test_tools_need_a_reset_first(env):
    observation = env.step(
        CallToolAction(tool_name="get_user_details", arguments={"user_id": "x"})
    )
    assert observation.error is not None


def test_providers(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    model, args = llm_config("hf")
    assert model == "openai/deepseek-ai/DeepSeek-V4.1-Flash"
    assert args["api_base"] == "https://router.huggingface.co/v1"
    assert args["api_key"] == "hf_test"
    assert llm_config("openai", "gpt-4.1")[0] == "openai/gpt-4.1"
    assert llm_config("anthropic")[0].startswith("anthropic/")
    with pytest.raises(ValueError):
        llm_config("azure")


def test_reset_needs_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = Tau2Environment(domain="airline")  # starts without it
    with pytest.raises(ValueError, match="hf_token"):
        env.reset(task_id="0")


def test_reset_takes_the_clients_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    keys = []

    def completion(**kwargs):
        keys.append(kwargs["api_key"])
        return litellm.completion(
            model="openai/scripted", messages=kwargs["messages"], mock_response="Hi."
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    env = Tau2Environment(domain="airline")
    env.reset(task_id="0", hf_token="hf_client")
    env.reset(task_id="1")  # kept for the session
    assert keys == ["hf_client", "hf_client"]
    assert "hf_client" not in env.state.model_dump_json()
    env.close()


def test_a_failed_reset_keeps_the_conversation(env, scripted_user, monkeypatch):
    scripted_user += ["Hi, my user id is noah_muller_9847.", "Thanks. ###STOP###"]
    env.reset(task_id="2")

    def completion(**kwargs):
        raise litellm.AuthenticationError("bad token", "openai", "scripted")

    scripted_completion = llm_utils.completion
    monkeypatch.setattr(llm_utils, "completion", completion)
    with pytest.raises(RuntimeError, match="could not open the conversation"):
        env.reset(task_id="0")

    monkeypatch.setattr(llm_utils, "completion", scripted_completion)
    observation = env.step(
        CallToolAction(tool_name="respond_to_user", arguments={"message": "Done."})
    )
    assert env.state.task_id == "2" and observation.done


def test_each_conversation_has_its_own_judge(monkeypatch):
    """Two conversations judged at the same time each use their own token."""
    judged = {}
    monkeypatch.setattr(
        "tau2_env.server.tau2_environment.generate",
        lambda **kwargs: judged.setdefault(threading.current_thread().name, kwargs),
    )
    both_stepping = threading.Barrier(2)

    class JudgingGym:  # τ²-bench judges the conversation inside its last step
        def step(self, action):
            both_stepping.wait()
            nl_assertions.generate(model="default", messages=[], call_name="judge")
            return "user: bye", 1.0, True, False, {"reward_info": '{"reward": 1.0}'}

    def converse(token):
        env = Tau2Environment(domain="airline", hf_token=token)
        env._gym = JudgingGym()
        env.act("Goodbye")

    threads = [
        threading.Thread(target=converse, args=(t,), name=t) for t in ("hf_a", "hf_b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert {name: kwargs["api_key"] for name, kwargs in judged.items()} == {
        "hf_a": "hf_a",
        "hf_b": "hf_b",
    }


def test_a_failed_user_ends_the_conversation_without_a_score(env, monkeypatch):
    replies = iter(["Hi, my user id is noah_muller_9847."])

    def completion(**kwargs):
        reply = next(replies, None)
        if reply is None:
            raise litellm.APIConnectionError("the provider is down", "openai", "x")
        return litellm.completion(
            model="openai/scripted", messages=kwargs["messages"], mock_response=reply
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    import tau2_env.server.gradio_ui as ui

    episode = ui.Episode(env, "2")
    observation, result = episode.act("respond_to_user", {"message": "Hello."})
    assert observation.error is not None
    assert "simulation failed" in result
    assert env.state.done and env.state.reward_info == {}
    # The web UI says so instead of showing a score.
    assert "Ended without a score" in episode.side(observation)


def test_tool_arguments_must_be_a_json_object():
    import tau2_env.server.gradio_ui as ui

    assert ui.parse_arguments('{"user_id": "x"}') == {"user_id": "x"}
    assert ui.parse_arguments(None) == {}
    for raw in ["not json", "[]", '"x"', "3"]:
        assert ui.parse_arguments(raw) is None


def test_ui_falls_back_when_no_model_calls_tools(monkeypatch):
    import io

    import tau2_env.server.gradio_ui as ui

    monkeypatch.setattr(
        ui.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b'{"data": []}')
    )
    # Uncached, so the test leaves the real model list alone.
    assert ui.model_choices.__wrapped__() == [(m, m) for m in ui.FALLBACK_MODELS]


def test_a_failed_reply_is_not_shown_as_the_customer(env, scripted_user, monkeypatch):
    import tau2_env.server.gradio_ui as ui

    scripted_user.append("Hi, my user id is noah_muller_9847.")
    episode = ui.Episode(env, "2")

    def fail(action):
        raise TimeoutError("the customer did not answer")

    monkeypatch.setattr(env, "act", fail)
    episode.act("respond_to_user", {"message": "Anything else?"})
    agent, failed = episode.events[-2:]
    assert agent == {"kind": "agent", "text": "Anything else?"}
    assert failed["kind"] == "tool" and failed["result"].startswith("Error")


def test_server_and_ui_start_without_credentials(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")
    import tau2_env.server.app as server
    import tau2_env.server.gradio_ui as ui

    # The model list comes from the Inference Providers router; keep the test offline.
    monkeypatch.setattr(
        ui, "model_choices", lambda: [(m, m) for m in ui.FALLBACK_MODELS]
    )
    from fastapi.testclient import TestClient

    client = TestClient(importlib.reload(server).app)
    assert client.get("/health").status_code == 200
    assert client.get("/web/").status_code == 200
    # The sign-in routes Gradio expects at the root forward to the UI under /web.
    login = client.get("/login/huggingface?_target_url=/", follow_redirects=False)
    assert login.headers["location"] == "/web/login/huggingface?_target_url=/"


def test_done_ends_the_conversation(env, scripted_user):
    scripted_user.append("Hi, I'd like to change my flight.")
    env.reset(task_id="2")
    observation = env.step(CallToolAction(tool_name="done", arguments={}))
    assert observation.done and env.state.done


def test_tools_after_the_end_keep_the_score(env, scripted_user):
    scripted_user.append("Hi, I'd like to change my flight.")
    env.reset(task_id="2")
    env.step(CallToolAction(tool_name="done", arguments={}))
    reward, reward_info = env.state.reward, env.state.reward_info

    observation = env.step(
        CallToolAction(
            tool_name="get_user_details", arguments={"user_id": "noah_muller_9847"}
        )
    )
    assert observation.result.data == "The conversation has ended."
    assert (env.state.reward, env.state.reward_info) == (reward, reward_info)


def test_reset_ends_the_previous_conversation(env, scripted_user):
    """An abandoned conversation stops its thread without another customer call."""
    scripted_user.extend(["Hi, I'd like to change my flight."] * 3)
    env.reset(task_id="2")
    before = threading.active_count()
    env.reset(task_id="2")
    env.reset(task_id="2")
    env.close()
    env.close()  # closing twice is fine
    for _ in range(50):
        if threading.active_count() < before:
            break
        time.sleep(0.1)
    assert threading.active_count() < before
    assert scripted_user == []
