# SPDX-License-Identifier: BSD-3-Clause

"""Offline tests for the Claude Code harness evaluation example (RFC 005).

`fake_claude_code.py` plays Claude Code: it speaks the same stream-json events
and calls the environment's tools over the real MCP bridge. The simulated
customer is scripted, so the adapter, `HarnessEnvironment` and a real τ²-bench
airline task run end to end without the CLI, a model or an API key.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("tau2", reason="τ²-bench is not installed")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))
sys.path.insert(0, str(_REPO_ROOT / "examples" / "claude_code_harness_eval"))

import litellm
import tau2.utils.llm_utils as llm_utils
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.types import Observation
from openenv.core.harness import HarnessAction, HarnessConfig
from tau2_env.server.tau2_environment import Tau2Environment
from tau2_harness import AGENT_PROMPT, Tau2Harness

FAKE_CLAUDE = Path(__file__).parent / "fake_claude_code.py"


@pytest.fixture
def customer(monkeypatch):
    """Make the simulated customer say each of `replies` in turn."""
    replies: list[str] = []

    def completion(**kwargs):
        return litellm.completion(
            model="openai/scripted",
            messages=kwargs["messages"],
            mock_response=replies.pop(0),
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return replies


def start(tmp_path: Path, session_timeout_s: float = 30.0):
    tau2 = Tau2Environment(domain="airline", split="test")
    config = HarnessConfig(
        name="claude-code",
        command=[sys.executable, "-u", str(FAKE_CLAUDE)],
        working_directory=str(tmp_path),
        env_vars={"FAKE_CLAUDE_ARGV": str(tmp_path / "argv.json")},
        model="haiku",
        session_timeout_s=session_timeout_s,
    )
    return tau2, Tau2Harness(tau2, "2", config)


def test_conversation_runs_through_the_adapter(tmp_path, customer):
    customer += [
        "Hi, my user id is noah_muller_9847.",
        "That's all, thanks. ###STOP###",
    ]
    tau2, harness = start(tmp_path)
    try:
        opening = harness.reset()
        turn = harness.step(HarnessAction(message=opening.metadata["customer"]))
    finally:
        harness.close()

    # Only the domain's tools reach Claude Code; it talks to the customer through its replies.
    injected = opening.metadata["injected_tools"]
    assert "get_user_details" in injected
    assert "respond_to_user" not in injected and "done" not in injected

    assert opening.metadata["customer"] == "Hi, my user id is noah_muller_9847."
    call, result = turn.metadata["turn_events"][:2]
    assert call["data"] == {
        "tool_name": "get_user_details",
        "arguments": {"user_id": "noah_muller_9847"},
    }
    assert '"user_id": "noah_muller_9847"' in result["data"]["result"]

    # Claude Code's reply went to the customer, who ended the conversation, and the
    # rubric put τ²-bench's score in the observation.
    assert turn.metadata["customer"] == "That's all, thanks."
    assert turn.done and tau2.state.done
    assert turn.reward == tau2.state.reward == 1.0
    assert tau2.state.reward_info["reward_basis"] == ["DB", "COMMUNICATE"]

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1] == "mcp__env"
    assert argv[argv.index("--append-system-prompt") + 1].startswith(AGENT_PROMPT)
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in argv


def test_each_reset_starts_the_task_again(tmp_path, customer):
    customer += [
        "Hi, my user id is noah_muller_9847.",
        "That's all, thanks. ###STOP###",
        "Hi again, my user id is noah_muller_9847.",
    ]
    tau2, harness = start(tmp_path)
    try:
        opening = harness.reset()
        harness.step(HarnessAction(message=opening.metadata["customer"]))
        assert tau2.state.done
        again = harness.reset()
    finally:
        harness.close()

    assert again.metadata["customer"] == "Hi again, my user id is noah_muller_9847."
    assert not tau2.state.done


def test_production_mode_talks_to_claude_code(tmp_path, customer):
    """`WS /harness`, as `serve.py` serves it: a person is the customer, nothing is scored."""
    customer += ["Hi, my user id is noah_muller_9847."]  # tau2's reset() opens with it
    app = create_fastapi_app(
        lambda: start(tmp_path)[1], HarnessAction, Observation, mode="production"
    )
    with TestClient(app).websocket_connect("/harness") as websocket:
        assert websocket.receive_json()["type"] == "session_started"
        websocket.send_json({"type": "message", "content": "What do I have booked?"})
        while (frame := websocket.receive_json())["type"] != "turn_complete":
            pass

    assert frame["data"]["response"].startswith("get_user_details said")
    assert not customer  # the customer only opened the conversation


def test_an_api_error_ends_the_turn_as_a_harness_failure(tmp_path, customer):
    customer += ["api error"]
    tau2, harness = start(tmp_path)
    try:
        opening = harness.reset()
        turn = harness.step(HarnessAction(message=opening.metadata["customer"]))
    finally:
        harness.close()

    assert turn.done and not tau2.state.done
    assert turn.metadata["error_type"] == "harness_crashed"
    assert "ECONNRESET" in turn.metadata["error"]


def test_a_failed_customer_ends_the_conversation_as_an_error(
    tmp_path, customer, monkeypatch
):
    customer += ["Hi, my user id is noah_muller_9847."]
    tau2, harness = start(tmp_path)

    act = tau2.act

    def customer_fails(action):  # tool calls are JSON, replies to the customer are not
        if action.startswith("{"):
            return act(action)
        raise RuntimeError("The simulated user failed")

    try:
        opening = harness.reset()
        monkeypatch.setattr(tau2, "act", customer_fails)
        turn = harness.step(HarnessAction(message=opening.metadata["customer"]))
    finally:
        harness.close()

    # Not a score of 0 for Claude Code: run_eval.py reports it as ERROR.
    assert turn.done and turn.metadata["error_type"] == "customer_failed"
    assert "simulated user failed" in turn.metadata["error"]


def test_claude_code_exiting_mid_turn_ends_the_conversation(tmp_path, customer):
    customer += ["crash"]
    tau2, harness = start(tmp_path)
    try:
        opening = harness.reset()  # starts τ²-bench's conversation thread
        threads = threading.active_count()
        turn = harness.step(HarnessAction(message=opening.metadata["customer"]))
    finally:
        harness.close()

    assert turn.done
    assert turn.metadata["error_type"] == "harness_crashed"
    assert "exited mid-turn" in turn.metadata["error"]
    # Closing the harness also ends the unfinished τ²-bench conversation.
    for _ in range(50):
        if threading.active_count() < threads:
            break
        time.sleep(0.1)
    assert threading.active_count() < threads


def test_claude_code_going_quiet_is_a_timeout(tmp_path, customer):
    customer += ["Hi, my user id is noah_muller_9847."]
    _, harness = start(tmp_path, session_timeout_s=1.0)
    try:
        harness.reset()
        turn = harness.step(HarnessAction(message="stall"))
    finally:
        harness.close()

    assert turn.done
    assert turn.metadata["error_type"] == "turn_timeout"
