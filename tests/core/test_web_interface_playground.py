# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the default web playground at /web and its environment hooks."""

import asyncio
import importlib
import json
import sys
from typing import Literal

import gradio as gr
import pytest
from openenv.core.env_server.gradio_ui import (
    _description,
    _Form,
    _params,
    _readme_body,
    _result_html,
    build_gradio_app,
)
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import (
    Action,
    EnvironmentMetadata,
    Observation,
    State,
)
from openenv.core.env_server.web_interface import (
    _extract_action_fields,
    WebInterfaceManager,
)


class MoveAction(Action):
    action_id: int


class BoardObservation(Observation):
    legal_actions: list[int] = []


class TinyEnv(Environment):
    def reset(self, seed=None, episode_id=None, **kwargs):
        return BoardObservation(legal_actions=[0, 1])

    def step(self, action, timeout_s=None, **kwargs):
        return BoardObservation(legal_actions=[0, 1], reward=1.0, done=True)

    @property
    def state(self):
        return State()


def test_environment_hooks_default_to_nothing():
    env = TinyEnv()
    assert env.render_web({"legal_actions": [0, 1]}) is None
    assert env.web_actions({"legal_actions": [0, 1]}) == []


def test_result_shows_tool_output_and_stats():
    data = {"observation": {"result": {"data": "Hello"}}, "reward": 0.0, "done": False}
    html = _result_html(data, step_count=1)
    assert "Hello" in html
    assert "reward" in html and "done" in html and "step" in html


def test_result_lists_observation_fields():
    data = {
        "observation": {"legal_actions": [0, 1], "metadata": {}},
        "reward": 1.0,
        "done": True,
    }
    html = _result_html(data, step_count=2)
    assert "legal_actions" in html
    assert "metadata" not in html


class ScreenObservation(Observation):
    screen: list = []
    board: list = []
    legal_actions: list[int] = []


class ScreenEnv(TinyEnv):
    def reset(self, seed=None, episode_id=None, **kwargs):
        return ScreenObservation(
            screen=[[[0, 0, 255]] * 1920] * 1080,
            board=[[0] * 7] * 6,
            legal_actions=list(range(7)),
        )

    def step(self, action, timeout_s=None, **kwargs):
        return ScreenObservation(reward=1.0, done=True)

    def render_web(self, observation):
        self.drawn = observation
        return "<svg></svg>"


def test_big_arrays_are_shown_as_their_shape():
    env = ScreenEnv()
    manager = WebInterfaceManager(env, MoveAction, ScreenObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    fns = {f.fn.__name__: f.fn for f in blocks.fns.values()}

    outputs = asyncio.run(fns["reset_env"]([]))

    assert "list · 1080×1920×3" in outputs[3]
    assert "[0, 0, 0, 0, 0, 0, 0]" in outputs[3]  # a 6x7 board keeps its cells
    assert json.loads(outputs[4])["observation"]["screen"] == "list · 1080×1920×3"
    assert len(outputs[4]) < 1000
    assert len(env.drawn["screen"]) == 1080  # the drawing gets the full observation
    assert env.drawn["reward"] is None and env.drawn["done"] is False

    asyncio.run(fns["step_fn"](outputs[6], 0))

    assert env.drawn["reward"] == 1.0 and env.drawn["done"] is True


def test_playground_builds_for_a_plain_env():
    manager = WebInterfaceManager(TinyEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    assert isinstance(blocks, gr.Blocks)


def test_playground_lists_mcp_tools():
    echo = pytest.importorskip("echo_env.server.echo_environment")
    manager = WebInterfaceManager(
        echo.EchoEnvironment(), CallToolAction, CallToolObservation
    )
    blocks = build_gradio_app(
        manager, _extract_action_fields(CallToolAction), None, False
    )
    radios = [b for b in blocks.blocks.values() if isinstance(b, gr.Radio)]
    tool_names = {value for radio in radios for _, value in radio.choices}
    assert {"echo_message", "echo_with_length"} <= tool_names


def test_catch_offers_legal_moves_and_a_board():
    pytest.importorskip("open_spiel")
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name="catch")
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [
        ("0 · left", {"action_id": 0}),
        ("1 · stay", {"action_id": 1}),
        ("2 · right", {"action_id": 2}),
    ]
    board = env.render_web(obs)
    assert 'aria-label="Catch board"' in board
    assert "#" not in board  # theme colours only, so it reads in dark mode


def test_openspiel_2048_web_playground():
    pytest.importorskip("open_spiel")
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name="2048")
    obs = env.reset().model_dump()
    names = {0: "up", 1: "right", 2: "down", 3: "left"}
    assert env.web_actions(obs) == [
        (f"{a} · {names[a]}", {"action_id": a}) for a in obs["legal_actions"]
    ]
    board = env.render_web(obs)
    assert 'aria-label="2048 board"' in board
    assert "#" not in board
    for value in obs["info_state"]:
        if value:
            assert f">{int(value)}<" in board


@pytest.mark.parametrize(
    "game, label, button",
    [
        ("tic_tac_toe", "Tic-Tac-Toe board", ("4 · centre", {"action_id": 4})),
        ("connect_four", "Connect Four board", ("col 3", {"action_id": 3})),
        ("blackjack", "Blackjack board", ("0 · hit", {"action_id": 0})),
        ("kuhn_poker", "Kuhn Poker board", ("1 · bet", {"action_id": 1})),
        ("cliff_walking", "Cliff Walking board", ("1 · up", {"action_id": 1})),
    ],
)
def test_openspiel_web_playground(game, label, button):
    pytest.importorskip("open_spiel")
    from openspiel_env.models import OpenSpielAction
    from openspiel_env.server.openspiel_environment import OpenSpielEnvironment

    env = OpenSpielEnvironment(game_name=game)
    obs = env.reset().model_dump()
    assert button in env.web_actions(obs)
    obs = env.step(OpenSpielAction(**button[1])).model_dump()
    drawing = env.render_web(obs)
    assert f'aria-label="{label}"' in drawing
    assert "#" not in drawing


def _sum_env():
    from fastmcp import FastMCP
    from openenv.core.env_server.mcp_environment import MCPEnvironment

    mcp = FastMCP("sum_env")

    @mcp.tool
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    class SumEnv(MCPEnvironment):
        def reset(self, seed=None, episode_id=None, **kwargs):
            return Observation()

        def _step_impl(self, action, timeout_s=None, **kwargs):
            return Observation()

        @property
        def state(self):
            return State()

    return SumEnv(mcp)


def test_tool_without_description_and_json_argument():
    manager = WebInterfaceManager(_sum_env(), CallToolAction, CallToolObservation)
    blocks = build_gradio_app(
        manager, _extract_action_fields(CallToolAction), None, False
    )
    radios = [b for b in blocks.blocks.values() if isinstance(b, gr.Radio)]
    assert ("total", "total") in [c for r in radios for c in r.choices]
    labels = [b.label for b in blocks.blocks.values() if isinstance(b, gr.Textbox)]
    assert "numbers (array, JSON) · required" in labels


def test_failed_tool_call_shows_the_error():
    data = {
        "observation": {"result": None, "error": {"message": "unknown tool"}},
        "reward": None,
        "done": False,
    }
    html = _result_html(data, step_count=1)
    assert "Error: unknown tool" in html
    assert "null" not in html


class RunAction(Action):
    command: str
    timeout: float | None = 30.0
    retries: int | None = None
    mode: str = "fast"
    tags: list[str] | None = None


def test_params_unwrap_optional_and_keep_defaults():
    params = {
        name: (schema, required)
        for name, schema, required in _params(RunAction.model_json_schema())
    }
    assert params["command"] == ({"title": "Command", "type": "string"}, True)
    assert (
        params["timeout"][0]["type"] == "number"
        and params["timeout"][0]["default"] == 30.0
    )
    assert params["retries"][0]["type"] == "integer" and not params["retries"][1]
    assert params["tags"][0]["type"] == "array"


def test_form_sends_defaults_skips_empty_optionals_and_requires_fields():
    with gr.Blocks():
        form = _Form(_params(RunAction.model_json_schema()))
    by_name = dict(zip([n for n, _ in form.names], form.inputs))
    assert isinstance(by_name["retries"], gr.Textbox)  # no default: empty, not 0
    assert by_name["timeout"].value == 30.0
    raw = {
        "command": "ls",
        "timeout": 30.0,
        "retries": "",
        "mode": "fast",
        "tags": '["a"]',
    }
    values = form.values([raw[n] for n, _ in form.names])
    assert values == {"command": "ls", "timeout": 30.0, "mode": "fast", "tags": ["a"]}
    raw["command"] = " "
    with pytest.raises(ValueError, match="Fill in command"):
        form.values([raw[n] for n, _ in form.names])


class TurnAction(Action):
    direction: Literal["LEFT", "RIGHT"]
    speed: Literal["slow", "fast"] | None = None


def test_required_choice_starts_on_its_first_option():
    with gr.Blocks():
        form = _Form(_params(TurnAction.model_json_schema()))
    assert [i.value for i in form.inputs] == ["LEFT", None]
    assert form.values([i.value for i in form.inputs]) == {"direction": "LEFT"}


def test_description_falls_back_to_the_readme():
    readme = "---\ntitle: X\n---\n# Catch\n\n> [!NOTE]\n> hi\n\nMove the [paddle](x) to catch the ball.\n"
    meta = EnvironmentMetadata(
        name="catch_env", description="catch_env environment", readme_content=readme
    )
    assert _description(meta) == "Move the paddle to catch the ball."
    meta = EnvironmentMetadata(
        name="x", description="A custom description.", readme_content=readme
    )
    assert _description(meta) == "A custom description."


def test_reset_errors_and_rewards_render_readably():
    html = _result_html(
        {
            "observation": {"metadata": {"error": "E2B_API_KEY is not set"}},
            "reward": None,
            "done": True,
        },
        0,
    )
    assert "Error: E2B_API_KEY is not set" in html
    assert "–" in html
    html = _result_html(
        {"observation": {"value": 1}, "reward": -0.45999999999999996}, 1
    )
    assert "-0.46" in html and "0.4599999" not in html


class LoginAction(Action):
    user: str
    api_key: str


def test_secret_arguments_are_masked_in_the_episode():
    manager = WebInterfaceManager(TinyEnv(), LoginAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(LoginAction), None, False)
    step_fn = next(f.fn for f in blocks.fns.values() if f.fn.__name__ == "step_fn")

    outputs = asyncio.run(step_fn([], "ana", "sk-123"))

    assert outputs[6][-1][1] == 'step(user="ana", api_key=***)'
    assert "sk-123" not in outputs[5]


def test_long_arguments_are_cut_in_the_episode():
    manager = WebInterfaceManager(TinyEnv(), LoginAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(LoginAction), None, False)
    step_fn = next(f.fn for f in blocks.fns.values() if f.fn.__name__ == "step_fn")
    svg = "<svg>" + "<circle r='1'/>" * 100 + "</svg>"

    outputs = asyncio.run(step_fn([], svg, "sk-123"))

    call = outputs[6][-1][1]
    assert call.startswith('step(user="<svg><circle') and "…, api_key=***)" in call
    assert len(call) < 100


def test_enter_in_a_text_field_runs_the_step():
    manager = WebInterfaceManager(TinyEnv(), RunAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(RunAction), None, False)
    step = next(f for f in blocks.fns.values() if f.fn and f.fn.__name__ == "step_fn")
    textboxes = [id for id, b in blocks.blocks.items() if isinstance(b, gr.Textbox)]

    assert textboxes and {(t, "submit") for t in textboxes} <= set(step.targets)


class NoStateEnv(TinyEnv):
    @property
    def state(self):
        raise RuntimeError("no episode")


def test_state_shows_nothing_when_unavailable():
    manager = WebInterfaceManager(NoStateEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    show_state = next(
        f.fn for f in blocks.fns.values() if f.fn.__name__ == "show_state"
    )

    assert show_state() == ""


class FailingResetEnv(TinyEnv):
    def reset(self, seed=None, episode_id=None, **kwargs):
        raise RuntimeError("no sandbox")


def test_failed_reset_keeps_the_episode():
    manager = WebInterfaceManager(FailingResetEnv(), MoveAction, BoardObservation)
    blocks = build_gradio_app(manager, _extract_action_fields(MoveAction), None, False)
    reset_env = next(f.fn for f in blocks.fns.values() if f.fn.__name__ == "reset_env")
    entries = [
        ["R", "reset()", "new episode"],
        ["1", "step(action_id=0)", "reward 1.0"],
    ]

    outputs = asyncio.run(reset_env(entries))

    assert "reset() failed: no sandbox" in outputs[3]
    assert outputs[6] == entries


def test_new_env_from_the_template_gets_a_working_playground(
    tmp_path, monkeypatch, request
):
    """An env scaffolded by `openenv init` needs no extra code for /web to work."""
    from fastapi.testclient import TestClient
    from openenv.cli.__main__ import app as cli
    from typer.testing import CliRunner

    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(cli, ["init", "fresh_env"], input="\n").exit_code == 0
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")

    def forget_fresh_env():
        for module in [m for m in sys.modules if m.startswith("fresh_env")]:
            del sys.modules[module]

    request.addfinalizer(forget_fresh_env)
    server = importlib.import_module("fresh_env.server.app")
    models = importlib.import_module("fresh_env.models")
    environment = importlib.import_module("fresh_env.server.fresh_env_environment")

    assert TestClient(server.app).get("/web/").status_code == 200

    manager = WebInterfaceManager(
        environment.FreshEnvironment, models.FreshAction, models.FreshObservation
    )
    blocks = build_gradio_app(
        manager, _extract_action_fields(models.FreshAction), None, False
    )
    fns = {f.fn.__name__: f.fn for f in blocks.fns.values()}
    entries = asyncio.run(fns["reset_env"]([]))[6]
    assert json.loads(fns["show_state"]())["step_count"] == 0
    outputs = asyncio.run(fns["step_fn"](entries, "hello"))

    assert "hello" in outputs[3]
    assert outputs[6][-1][1] == 'step(message="hello")'
    assert json.loads(fns["show_state"]())["step_count"] == 1


def test_readme_drops_the_space_front_matter():
    readme = "---\ntitle: Catch\nsdk: docker\n---\n\n# Catch\n\nMove the paddle.\n"
    assert _readme_body(readme) == "\n# Catch\n\nMove the paddle.\n"
