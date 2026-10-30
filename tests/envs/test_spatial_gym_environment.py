"""Tests for the Spatial-Gym environment (downloads the SPaRC dataset on first use)."""

import json
import socket
import threading
import time
import urllib.request

import pytest

pytest.importorskip("Spatial_Gym", reason="Spatial-Gym is not installed")

import uvicorn  # noqa: E402
from datasets import load_dataset  # noqa: E402
from spatial_gym_env import SpatialGymAction, SpatialGymEnv  # noqa: E402
from spatial_gym_env.models import parse_action, SpatialGymState  # noqa: E402
from spatial_gym_env.server.app import app  # noqa: E402
from spatial_gym_env.server.spatial_gym_environment import (  # noqa: E402
    SpatialGymEnvironment,
)

MOVES = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}


@pytest.fixture(scope="module")
def sparc_rows():
    return list(load_dataset("lkaesberg/SPaRC", "all", split="test"))


def one_per_level(rows):
    seen = {}
    for row in rows:
        seen.setdefault(row["difficulty_level"], row)
    return [seen[level] for level in sorted(seen)]


def gold_moves(row):
    path = [(p["x"], p["y"]) for p in row["solutions"][0]["path"]]
    return [MOVES[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]


def test_gold_solution_solves_each_level(sparc_rows):
    for row in one_per_level(sparc_rows):
        env = SpatialGymEnvironment()
        obs = env.reset(puzzle_id=row["id"])
        assert obs.puzzle_id == row["id"] and obs.difficulty == row["difficulty_level"]
        assert obs.system_prompt and obs.prompt.startswith("Step: 1\n")
        for move in gold_moves(row):
            assert not obs.done
            assert move in obs.legal_actions
            obs = env.step(SpatialGymAction(action=move))
        assert obs.done and obs.solved and obs.reached_end
        assert obs.reward == 1.0 and obs.prompt == ""
        assert env.state.solved and env.state.done
        assert env.state.path[-1] == obs.position


def test_wrong_line_scores_zero(sparc_rows):
    row = one_per_level(sparc_rows)[0]
    env = SpatialGymEnvironment()
    obs = env.reset(puzzle_id=row["id"], max_steps=3)
    illegal = next(a for a in range(4) if a not in obs.legal_actions)
    start = obs.position
    for _ in range(3):
        obs = env.step(SpatialGymAction(action=illegal))
    # An illegal move leaves the line where it is but still counts as a step.
    assert obs.position == start and obs.step == 3
    assert obs.done and not obs.solved and obs.reward == 0.0


def test_backtracking_undoes_a_move(sparc_rows):
    row = one_per_level(sparc_rows)[1]
    env = SpatialGymEnvironment()
    obs = env.reset(puzzle_id=row["id"], backtracking=True)
    first = gold_moves(row)[0]
    back = {0: 2, 1: 3, 2: 0, 3: 1}[first]
    obs = env.step(SpatialGymAction(action=first))
    assert len(env.state.path) == 2
    obs = env.step(SpatialGymAction(action=back))
    assert len(env.state.path) == 1 and obs.position == env.state.path[0]


def test_reset_options():
    env = SpatialGymEnvironment()
    a = env.reset(seed=7, difficulty=[4, 5])
    b = SpatialGymEnvironment().reset(seed=7, difficulty=[4, 5])
    assert a.puzzle_id == b.puzzle_id and a.difficulty in (4, 5)
    train = env.reset(split="train", difficulty=2, seed=1)
    assert env.state.split == "train" and train.difficulty == 2
    for bad in (
        {"split": "dev"},
        {"difficulty": 6},
        {"max_steps": 0},
        {"puzle_id": "x"},
        {"puzzle_id": "nope"},
    ):
        with pytest.raises(ValueError):
            env.reset(**bad)


def test_step_needs_an_open_episode(sparc_rows):
    env = SpatialGymEnvironment()
    with pytest.raises(RuntimeError):
        env.step(SpatialGymAction(action=0))
    row = one_per_level(sparc_rows)[0]
    env.reset(puzzle_id=row["id"])
    for move in gold_moves(row):
        env.step(SpatialGymAction(action=move))
    with pytest.raises(RuntimeError):
        env.step(SpatialGymAction(action=0))


def test_sessions_are_independent(sparc_rows):
    row = one_per_level(sparc_rows)[0]
    one, two = SpatialGymEnvironment(), SpatialGymEnvironment()
    one.reset(puzzle_id=row["id"])
    two.reset(puzzle_id=row["id"])
    one.step(SpatialGymAction(action=gold_moves(row)[0]))
    assert len(one.state.path) == 2 and len(two.state.path) == 1


def test_web_playground_draws_the_board(sparc_rows):
    row = one_per_level(sparc_rows)[4]
    env = SpatialGymEnvironment()
    assert env.render_web({}) is None
    obs = env.reset(puzzle_id=row["id"])
    buttons = env.web_actions(obs.model_dump())
    assert [action["action"] for _, action in buttons] == sorted(obs.legal_actions)
    obs = env.step(SpatialGymAction(action=gold_moves(row)[0]))
    assert obs.path == env.state.path and len(obs.path) == 2
    html = env.render_web(obs.model_dump())
    assert html.count("<svg") == 1 and 'stroke="#ffffff"' in html
    assert f"level {row['difficulty_level']}" in html


def test_action_parsing():
    assert parse_action("I go up.\nFinal: 1") == 1
    assert SpatialGymAction.from_reply("<think>maybe 0</think> Final: left").action == 2
    with pytest.raises(ValueError):
        SpatialGymAction.from_reply("no move here")
    with pytest.raises(ValueError):
        SpatialGymAction(action=4)


# End to end: run the FastAPI server and play through the WebSocket client.


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def test_play_gold_solution_over_websocket(base_url):
    row = next(
        r
        for r in load_dataset("lkaesberg/SPaRC", "all", split="test")
        if r["difficulty_level"] == 3
    )
    path = [(p["x"], p["y"]) for p in row["solutions"][0]["path"]]
    with SpatialGymEnv(base_url=base_url).sync() as env:
        result = env.reset(puzzle_id=row["id"])
        assert result.observation.puzzle_id == row["id"] and not result.done
        assert result.observation.system_prompt
        for a, b in zip(path, path[1:]):
            result = env.step(
                SpatialGymAction(action=MOVES[(b[0] - a[0], b[1] - a[1])])
            )
        assert result.done and result.reward == 1.0 and result.observation.solved
        state = env.state()
        assert state.solved and state.step_count == len(path) - 1


def test_http_and_websocket_serve_spatial_state(base_url, sparc_rows):
    row = one_per_level(sparc_rows)[0]
    with urllib.request.urlopen(f"{base_url}/schema") as response:
        properties = json.load(response)["state"]["properties"]
    assert set(properties) == set(SpatialGymState.model_fields)

    # HTTP requests each get a fresh environment; only WebSocket keeps an episode.
    with urllib.request.urlopen(f"{base_url}/state") as response:
        http_state = json.load(response)
    assert set(http_state) == set(SpatialGymState.model_fields)
    assert (
        http_state == SpatialGymState(episode_id=http_state["episode_id"]).model_dump()
    )

    with SpatialGymEnv(base_url=base_url).sync() as env:
        result = env.reset(seed=7, episode_id="state-contract", puzzle_id=row["id"])
        state = env.state().model_dump()
        assert set(state) == set(http_state)
        assert state["episode_id"] == "state-contract"
        assert state["puzzle_id"] == row["id"]
        assert state["difficulty"] == row["difficulty_level"]
        assert state["path"] == result.observation.path
        assert state["seed"] == 7


def test_two_clients_get_separate_episodes(base_url):
    with (
        SpatialGymEnv(base_url=base_url).sync() as one,
        SpatialGymEnv(base_url=base_url).sync() as two,
    ):
        first = one.reset(seed=3)
        two.reset(seed=3)
        one.step(SpatialGymAction(action=first.observation.legal_actions[0]))
        assert one.state().step_count == 1 and two.state().step_count == 0
