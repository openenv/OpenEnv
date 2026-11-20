"""Reset contract tests for the Maze environment."""

import asyncio

from envs.maze_env.server.app import app
from envs.maze_env.server.maze_env_environment import MazeEnvironment
from fastapi.testclient import TestClient


def test_reset_preserves_requested_episode_id() -> None:
    env = MazeEnvironment()

    env.reset(episode_id="requested-episode")

    assert env.state.episode_id == "requested-episode"


def test_reset_async_preserves_requested_episode_id() -> None:
    env = MazeEnvironment()

    asyncio.run(env.reset_async(episode_id="requested-async-episode"))

    assert env.state.episode_id == "requested-async-episode"


def test_reset_preserves_empty_episode_id() -> None:
    env = MazeEnvironment()

    env.reset(episode_id="")

    assert env.state.episode_id == ""


def test_reset_accepts_start_cell_as_list() -> None:
    env = MazeEnvironment()

    obs = env.reset(start_cell=[1, 0])

    assert obs.current_position == [1, 0]


def test_http_reset_with_start_cell() -> None:
    client = TestClient(app)

    response = client.post("/reset", json={"start_cell": [1, 0]})

    assert response.status_code == 200
    assert response.json()["observation"]["current_position"] == [1, 0]
