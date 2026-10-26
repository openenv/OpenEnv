"""Reset contract tests for the Maze environment."""

import asyncio

from envs.maze_env.server.maze_env_environment import MazeEnvironment


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
