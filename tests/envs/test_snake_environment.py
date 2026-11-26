# SPDX-License-Identifier: BSD-3-Clause

"""The Snake board and moves in the web playground."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("marlenv")

from snake_env.models import SnakeAction
from snake_env.server.snake_environment import SnakeEnvironment


def test_snake_web_playground():
    env = SnakeEnvironment(height=10, width=10)
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [
        ("0 · straight", {"action": 0}),
        ("1 · turn left", {"action": 1}),
        ("2 · turn right", {"action": 2}),
    ]
    obs = env.step(SnakeAction(action=0)).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Snake board"' in board
    assert board.count("<span") == 100
    assert "radius:4px" in env.render_web({"grid": [[13]]})  # a second snake's head
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None


def _play(env, seed=None, steps=40):
    """Reset (seeded if given), play a fixed action cycle, return what was seen."""
    obs = env.reset() if seed is None else env.reset(seed=seed)
    seen = [obs.grid]
    for i in range(steps):
        obs = env.step(SnakeAction(action=i % 3))
        seen.append((obs.grid, obs.reward))
        if obs.done:
            break
    return seen


class TestResetSeed:
    """reset(seed=...) makes the snake and fruit placement reproducible."""

    def test_same_seed_same_observations_across_instances(self):
        first = _play(SnakeEnvironment(height=10, width=10), seed=11)
        second = _play(SnakeEnvironment(height=10, width=10), seed=11)
        assert first == second

    def test_reseeding_restarts_the_sequence(self):
        env = SnakeEnvironment(height=10, width=10)
        first = _play(env, seed=11)
        _play(env)  # an unseeded game in between
        assert _play(env, seed=11) == first

    def test_different_seeds_give_different_games(self):
        env = SnakeEnvironment(height=10, width=10)
        assert len({str(_play(env, seed=seed)) for seed in range(10)}) > 1

    def test_seed_does_not_touch_the_global_numpy_rng(self):
        env = SnakeEnvironment(height=10, width=10)
        np.random.seed(5)
        expected = np.random.rand()
        np.random.seed(5)
        env.reset(seed=11)
        env.step(SnakeAction(action=0))
        assert np.random.rand() == expected

    def test_sessions_do_not_share_a_seed(self):
        first = SnakeEnvironment(height=10, width=10)
        second = SnakeEnvironment(height=10, width=10)
        expected = _play(first, seed=11)
        second.reset(seed=12)  # must not disturb the first session
        assert _play(first, seed=11) == expected

    def test_unseeded_reset_still_varies(self):
        env = SnakeEnvironment(height=10, width=10)
        assert len({str(env.reset().grid) for _ in range(20)}) > 1

    def test_episode_id_is_kept_or_generated(self):
        env = SnakeEnvironment(height=10, width=10)
        env.reset(episode_id="game-1")
        assert env.state.episode_id == "game-1"
        env.reset(seed=3, episode_id="game-2")
        assert env.state.episode_id == "game-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first
        assert env.state.step_count == 0

    def test_reset_async_forwards_seed_and_episode_id(self):
        env = SnakeEnvironment(height=10, width=10)
        expected = env.reset(seed=3).grid
        obs = asyncio.run(env.reset_async(seed=3, episode_id="async-1"))
        assert env.state.episode_id == "async-1"
        assert obs.grid == expected
