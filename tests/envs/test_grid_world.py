# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import pytest

# Import your client and models DIRECTLY
from envs.grid_world_env.client import GridWorldEnv
from envs.grid_world_env.models import GridWorldAction, MoveAction
from envs.grid_world_env.server.grid_world_environment import GridWorldEnvironment


def test_grid_world_flow():
    """
    Test the full flow of the Grid World environment using the WebSocket client.
    """
    # 1. Initialize the client
    try:
        # We use a dummy URL for unit testing logic
        client = GridWorldEnv("ws://localhost:8000/ws")
    except Exception as e:
        pytest.fail(f"Failed to initialize client: {e}")

    # 2. Test Action Creation
    # FIX: Use GridWorldAction directly, not client.action_model
    action_up = GridWorldAction(action=MoveAction.UP)
    assert action_up.action == "UP"

    action_right = GridWorldAction(action=MoveAction.RIGHT)
    assert action_right.action == "RIGHT"

    # 3. Test Payload Serialization (The new abstract method you added)
    # This verifies that the strict method you wrote in client.py works correctly
    payload = client._step_payload(action_up)
    assert isinstance(payload, dict)
    assert payload["action"] == "UP"

    print("Grid World Client tests passed!")


def test_grid_world_web_playground():
    env = GridWorldEnvironment()
    obs = env.reset().model_dump()
    assert env.web_actions(obs) == [
        ("↑ up", {"action": "UP"}),
        ("↓ down", {"action": "DOWN"}),
        ("← left", {"action": "LEFT"}),
        ("→ right", {"action": "RIGHT"}),
    ]
    obs = env.step(GridWorldAction(action=MoveAction.DOWN)).model_dump()
    grid = env.render_web(obs)
    assert 'aria-label="Grid World board"' in grid
    assert grid.count("border-radius:50%") == 1  # one agent
    assert "★" in grid
    assert "#" not in grid  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None


class TestResetSeed:
    """reset(seed=..., episode_id=...) must be accepted.

    Grid World has no randomness, so a seed changes nothing: the agent always
    starts at [0, 0]. The server only passes arguments that `reset` declares, so
    an environment that omits them silently drops the caller's seed and episode id.
    """

    def test_seeded_reset_gives_same_start(self):
        env = GridWorldEnvironment()
        unseeded = env.reset()
        for seed in (0, 1, 1234):
            obs = env.reset(seed=seed)
            assert (obs.x, obs.y) == (unseeded.x, unseeded.y) == (0, 0)
            assert env.state.step_count == 0

    def test_episode_id_is_kept_or_generated(self):
        env = GridWorldEnvironment()
        env.reset(episode_id="game-1")
        assert env.state.episode_id == "game-1"
        env.reset(seed=3, episode_id="game-2")
        assert env.state.episode_id == "game-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first

    def test_seeded_reset_returns_agent_to_start(self):
        env = GridWorldEnvironment()
        env.step(GridWorldAction(action=MoveAction.DOWN))
        obs = env.reset(seed=5)
        assert (obs.x, obs.y) == (0, 0)
        assert env.state.step_count == 0

    def test_reset_async_forwards_seed_and_episode_id(self):
        env = GridWorldEnvironment()
        asyncio.run(env.reset_async(seed=3, episode_id="async-1"))
        assert env.state.episode_id == "async-1"


class TestStepAfterDone:
    """Steps after the goal is reached return the final observation again."""

    def test_goal_pays_only_once(self):
        env = GridWorldEnvironment()
        env.reset()
        for move in [MoveAction.DOWN] * 4 + [MoveAction.RIGHT] * 4:
            final = env.step(GridWorldAction(action=move))
        assert final.done and final.reward == 1.0

        for move in (MoveAction.UP, MoveAction.DOWN):
            assert env.step(GridWorldAction(action=move)) == final
        assert env.state.step_count == 8

        assert not env.reset().done
        assert env.step(GridWorldAction(action=MoveAction.DOWN)).reward == -0.1
