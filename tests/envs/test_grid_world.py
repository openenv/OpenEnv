# SPDX-License-Identifier: BSD-3-Clause

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
