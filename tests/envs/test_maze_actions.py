"""Action handling tests for the Maze environment."""

import pytest
from envs.maze_env.models import MazeAction
from envs.maze_env.server.maze_env_environment import MazeEnvironment


@pytest.mark.parametrize("action", [4, 7, -1])
def test_action_outside_the_four_moves_is_an_invalid_move(action):
    env = MazeEnvironment()
    env.reset()

    obs = env.step(MazeAction(action=action))

    assert obs.reward == -0.75
    assert not obs.done
    assert obs.current_position == [0, 0]
    assert env.state.step_count == 1
