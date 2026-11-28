"""Step tests for the Maze environment."""

import numpy as np
from envs.maze_env.models import MazeAction
from envs.maze_env.server.maze_env_environment import MazeEnvironment

UP, DOWN, RIGHT = 0, 1, 3


class TestStepAfterDone:
    """Steps after the episode ends return the final observation again."""

    @staticmethod
    def _won_env():
        # Start at (0, 0), exit at (1, 1), reached by moving right then down.
        env = MazeEnvironment(maze_array=np.array([[0, 0], [1, 0]]))
        env.reset()
        env.step(MazeAction(action=RIGHT))
        final = env.step(MazeAction(action=DOWN))
        assert final.done and final.reward == 10.0
        return env, final

    def test_exit_pays_only_once(self):
        env, final = self._won_env()
        for action in (UP, DOWN, UP, DOWN):
            obs = env.step(MazeAction(action=action))
            assert obs == final
        assert env.state.step_count == 2
        assert env.state.current_position == [1, 1]

    def test_returned_observation_is_a_copy(self):
        env, _ = self._won_env()
        env.step(MazeAction(action=UP)).metadata["status"] = "playing"
        assert env.step(MazeAction(action=UP)).metadata["status"] == "win"

    def test_lost_episode_stays_lost(self):
        # Moving into a wall costs -0.75; the default 8x8 maze loses below -32.
        env = MazeEnvironment()
        env.reset()
        obs = env.step(MazeAction(action=UP))
        while not obs.done:
            obs = env.step(MazeAction(action=UP))
        assert obs.metadata["status"] == "lose"
        assert env.step(MazeAction(action=RIGHT)) == obs

    def test_reset_starts_a_new_episode(self):
        env, _ = self._won_env()
        obs = env.reset()
        assert not obs.done and obs.current_position == [0, 0]
        assert env.step(MazeAction(action=RIGHT)).reward == -0.05
