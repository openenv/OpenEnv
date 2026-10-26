# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Snake Environment Implementation.

A multi-agent snake game environment that wraps marlenv's Snake-v1.
This implementation provides a single-agent interface by wrapping the
multi-agent marlenv environment.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import gym
import marlenv.envs  # Register marlenv environments with gym
import numpy as np

# Support both in-repo and standalone imports
try:
    # Package import path used by installed envs and PYTHONPATH=src:envs.
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State

    from ..models import SnakeAction, SnakeObservation
except ImportError:
    # Direct execution from envs/snake_env.
    from models import SnakeAction, SnakeObservation
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State


class SingleAgentWrapper(gym.Wrapper):
    """
    Custom wrapper to convert multi-agent marlenv to single-agent.

    This wrapper properly handles the conversion without triggering
    gym 0.24.1's strict type checking on done flags.
    """

    def __init__(self, env):
        super().__init__(env)
        # Unwrap observation and action spaces for single agent
        if hasattr(env.observation_space, "__getitem__"):
            self.observation_space = env.observation_space[0]
        if hasattr(env.action_space, "__getitem__"):
            self.action_space = env.action_space[0]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Remove first dimension if it's a multi-agent array (num_agents, H, W, C)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            return obs[0]  # Return (H, W, C)
        # Return first agent's observation if it's a list
        if isinstance(obs, list):
            return obs[0]
        return obs

    def step(self, action):
        # Wrap action in list for multi-agent env
        obs, rewards, dones, info = self.env.step([action])

        # Unwrap returns for single agent
        # Handle observation: remove first dimension if shape is (1, H, W, C)
        if hasattr(obs, "shape") and len(obs.shape) == 4 and obs.shape[0] == 1:
            obs = obs[0]  # Convert (1, H, W, C) -> (H, W, C)
        elif isinstance(obs, list):
            obs = obs[0]

        reward = rewards[0] if isinstance(rewards, list) else rewards
        done = dones[0] if isinstance(dones, list) else dones

        # Ensure done is a boolean (not numpy bool)
        done = bool(done)

        return obs, reward, done, info


def _json_safe(value: Any) -> Any:
    """Convert marlenv metadata into JSON-serializable Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(value)


class SnakeEnvironment(Environment):
    """
    A snake game environment that wraps marlenv's Snake-v1.

    This environment provides a single-agent interface to the multi-agent
    snake game. The snake must navigate a grid, eat fruits, and avoid walls
    and its own body.

    Args:
        height: Height of the grid map (default: 20)
        width: Width of the grid map (default: 20)
        snake_length: Initial length of the snake (default: 3)
        vision_range: Vision range for partial observability (default: None for full grid)
        observer: 'snake' for relative actions or 'human' for global directions (default: 'snake')
        max_episode_steps: Maximum steps per episode (default: 1000)
        reward_dict: Custom reward function (default: fruit=1.0, others=0.0)

    Example:
        >>> env = SnakeEnvironment()
        >>> obs = env.reset()
        >>> print(obs.alive)  # True
        >>>
        >>> obs = env.step(SnakeAction(action=1))  # Turn left
        >>> print(obs.episode_score)
        >>> print(obs.reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        snake_length: int = 3,
        vision_range: int = None,
        observer: str = "snake",
        max_episode_steps: int = 1000,
        reward_dict: dict = None,
    ):
        """Initialize the snake environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Default reward function
        if reward_dict is None:
            reward_dict = {
                "fruit": 1.0,
                "kill": 0.0,
                "lose": -1.0,
                "win": 100.0,
                "time": 0.001,
            }

        # Create the marlenv snake environment for single agent
        # Note: We don't use gym.make directly to avoid gym 0.24.1 wrappers
        from marlenv.envs.snake_env import SnakeEnv as MarlenvSnake

        self.base_env = MarlenvSnake(
            height=height,
            width=width,
            num_snakes=1,  # Single agent
            snake_length=snake_length,
            vision_range=vision_range,
            frame_stack=1,
            observer=observer,
            reward_dict=reward_dict,
            max_episode_steps=max_episode_steps,
        )

        # Wrap with our custom SingleAgent wrapper
        self.env = SingleAgentWrapper(self.base_env)

        # Track episode statistics
        self._episode_score = 0.0
        self._episode_fruits = 0
        self._episode_kills = 0

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The moves as buttons, named for the observer's action space."""
        if self.base_env.observer == "human":
            names = ["noop", "left", "right", "down", "up"]
        else:
            names = ["straight", "turn left", "turn right"]
        return [(f"{a} · {name}", {"action": a}) for a, name in enumerate(names)]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the grid: walls, the fruit, and the snake with its head highlighted."""
        if "grid" not in observation:
            return None
        grid = observation["grid"]
        styles = {
            0: "background:var(--background-fill-primary);border-radius:2px",
            1: "background:var(--border-color-primary);border-radius:2px",
            2: "background:var(--body-text-color);border-radius:50%",
            3: "background:var(--color-accent);border-radius:4px",
        }
        body = "background:var(--color-accent);opacity:0.45;border-radius:2px"
        cells = [
            f'<span style="{styles.get(value % 10, body)}"></span>'
            for row in grid
            for value in row
        ]
        return (
            '<div role="img" aria-label="Snake board" style="display:inline-grid;'
            f"grid-template-columns:repeat({len(grid[0])},16px);grid-auto-rows:16px;gap:2px;padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    def reset(self) -> SnakeObservation:
        """
        Reset the environment.

        Returns:
            SnakeObservation with initial game state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_score = 0.0
        self._episode_fruits = 0
        self._episode_kills = 0

        # Reset the marlenv environment
        obs = self.env.reset()

        # Convert observation to list format
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else obs

        # Get the grid from the environment (access base env directly)
        grid = self.base_env.grid.tolist() if hasattr(self.base_env, "grid") else []

        return SnakeObservation(
            grid=grid,
            observation=obs_list,
            episode_score=self._episode_score,
            episode_steps=self._state.step_count,
            episode_fruits=self._episode_fruits,
            episode_kills=self._episode_kills,
            alive=True,
            done=False,
            reward=0.0,
        )

    def step(self, action: SnakeAction) -> SnakeObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: SnakeAction containing the action to take

        Returns:
            SnakeObservation with the result of the action
        """
        self._state.step_count += 1

        # Execute action in marlenv
        obs, reward, done, info = self.env.step(action.action)

        # Update episode statistics
        self._episode_score += reward

        # Convert observation to list format
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else obs

        # Get the grid from the environment (access base env directly)
        grid = self.base_env.grid.tolist() if hasattr(self.base_env, "grid") else []

        # Extract episode statistics from info if available
        episode_fruits = (
            info.get("episode_fruits", [self._episode_fruits])[0]
            if "episode_fruits" in info
            else self._episode_fruits
        )
        episode_kills = (
            info.get("episode_kills", [self._episode_kills])[0]
            if "episode_kills" in info
            else self._episode_kills
        )

        return SnakeObservation(
            grid=grid,
            observation=obs_list,
            episode_score=self._episode_score,
            episode_steps=self._state.step_count,
            episode_fruits=int(episode_fruits),
            episode_kills=int(episode_kills),
            alive=not done,
            done=done,
            reward=float(reward),
            metadata={"info": _json_safe(info)},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
