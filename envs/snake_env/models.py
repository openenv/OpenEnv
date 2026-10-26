# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Snake Environment.

The Snake environment is a multi-agent reinforcement learning environment
based on marlenv's Snake-v1. Multiple snakes battle on a fixed size grid map.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SnakeAction(Action):
    """
    Action for the Snake environment.

    For single snake (observer='snake'):
        action: int in [0, 1, 2]
            0 = noop (continue in same direction)
            1 = turn left (90 degrees)
            2 = turn right (90 degrees)

    For single snake (observer='human'):
        action: int in [0, 1, 2, 3, 4]
            0 = noop
            1 = left
            2 = right
            3 = down
            4 = up
    """

    action: int

    def __post_init__(self):
        """Ensure action is converted to int (handles string inputs from web interface)."""
        self.action = int(self.action)


class SnakeObservation(Observation):
    """
    Observation from the Snake environment.

    Attributes:
        grid: The current game grid as a nested list (height x width)
        observation: The encoded observation for the snake (can be full grid or vision range)
        episode_score: Total score accumulated in this episode
        episode_steps: Number of steps taken in this episode
        episode_fruits: Number of fruits eaten in this episode
        episode_kills: Number of kills in this episode
        alive: Whether the snake is still alive
    """

    grid: List[List[int]]
    observation: List[List[List[float]]]  # H x W x C observation
    episode_score: float = 0.0
    episode_steps: int = 0
    episode_fruits: int = 0
    episode_kills: int = 0
    alive: bool = True
