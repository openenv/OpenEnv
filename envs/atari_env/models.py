# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Atari Environment.

This module defines the Action, Observation, and State types for Atari games
via the Arcade Learning Environment (ALE).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State


class AtariAction(Action):
    """
    Action for Atari environments.

    Attributes:
        action_id: The integer action ID to take (from legal_actions).
        game_name: Name of the Atari game (e.g., "pong", "breakout", "space_invaders").
        obs_type: Observation type ("rgb", "grayscale", or "ram").
        full_action_space: Whether to use full (18 actions) or minimal action space.
    """

    action_id: int
    game_name: str = "pong"
    obs_type: Literal["rgb", "grayscale", "ram"] = "rgb"
    full_action_space: bool = False


class AtariObservation(Observation):
    """
    Observation from Atari environment.

    This represents what the agent sees after taking an action.

    Attributes:
        screen: Screen observation as a flattened list of pixels.
                Shape depends on obs_type:
                - rgb: [210, 160, 3] flattened
                - grayscale: [210, 160] flattened
                - ram: [128] (RAM contents)
        screen_shape: Original shape of the screen before flattening.
        legal_actions: List of legal action IDs the agent can take.
        lives: Number of lives remaining.
        episode_frame_number: Frame number within current episode.
        frame_number: Total frame number since environment creation.
    """

    screen: List[int]
    screen_shape: List[int]
    legal_actions: List[int]
    lives: int = 0
    episode_frame_number: int = 0
    frame_number: int = 0


class AtariState(State):
    """
    State for Atari environment.

    Attributes:
        game_name: Name of the Atari game.
        obs_type: Observation type ("rgb", "grayscale", or "ram").
        full_action_space: Whether using full or minimal action space.
        mode: Game mode (if applicable).
        difficulty: Game difficulty (if applicable).
        repeat_action_probability: Probability of repeating previous action (sticky actions).
        frameskip: Number of frames to skip per action.
    """

    game_name: str = "pong"
    obs_type: Literal["rgb", "grayscale", "ram"] = "rgb"
    full_action_space: bool = False
    mode: Optional[int] = None
    difficulty: Optional[int] = None
    repeat_action_probability: float = 0.0
    frameskip: int = 4
