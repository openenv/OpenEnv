# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for OpenSpiel Environment.

This module defines the Action, Observation, and State types for OpenSpiel games.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class OpenSpielAction(Action):
    """
    Action for OpenSpiel environments.

    Attributes:
        action_id: The integer action ID to take (from legal_actions).
        game_name: Name of the OpenSpiel game (e.g., "catch", "tic_tac_toe").
        game_params: Optional game-specific parameters (e.g., {"rows": 8, "columns": 6}).
    """

    action_id: int
    game_name: str = "catch"
    game_params: Dict[str, Any] = Field(default_factory=dict)


class OpenSpielObservation(Observation):
    """
    Observation from OpenSpiel environment.

    This represents what the agent sees after taking an action.
    For single-player games, this is straightforward.
    For multi-player games, this is from the perspective of the agent player.

    Attributes:
        info_state: Information state tensor (list of floats) for the agent.
                   This contains all information available to the agent.
        legal_actions: List of legal action IDs the agent can take.
        game_phase: String describing the current phase (e.g., "playing", "terminal").
        current_player_id: ID of the current player (-1 for simultaneous, player ID otherwise).
        opponent_last_action: Last action taken by opponent (if available, None otherwise).
    """

    info_state: List[float]
    legal_actions: List[int]
    game_phase: str = "playing"
    current_player_id: int = 0
    opponent_last_action: Optional[int] = None


class OpenSpielState(State):
    """
    State for OpenSpiel environment.

    Attributes:
        game_name: Name of the OpenSpiel game.
        agent_player: Which player ID the agent controls (0 by default).
        opponent_policy: Name of the opponent policy ("random", "fixed", etc.).
        game_params: Game-specific parameters.
        num_players: Total number of players in the game.
    """

    game_name: str = "catch"
    agent_player: int = 0
    opponent_policy: str = "random"
    game_params: Dict[str, Any] = Field(default_factory=dict)
    num_players: int = 1
