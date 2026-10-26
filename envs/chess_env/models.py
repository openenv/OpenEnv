# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Chess Environment.

This module defines the Action, Observation, and State types for chess games
via the OpenEnv interface. Uses the moonfish chess engine for move search
and position evaluation.
"""

from __future__ import annotations

from typing import List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class ChessAction(Action):
    """
    Action for Chess environment.

    Attributes:
        move: UCI format move string (e.g., "e2e4", "e7e8q" for promotion).
    """

    move: str


class ChessObservation(Observation):
    """
    Observation for Chess environment.

    Attributes:
        fen: Board position in FEN notation.
        legal_moves: List of legal moves in UCI format.
        is_check: Whether the current player is in check.
        done: Whether the game is over.
        reward: Reward for the last action.
        result: Game result string if game is over (e.g., "1-0", "0-1", "1/2-1/2").
    """

    fen: str = ""
    legal_moves: List[str] = Field(default_factory=list)
    is_check: bool = False
    result: Optional[str] = None


class ChessState(State):
    """
    State for Chess environment.

    Attributes:
        episode_id: Unique ID for the current game.
        fen: Current board position in FEN notation.
        current_player: "white" or "black".
        move_history: List of moves played in UCI format.
        step_count: Number of half-moves played.
    """

    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    current_player: str = "white"
    move_history: List[str] = Field(default_factory=list)
    step_count: int = 0
