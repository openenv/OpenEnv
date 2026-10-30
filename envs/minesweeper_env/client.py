# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minesweeper Environment Client.

This module provides the client for connecting to a Minesweeper Environment server
via WebSocket for persistent sessions.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import MinesweeperAction, MinesweeperObservation


class MinesweeperEnv(EnvClient[MinesweeperAction, MinesweeperObservation, State]):
    """
    Client for the Minesweeper Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MinesweeperEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.board)
        ...     print(result.observation.game_status)
        ...
        ...     # Reveal a cell
        ...     result = client.step(MinesweeperAction(row=0, col=0, action_type="reveal"))
        ...     print(result.observation.board)
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MinesweeperEnv.from_docker_image("minesweeper-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MinesweeperAction(row=2, col=3, action_type="reveal"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MinesweeperAction) -> Dict:
        """
        Convert MinesweeperAction to JSON payload for step request.

        Args:
            action: MinesweeperAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "row": action.row,
            "col": action.col,
            "action_type": action.action_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MinesweeperObservation]:
        """
        Parse server response into StepResult[MinesweeperObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with MinesweeperObservation
        """
        obs_data = payload.get("observation", {})
        observation = MinesweeperObservation(
            board=obs_data.get("board", []),
            num_mines=obs_data.get("num_mines", 0),
            flags_placed=obs_data.get("flags_placed", 0),
            cells_revealed=obs_data.get("cells_revealed", 0),
            game_status=obs_data.get("game_status", "ongoing"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
