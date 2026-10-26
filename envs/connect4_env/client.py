# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Connect4 Environment Client.

This module provides the client for connecting to a Connect4 Environment server
via WebSocket for persistent sessions.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import Connect4Action, Connect4Observation, Connect4State

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class Connect4Env(EnvClient[Connect4Action, Connect4Observation, Connect4State]):
    """
    Client for Connect4 Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with Connect4Env(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.board)
        ...
        ...     result = client.step(Connect4Action(column=3))
        ...     print(result.reward, result.done)
    """

    def _step_payload(self, action: Connect4Action) -> Dict[str, Any]:
        """
        Convert Connect4Action to JSON payload for step request.

        Args:
            action: Connect4Action instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "column": action.column,  # column index to drop piece
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Connect4Observation]:
        """
        Parse server response into StepResult[Connect4Observation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with Connect4Observation.
        """
        obs_data = payload.get("observation", {})

        observation = Connect4Observation(
            board=obs_data.get("board", [[0] * 7 for _ in range(6)]),
            legal_actions=obs_data.get("legal_actions", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Connect4State:
        """
        Parse server response into Connect4State object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            Connect4State object with environment state information.
        """
        return Connect4State(
            episode_id=payload.get("episode_id", ""),
            board=payload.get("board", [[0] * 7 for _ in range(6)]),
            next_player=payload.get("next_player", 1),
            step_count=payload.get("step_count", 0),
        )
