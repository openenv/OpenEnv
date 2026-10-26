# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Snake Environment Client.

This module provides the client for connecting to a Snake Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State

    from .models import SnakeAction, SnakeObservation
except ImportError:
    from models import SnakeAction, SnakeObservation

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State


class SnakeEnv(EnvClient[SnakeAction, SnakeObservation, State]):
    """
    HTTP client for the Snake Environment.

    This client connects to a SnakeEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = SnakeEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.alive)  # True
        >>>
        >>> # Take an action (turn left)
        >>> result = client.step(SnakeAction(action=1))
        >>> print(result.observation.episode_score)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SnakeEnv.from_docker_image("snake-env:latest")
        >>> result = client.reset()
        >>> result = client.step(SnakeAction(action=0))  # noop
    """

    def _step_payload(self, action: SnakeAction) -> Dict:
        """
        Convert SnakeAction to JSON payload for step request.

        Args:
            action: SnakeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SnakeObservation]:
        """
        Parse server response into StepResult[SnakeObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with SnakeObservation
        """
        obs_data = payload.get("observation", {})
        observation = SnakeObservation(
            grid=obs_data.get("grid", []),
            observation=obs_data.get("observation", []),
            episode_score=obs_data.get("episode_score", 0.0),
            episode_steps=obs_data.get("episode_steps", 0),
            episode_fruits=obs_data.get("episode_fruits", 0),
            episode_kills=obs_data.get("episode_kills", 0),
            alive=obs_data.get("alive", True),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
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
