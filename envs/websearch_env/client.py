# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
WebSearch Env Environment Client.

This module provides the client for connecting to a WebSearch Env Environment server
via WebSocket for persistent sessions.
"""

from typing import Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State

    from .models import WebSearchAction, WebSearchObservation
except ImportError:
    from models import WebSearchAction, WebSearchObservation

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State


class WebSearchEnv(EnvClient[WebSearchAction, WebSearchObservation, State]):
    """
    HTTP client for the WebSearch Env Environment.

    This client connects to a WebSearchEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = WebSearchEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.echoed_message)
        >>>
        >>> # Send a message
        >>> result = client.step(WebSearchAction(message="Hello!"))
        >>> print(result.observation.echoed_message)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = WebSearchEnv.from_docker_image("WebSearch_env-env:latest")
        >>> result = client.reset()
        >>> result = client.step(WebSearchAction(message="Test"))
    """

    def _step_payload(self, action: WebSearchAction) -> Dict:
        """
        Convert WebSearchAction to JSON payload for step request.

        Args:
            action: WebSearchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "query": action.query,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WebSearchObservation]:
        """
        Parse server response into StepResult[WebSearchObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with WebSearchObservation
        """
        obs_data = payload.get("observation", {})
        observation = WebSearchObservation(
            content=obs_data.get("content", ""),
            web_contents=obs_data.get("web_contents", []),
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
