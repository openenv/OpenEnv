# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic MCP Environment Client with WebSocket Support.

This module provides a generic client for connecting to any MCP Environment
server over WebSocket (for persistent sessions) or HTTP. It's fully reusable
across different MCP integrations.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .data_models import MCPAction, MCPObservation


class MCPEnvClient(EnvClient[MCPAction, MCPObservation, State]):
    """
    Generic WebSocket client for MCP Environment.

    This client connects to any MCPEnvironment server via WebSocket and provides
    methods to interact with it: reset(), step(), and state access.
    Each client maintains a persistent WebSocket connection for better performance.

    Example:
        >>> # Connect to a running server using context manager (recommended)
        >>> with MCPEnvClient(base_url="http://localhost:8004") as client:
        ...     result = client.reset()
        ...     print(result.observation.metadata)
        ...
        ...     # List tools
        ...     result = client.step(MCPAction(action_type="ListToolsAction"))
        ...     print(result.observation.tools_list)
        ...
        ...     # Call a tool
        ...     result = client.step(MCPAction(
        ...         action_type="ToolCallAction",
        ...         tool_name="create_resource",
        ...         arguments={"name": "Test"}
        ...     ))
        ...     print(result.observation.tool_result)
        ...     print(result.reward)

    Example with manual cleanup:
        >>> try:
        ...     client = MCPEnvClient(base_url="http://localhost:8004")
        ...     result = client.reset()
        ...     result = client.step(MCPAction(action_type="ListToolsAction"))
        ... finally:
        ...     client.close()

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MCPEnvClient.from_docker_image("calendar-env:latest")
        >>> result = client.reset()
        >>> result = client.step(MCPAction(action_type="ListToolsAction"))
    """

    def _step_payload(self, action: MCPAction) -> Dict:
        """
        Convert MCPAction to JSON payload for step request.

        Args:
            action: MCPAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "action_type": action.action_type,
        }

        if action.tool_name is not None:
            payload["tool_name"] = action.tool_name

        if action.arguments is not None:
            payload["arguments"] = action.arguments

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[MCPObservation]:
        """
        Parse server response into StepResult[MCPObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with MCPObservation
        """
        obs_data = payload.get("observation", {})
        observation = MCPObservation(
            success=obs_data.get("success", True),
            error_message=obs_data.get("error_message"),
            tools_list=obs_data.get("tools_list"),
            tool_result=obs_data.get("tool_result"),
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
