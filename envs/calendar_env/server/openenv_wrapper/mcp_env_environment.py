# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic MCP Environment Implementation with MCP Tool Support.

This environment integrates any MCP server following RFC 003's
traditional tool calling approach. Agents can:
1. Discover available tools using ListToolsAction
2. Execute tools using ToolCallAction

The environment wraps any MCP server and provides reward signals
based on successful tool execution.

This file is fully generic and doesn't require modification when
copying to different MCP projects. All MCP-specific configuration
is in config.py.
"""

import asyncio
import logging
from typing import Union
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .config import get_tool_handlers, get_user_manager_class, MCP_NAME
from .data_models import ListToolsAction, MCPAction, MCPObservation, ToolCallAction

logger = logging.getLogger(__name__)

# Load tool handlers dynamically from config
_tool_handlers = get_tool_handlers()
MCP_TOOLS_LIST = _tool_handlers["MCP_TOOLS_LIST"]
TOOL_HANDLERS = _tool_handlers["TOOL_HANDLERS"]


class MCPEnvironment(Environment):
    """
    Generic MCP Environment with Tool Integration.

    This environment provides access to any MCP operations through
    MCP tools. It supports two action types:

    1. ListToolsAction - Discover available MCP tools
    2. ToolCallAction - Execute a specific MCP tool

    Example:
        >>> env = MCPEnvironment(database_id="test")
        >>>
        >>> # Discover tools
        >>> obs = env.reset()
        >>> list_action = ListToolsAction()
        >>> obs = env.step(list_action)
        >>> print(len(obs.tools_list))  # Number of available tools
        >>>
        >>> # Call a tool
        >>> call_action = ToolCallAction(
        ...     tool_name="create_resource",
        ...     arguments={"name": "Example", "type": "test"}
        ... )
        >>> obs = env.step(call_action)
        >>> print(obs.success)  # True if tool executed successfully
        >>> print(obs.tool_result)  # Result from the tool
    """

    def __init__(self, database_id: str = "default", auth_token: str = None):
        """
        Initialize the MCP environment.

        Args:
            database_id: Default database identifier for multi-tenancy support
            auth_token: Optional default authentication token (for future use)
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._default_database_id = database_id
        self._default_auth_token = auth_token
        self._reset_count = 0
        self._successful_tool_calls = 0
        self._failed_tool_calls = 0

        # Current request context (set from HTTP headers)
        self._current_database_id = database_id
        self._current_access_token = auth_token

        logger.info(
            f"{MCP_NAME} environment initialized with database_id: {database_id}"
        )

    def set_request_context(self, database_id: str = None, access_token: str = None):
        """
        Set the current request context from HTTP headers.

        This method should be called before step() to provide database_id and access_token
        from the incoming HTTP request headers.

        Args:
            database_id: Database ID from x-database-id header
            access_token: Access token from x-{mcp}-access-token header
        """
        self._current_database_id = database_id or self._default_database_id
        self._current_access_token = access_token or self._default_auth_token

        logger.debug(f"Request context set: database_id={self._current_database_id}")

    def reset(self) -> MCPObservation:
        """
        Reset the environment to initial state.

        Returns:
            MCPObservation with environment information and available tool count
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._successful_tool_calls = 0
        self._failed_tool_calls = 0

        logger.info(f"Environment reset (episode {self._reset_count})")

        return MCPObservation(
            success=True,
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "available_tools": len(MCP_TOOLS_LIST),
                "database_id": self._current_database_id,
                "message": f"{MCP_NAME} environment ready. Use ListToolsAction to discover tools.",
            },
        )

    def step(self, action: MCPAction) -> MCPObservation:  # type: ignore[override]
        """
        Execute an action in the environment.

        Supports two action types via action_type discriminator:
        1. "ListToolsAction" - Returns list of available MCP tools
        2. "ToolCallAction" - Executes a specific MCP tool

        Args:
            action: MCPAction with action_type discriminator

        Returns:
            MCPObservation with results and reward
        """
        self._state.step_count += 1

        try:
            # Dispatch based on action_type
            if action.action_type == "ListToolsAction":
                internal_action = ListToolsAction()
                return self._handle_list_tools(internal_action)
            elif action.action_type == "ToolCallAction":
                if not action.tool_name:
                    return MCPObservation(
                        success=False,
                        error_message="tool_name is required for ToolCallAction",
                        done=False,
                        reward=-1.0,
                        metadata={"step": self._state.step_count},
                    )
                internal_action = ToolCallAction(
                    tool_name=action.tool_name, arguments=action.arguments or {}
                )
                return self._handle_tool_call(internal_action)
            else:
                logger.error(f"Unknown action_type: {action.action_type}")
                return MCPObservation(
                    success=False,
                    error_message=f"Unknown action_type: {action.action_type}. Must be 'ListToolsAction' or 'ToolCallAction'",
                    done=False,
                    reward=-1.0,
                    metadata={"step": self._state.step_count},
                )

        except Exception as e:
            logger.error(f"Error executing action: {e}", exc_info=True)
            return MCPObservation(
                success=False,
                error_message=f"Internal error: {str(e)}",
                done=False,
                reward=-1.0,
                metadata={
                    "step": self._state.step_count,
                    "error_type": type(e).__name__,
                },
            )

    def _handle_list_tools(self, action: ListToolsAction) -> MCPObservation:
        """
        Handle ListToolsAction by returning available MCP tools.

        Args:
            action: ListToolsAction instance

        Returns:
            MCPObservation with tools_list populated
        """
        logger.info("Listing available MCP tools")

        # MCP_TOOLS_LIST already contains dictionaries, no need to convert
        tools_list = MCP_TOOLS_LIST

        return MCPObservation(
            success=True,
            tools_list=tools_list,
            done=False,
            reward=0.1,  # Small positive reward for successful discovery
            metadata={
                "step": self._state.step_count,
                "action_type": "list_tools",
                "tools_count": len(tools_list),
            },
        )

    def _handle_tool_call(self, action: ToolCallAction) -> MCPObservation:
        """
        Handle ToolCallAction by executing the specified MCP tool.

        Args:
            action: ToolCallAction with tool_name and arguments

        Returns:
            MCPObservation with tool_result populated
        """
        tool_name = action.tool_name
        arguments = action.arguments or {}

        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

        # Check if tool exists
        if tool_name not in TOOL_HANDLERS:
            logger.warning(f"Tool not found: {tool_name}")
            self._failed_tool_calls += 1
            return MCPObservation(
                success=False,
                error_message=f"Tool not found: {tool_name}",
                done=False,
                reward=-0.5,
                metadata={
                    "step": self._state.step_count,
                    "action_type": "tool_call",
                    "tool_name": tool_name,
                    "available_tools": len(TOOL_HANDLERS),
                },
            )

        # Execute the tool
        try:
            # Get the tool handler
            handler = TOOL_HANDLERS[tool_name]
            # Get database_id and access_token from current request context
            database_id = self._current_database_id
            access_token = self._current_access_token

            # Handle access token validation (same logic as openenv_routes)
            UserManager = get_user_manager_class()
            user_manager = UserManager(database_id)

            # If no access token provided, get first user's token
            if not access_token or (
                isinstance(access_token, str) and access_token.strip() == ""
            ):
                fallback_token = user_manager.get_first_user_token(db_id=database_id)
                if not fallback_token:
                    return MCPObservation(
                        success=False,
                        error_message="Access token is required and no users are available for fallback",
                        done=False,
                        reward=-1.0,
                        metadata={
                            "step": self._state.step_count,
                            "tool_name": tool_name,
                        },
                    )
                access_token = fallback_token

            # Clean access token (remove invisible characters)
            if access_token:
                invisible_chars = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
                for char in invisible_chars:
                    access_token = access_token.replace(char, "")
                access_token = access_token.strip()

            # Get user_id from access token (same logic as openenv_routes)
            try:
                user = user_manager.get_user_by_static_token(access_token)
            except Exception as e:
                logger.error(f"Error validating access token: {e}")
                return MCPObservation(
                    success=False,
                    error_message=f"Failed to validate access token: {str(e)}",
                    done=False,
                    reward=-1.0,
                    metadata={"step": self._state.step_count, "tool_name": tool_name},
                )

            if not user:
                return MCPObservation(
                    success=False,
                    error_message="Invalid or expired access token. User not found.",
                    done=False,
                    reward=-1.0,
                    metadata={"step": self._state.step_count, "tool_name": tool_name},
                )

            try:
                if isinstance(user, dict) and "id" in user:
                    user_id = user["id"]
                else:
                    return MCPObservation(
                        success=False,
                        error_message=f"Invalid user object structure: expected dict with 'id' field, got {type(user)}",
                        done=False,
                        reward=-1.0,
                        metadata={
                            "step": self._state.step_count,
                            "tool_name": tool_name,
                        },
                    )
            except (KeyError, TypeError) as e:
                return MCPObservation(
                    success=False,
                    error_message=f"Failed to extract user_id from user object: {e}",
                    done=False,
                    reward=-1.0,
                    metadata={"step": self._state.step_count, "tool_name": tool_name},
                )

            # Execute the tool asynchronously
            # Check if handler accepts access_token parameter (some MCPs don't need it)
            import inspect

            handler_signature = inspect.signature(handler)
            handler_params = handler_signature.parameters

            # Build kwargs based on what the handler accepts
            handler_kwargs = {
                "tool_name": tool_name,
                "arguments": arguments,
                "database_id": database_id,
                "user_id": user_id,
            }

            # Only add access_token if the handler accepts it
            if "access_token" in handler_params:
                handler_kwargs["access_token"] = access_token

            result = asyncio.run(handler(**handler_kwargs))

            self._successful_tool_calls += 1

            # Compute reward based on success
            reward = self._compute_reward(tool_name, result)

            logger.info(f"Tool {tool_name} executed successfully")

            return MCPObservation(
                success=True,
                tool_result=result,
                done=False,
                reward=reward,
                metadata={
                    "step": self._state.step_count,
                    "action_type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}", exc_info=True)
            self._failed_tool_calls += 1
            return MCPObservation(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                tool_result={"error": str(e)},
                done=False,
                reward=-1.0,
                metadata={
                    "step": self._state.step_count,
                    "action_type": "tool_call",
                    "tool_name": tool_name,
                    "error_type": type(e).__name__,
                },
            )

    def _compute_reward(self, tool_name: str, result: dict) -> float:
        """
        Compute reward based on tool execution result.

        Reward structure:
        - Successful read operations: +0.5
        - Successful write operations: +1.0
        - Operations with errors: Based on HTTP status code

        Args:
            tool_name: Name of the executed tool
            result: Result dictionary from tool execution

        Returns:
            Float reward value
        """
        # Check for errors in result
        if isinstance(result, dict):
            if "error" in result:
                return -0.5

            # Check status code if present
            status = result.get("status_code") or result.get("statusCode")
            if status:
                if status >= 400:
                    return -0.5
                elif status >= 200 and status < 300:
                    return 1.0

        # Default positive reward for successful execution
        return 0.5

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    @property
    def stats(self) -> dict:
        """
        Get environment statistics.

        Returns:
            Dictionary with execution statistics
        """
        total_calls = self._successful_tool_calls + self._failed_tool_calls
        success_rate = (
            self._successful_tool_calls / total_calls if total_calls > 0 else 0.0
        )

        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "reset_count": self._reset_count,
            "successful_tool_calls": self._successful_tool_calls,
            "failed_tool_calls": self._failed_tool_calls,
            "success_rate": success_rate,
            "database_id": self._current_database_id,
        }
