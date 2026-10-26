# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic Data Models for MCP Environment with Tool Support.

Following RFC 003 - Traditional Tool Calling Approach:
- MCPAction: Wrapper action that dispatches to ListToolsAction or ToolCallAction
- ListToolsAction: Discover available MCP tools
- ToolCallAction: Execute a specific MCP tool

These models are fully generic and work with any MCP integration.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MCPAction(Action):
    """
    Generic wrapper action for MCP environment that supports multiple action types.

    This action uses an action_type discriminator to determine which action to execute:
    - "ListToolsAction": Discover available MCP tools
    - "ToolCallAction": Execute a specific MCP tool

    Args:
        action_type: Type of action ("ListToolsAction" or "ToolCallAction")
        tool_name: Name of tool to call (required for ToolCallAction)
        arguments: Arguments for tool (optional, for ToolCallAction)

    Examples:
        >>> # List tools
        >>> action = MCPAction(action_type="ListToolsAction")
        >>>
        >>> # Call a tool
        >>> action = MCPAction(
        ...     action_type="ToolCallAction",
        ...     tool_name="create_resource",
        ...     arguments={"name": "New Resource", "type": "example"}
        ... )
    """

    action_type: Literal["ListToolsAction", "ToolCallAction"] = Field(
        ..., description="Type of action to perform"
    )
    tool_name: Optional[str] = Field(
        None, description="Name of tool to call (required for ToolCallAction)"
    )
    arguments: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Arguments for the tool"
    )


# Internal action types (not exposed to HTTP API)
class ListToolsAction(Action):
    """
    Internal: Request list of available tools from MCP server.

    This action corresponds to the MCP tools/list API.
    Use MCPAction with action_type="ListToolsAction" instead.
    """

    pass  # No parameters needed


class ToolCallAction(Action):
    """
    Internal: Call a specific MCP tool with arguments.

    This action corresponds to the MCP tools/call API.
    Use MCPAction with action_type="ToolCallAction" instead.
    """

    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class MCPObservation(Observation):
    """
    Generic observation from the MCP Environment.

    Depending on the action type, different fields will be populated:
    - For ListToolsAction: tools_list contains available tool schemas
    - For ToolCallAction: tool_result contains the execution result

    Args:
        success: Whether the action succeeded
        error_message: Error message if action failed
        tools_list: List of available tools (for ListToolsAction)
        tool_result: Result from tool execution (for ToolCallAction)
        metadata: Additional metadata about the execution
        done: Whether the episode is complete
        reward: Reward for the action
    """

    success: bool = Field(True, description="Whether the action succeeded")
    error_message: Optional[str] = Field(
        None, description="Error message if action failed"
    )
    tools_list: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of available tools (for ListToolsAction)"
    )
    tool_result: Optional[Dict[str, Any]] = Field(
        None, description="Result from tool execution (for ToolCallAction)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the execution"
    )
    done: bool = Field(False, description="Whether the episode is complete")
    reward: Optional[float] = Field(None, description="Reward for the action")
