# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Calendar MCP Environment.

These models define the action and observation types used by the OpenEnv
integration for the calendar server.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class MCPAction(Action):
    """
    Generic wrapper action for MCP tool access.

    action_type values:
    - "ListToolsAction": list available tools
    - "ToolCallAction": execute a tool by name
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


class ListToolsAction(Action):
    """Internal action type for listing tools."""

    pass


class ToolCallAction(Action):
    """Internal action type for calling a tool."""

    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class MCPObservation(Observation):
    """
    Observation returned by the MCP environment.

    tools_list is populated for ListToolsAction.
    tool_result is populated for ToolCallAction.
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


class CalendarAction(MCPAction):
    """Action type for the Calendar environment."""

    pass


class CalendarObservation(MCPObservation):
    """Observation type for the Calendar environment."""

    pass


__all__ = [
    "MCPAction",
    "MCPObservation",
    "ListToolsAction",
    "ToolCallAction",
    "CalendarAction",
    "CalendarObservation",
]
