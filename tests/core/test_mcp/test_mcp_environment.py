# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MCPEnvironment base class."""

import pytest
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    RESERVED_TOOL_NAMES,
)


class TestReservedToolNames:
    """Tests for reserved tool name validation."""

    def test_reserved_names_prevent_reset(self):
        """Test that 'reset' is a reserved name."""
        assert "reset" in RESERVED_TOOL_NAMES

    def test_reserved_names_prevent_step(self):
        """Test that 'step' is a reserved name."""
        assert "step" in RESERVED_TOOL_NAMES

    def test_reserved_names_prevent_state(self):
        """Test that 'state' is a reserved name."""
        assert "state" in RESERVED_TOOL_NAMES

    def test_reserved_names_prevent_close(self):
        """Test that 'close' is a reserved name."""
        assert "close" in RESERVED_TOOL_NAMES

    def test_reserved_names_immutable(self):
        """Test that reserved names cannot be modified."""
        with pytest.raises(AttributeError):
            RESERVED_TOOL_NAMES.add("new_name")


class TestMCPEnvironmentImports:
    """Tests that MCPEnvironment can be imported."""

    def test_import_mcp_environment(self):
        """Test that MCPEnvironment can be imported."""
        from openenv.core.env_server.mcp_environment import MCPEnvironment

        assert MCPEnvironment is not None

    def test_import_from_package(self):
        """Test that MCPEnvironment is exported from the package."""
        from openenv.core.env_server import MCPEnvironment

        assert MCPEnvironment is not None


class TestMCPActions:
    """Tests for MCP action types."""

    def test_list_tools_action_default_type(self):
        """Test ListToolsAction has correct default type."""
        action = ListToolsAction()
        assert action.type == "list_tools"

    def test_call_tool_action_stores_values(self):
        """Test CallToolAction stores tool_name and arguments."""
        action = CallToolAction(tool_name="my_tool", arguments={"key": "value"})
        assert action.tool_name == "my_tool"
        assert action.arguments == {"key": "value"}

    def test_call_tool_action_default_arguments(self):
        """Test CallToolAction has empty default arguments."""
        action = CallToolAction(tool_name="my_tool")
        assert action.arguments == {}


class TestMCPObservations:
    """Tests for MCP observation types."""

    def test_list_tools_observation_empty_tools(self):
        """Test ListToolsObservation with empty tools list."""
        obs = ListToolsObservation(tools=[])
        assert obs.tools == []
        assert obs.done is False

    def test_call_tool_observation_success(self):
        """Test CallToolObservation for successful call."""
        obs = CallToolObservation(tool_name="test_tool", result="success result")
        assert obs.tool_name == "test_tool"
        assert obs.result == "success result"
        assert obs.error is None
