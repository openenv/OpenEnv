# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MCP type definitions and deserialization routing."""

import pytest
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    RESERVED_TOOL_NAMES,
    Tool,
    ToolError,
    ToolErrorType,
    WSMCPMessage,
    WSMCPResponse,
)
from openenv.core.env_server.serialization import (
    deserialize_action,
    deserialize_action_with_preprocessing,
)
from openenv.core.env_server.types import Action
from pydantic import ValidationError


class TestTool:
    """Tests for the Tool model."""

    def test_tool_creation(self):
        """Test creating a valid Tool."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "properties" in tool.input_schema

    def test_tool_requires_all_fields(self):
        """Test that Tool requires name, description, and input_schema."""
        with pytest.raises(ValidationError):
            Tool(name="test")  # Missing description and input_schema

    def test_tool_serialization(self):
        """Test Tool can be serialized to dict."""
        tool = Tool(
            name="echo",
            description="Echo message",
            input_schema={"type": "object"},
        )
        data = tool.model_dump()
        assert data["name"] == "echo"
        assert data["description"] == "Echo message"


class TestToolError:
    """Tests for the ToolError model."""

    def test_tool_error_creation(self):
        """Test creating a ToolError."""
        error = ToolError(
            error_type=ToolErrorType.EXECUTION_ERROR,
            message="Something went wrong",
        )
        assert error.error_type == ToolErrorType.EXECUTION_ERROR
        assert error.message == "Something went wrong"

    def test_all_error_types(self):
        """Test all error types can be used."""
        for error_type in ToolErrorType:
            error = ToolError(error_type=error_type, message="test")
            assert error.error_type == error_type


class TestListToolsAction:
    """Tests for ListToolsAction."""

    def test_list_tools_action_creation(self):
        """Test creating a ListToolsAction."""
        action = ListToolsAction()
        assert action.type == "list_tools"

    def test_list_tools_action_metadata(self):
        """Test ListToolsAction supports metadata."""
        action = ListToolsAction(metadata={"request_id": "123"})
        assert action.metadata["request_id"] == "123"


class TestCallToolAction:
    """Tests for CallToolAction."""

    def test_call_tool_action_creation(self):
        """Test creating a CallToolAction."""
        action = CallToolAction(tool_name="echo", arguments={"message": "hello"})
        assert action.type == "call_tool"
        assert action.tool_name == "echo"
        assert action.arguments["message"] == "hello"

    def test_call_tool_action_default_arguments(self):
        """Test CallToolAction has empty dict as default arguments."""
        action = CallToolAction(tool_name="list")
        assert action.arguments == {}

    def test_call_tool_requires_tool_name(self):
        """Test CallToolAction requires tool_name."""
        with pytest.raises(ValidationError):
            CallToolAction()


class TestListToolsObservation:
    """Tests for ListToolsObservation."""

    def test_list_tools_observation_creation(self):
        """Test creating a ListToolsObservation."""
        tools = [
            Tool(name="echo", description="Echo message", input_schema={}),
            Tool(name="greet", description="Greet user", input_schema={}),
        ]
        obs = ListToolsObservation(tools=tools)
        assert len(obs.tools) == 2
        assert obs.tools[0].name == "echo"
        assert obs.done is False  # Default from Observation

    def test_list_tools_observation_empty(self):
        """Test ListToolsObservation with no tools."""
        obs = ListToolsObservation(tools=[])
        assert obs.tools == []


class TestCallToolObservation:
    """Tests for CallToolObservation."""

    def test_call_tool_observation_success(self):
        """Test CallToolObservation for successful call."""
        obs = CallToolObservation(
            tool_name="echo",
            result={"message": "hello", "length": 5},
        )
        assert obs.tool_name == "echo"
        assert obs.result["message"] == "hello"
        assert obs.error is None

    def test_call_tool_observation_with_error(self):
        """Test CallToolObservation with error."""
        obs = CallToolObservation(
            tool_name="broken_tool",
            result=None,
            error=ToolError(
                error_type=ToolErrorType.EXECUTION_ERROR,
                message="Tool crashed",
            ),
        )
        assert obs.tool_name == "broken_tool"
        assert obs.error is not None
        assert obs.error.error_type == ToolErrorType.EXECUTION_ERROR


class TestWSMCPMessage:
    """Tests for WebSocket MCP messages."""

    def test_ws_mcp_message_creation(self):
        """Test creating a WSMCPMessage."""
        msg = WSMCPMessage(data={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert msg.type == "mcp"
        assert msg.data["method"] == "tools/list"

    def test_ws_mcp_response_creation(self):
        """Test creating a WSMCPResponse."""
        response = WSMCPResponse(
            data={"jsonrpc": "2.0", "result": {"tools": []}, "id": 1}
        )
        assert response.type == "mcp"
        assert response.data["result"]["tools"] == []


class TestReservedToolNames:
    """Tests for reserved tool names."""

    def test_reserved_names_exist(self):
        """Test that reserved names are defined."""
        assert "reset" in RESERVED_TOOL_NAMES
        assert "step" in RESERVED_TOOL_NAMES
        assert "state" in RESERVED_TOOL_NAMES
        assert "close" in RESERVED_TOOL_NAMES

    def test_reserved_names_is_frozenset(self):
        """Test that reserved names cannot be modified."""
        assert isinstance(RESERVED_TOOL_NAMES, frozenset)


# Deserialization routing regression tests


class _DummyEnvAction(Action):
    """A non-MCP action class used to simulate env-specific action types."""

    value: str = "hello"


class TestDeserializeActionMCPRouting:
    """MCP action types are routed correctly when action_cls is the base Action."""

    def test_list_tools_with_base_action_cls(self):
        data = {"type": "list_tools"}
        action = deserialize_action(data, Action)
        assert isinstance(action, ListToolsAction)
        assert action.type == "list_tools"

    def test_list_tools_with_call_tool_action_cls(self):
        data = {"type": "list_tools"}
        action = deserialize_action(data, CallToolAction)
        assert isinstance(action, ListToolsAction)

    def test_call_tool_with_base_action_cls(self):
        data = {"type": "call_tool", "tool_name": "echo", "arguments": {"msg": "hi"}}
        action = deserialize_action(data, Action)
        assert isinstance(action, CallToolAction)
        assert action.tool_name == "echo"
        assert action.arguments == {"msg": "hi"}

    def test_non_mcp_action_uses_action_cls(self):
        data = {"value": "world"}
        action = deserialize_action(data, _DummyEnvAction)
        assert isinstance(action, _DummyEnvAction)
        assert action.value == "world"

    def test_call_tool_with_string_arguments_is_rejected_without_preprocessing(self):
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"}',
        }
        with pytest.raises(ValidationError):
            deserialize_action(data, Action)

    def test_invalid_non_mcp_action_raises(self):
        data = {"nonexistent_field": 123}
        with pytest.raises(ValidationError):
            deserialize_action(data, _DummyEnvAction)


class TestDeserializeActionNonMCPGuard:
    """MCP routing does NOT hijack payloads when action_cls is a specific non-MCP class."""

    def test_non_mcp_cls_with_call_tool_type_falls_through(self):
        data = {"type": "call_tool", "tool_name": "echo", "arguments": {}}
        with pytest.raises(ValidationError):
            deserialize_action(data, _DummyEnvAction)

    def test_non_mcp_cls_with_list_tools_type_falls_through(self):
        data = {"type": "list_tools"}
        with pytest.raises(ValidationError):
            deserialize_action(data, _DummyEnvAction)


class TestDeserializeWithPreprocessingMCPRouting:
    """Same MCP routing works in the preprocessing variant, with string argument decoding."""

    def test_list_tools_bypasses_preprocessing(self):
        data = {"type": "list_tools"}
        action = deserialize_action_with_preprocessing(data, Action)
        assert isinstance(action, ListToolsAction)

    @pytest.mark.parametrize("action_cls", [Action, CallToolAction])
    def test_call_tool_with_dict_arguments(self, action_cls):
        data = {"type": "call_tool", "tool_name": "solve", "arguments": {"x": 1}}
        action = deserialize_action_with_preprocessing(data, action_cls)
        assert isinstance(action, CallToolAction)
        assert action.tool_name == "solve"
        assert action.arguments == {"x": 1}

    @pytest.mark.parametrize("action_cls", [Action, CallToolAction])
    def test_call_tool_with_valid_json_string_arguments(self, action_cls):
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"}',
        }
        action = deserialize_action_with_preprocessing(data, action_cls)
        assert isinstance(action, CallToolAction)
        assert action.tool_name == "echo"
        assert action.arguments == {"message": "hello"}

    def test_call_tool_string_preprocessing_does_not_mutate_input(self):
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"}',
        }
        deserialize_action_with_preprocessing(data, Action)
        assert data["arguments"] == '{"message": "hello"}'

    def test_call_tool_with_malformed_json_string_arguments_is_rejected(self):
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '{"message": "hello"',
        }
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, Action)

    def test_call_tool_with_json_array_string_arguments_is_rejected(self):
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '["hello"]',
        }
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, Action)

    def test_call_tool_with_scalar_json_string_arguments_is_rejected(self):
        data = {
            "type": "call_tool",
            "tool_name": "echo",
            "arguments": '"hello"',
        }
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, Action)

    def test_non_mcp_still_preprocessed(self):
        data = {"value": "test"}
        action = deserialize_action_with_preprocessing(data, _DummyEnvAction)
        assert isinstance(action, _DummyEnvAction)
        assert action.value == "test"


class TestDeserializeWithPreprocessingNonMCPGuard:
    """Preprocessing variant also guards against MCP hijacking."""

    def test_non_mcp_cls_with_call_tool_type_falls_through(self):
        data = {"type": "call_tool", "tool_name": "echo", "arguments": {}}
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, _DummyEnvAction)

    def test_non_mcp_cls_with_list_tools_type_falls_through(self):
        data = {"type": "list_tools"}
        with pytest.raises(ValidationError):
            deserialize_action_with_preprocessing(data, _DummyEnvAction)
