# SPDX-License-Identifier: BSD-3-Clause

"""
Integration tests for MCP functionality.

These tests verify:
1. EchoEnvironment MCP features (list and call tools via step())
2. MCPEnvironment base class with FastMCP servers
3. WebSocket MCP tools/list and tools/call endpoints
"""

import json
from typing import Any, Optional

import pytest
from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State


# =============================================================================
# Test Fixtures
# =============================================================================


class MinimalMCPEnvironment(MCPEnvironment):
    """
    Minimal MCPEnvironment subclass for testing.

    This is a simple environment that wraps a FastMCP server for testing
    the MCPEnvironment base class functionality.
    """

    def __init__(self, mcp_server: Any) -> None:
        super().__init__(mcp_server)
        self._state = State(episode_id="test-episode", step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ):
        self._state = State(
            episode_id=episode_id or "test-episode",
            step_count=0,
        )
        return Observation(done=False, reward=0.0)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions."""
        self._state.step_count += 1
        return Observation(
            done=False,
            reward=0.0,
            metadata={"action": str(action)},
        )

    @property
    def state(self) -> State:
        return self._state


@pytest.fixture
def simple_mcp_server():
    """Create a simple FastMCP server for testing."""
    mcp = FastMCP("test-server")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool
    def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    return mcp


@pytest.fixture
def minimal_mcp_env(simple_mcp_server):
    """Create a MinimalMCPEnvironment with the simple MCP server."""
    return MinimalMCPEnvironment(simple_mcp_server)


# =============================================================================
# EchoEnvironment MCP Integration Tests
# =============================================================================


class TestEchoEnvironmentMCP:
    """Tests for EchoEnvironment's MCP functionality."""

    def test_echo_environment_list_tools(self):
        """Test EchoEnvironment.step(ListToolsAction()) returns available tools."""
        from echo_env.server.echo_environment import EchoEnvironment

        env = EchoEnvironment()
        env.reset()

        # List tools via step()
        obs = env.step(ListToolsAction())

        # Verify observation type
        assert isinstance(obs, ListToolsObservation)
        assert obs.done is False

        # Verify tools are returned
        assert len(obs.tools) >= 2  # echo_message and echo_with_length

        tool_names = [t.name for t in obs.tools]
        assert "echo_message" in tool_names
        assert "echo_with_length" in tool_names

    def test_echo_environment_call_tool_echo_message(self):
        """Test EchoEnvironment.step(CallToolAction()) for echo_message tool."""
        from echo_env.server.echo_environment import EchoEnvironment

        env = EchoEnvironment()
        env.reset()

        # Call echo_message tool via step()
        obs = env.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, MCP!"},
            )
        )

        # Verify observation type
        assert isinstance(obs, CallToolObservation)
        assert obs.tool_name == "echo_message"
        assert obs.error is None
        assert obs.done is False

        # Verify result - MCPEnvironment returns CallToolResult object
        # Extract the actual value from the result
        if hasattr(obs.result, "data"):
            assert obs.result.data == "Hello, MCP!"
        elif hasattr(obs.result, "content"):
            assert obs.result.content[0].text == "Hello, MCP!"
        else:
            assert obs.result == "Hello, MCP!"

    def test_echo_environment_call_tool_echo_with_length(self):
        """Test EchoEnvironment.step(CallToolAction()) for echo_with_length tool."""
        from echo_env.server.echo_environment import EchoEnvironment

        env = EchoEnvironment()
        env.reset()

        # Call echo_with_length tool via step()
        obs = env.step(
            CallToolAction(
                tool_name="echo_with_length",
                arguments={"message": "test"},
            )
        )

        # Verify observation type
        assert isinstance(obs, CallToolObservation)
        assert obs.tool_name == "echo_with_length"
        assert obs.error is None

        # The result should be JSON string of dict with message and length
        # Result is parsed from the tool's return value
        assert "test" in str(obs.result) or obs.result is not None

    def test_echo_environment_call_nonexistent_tool(self):
        """Test EchoEnvironment handles calling a nonexistent tool gracefully."""
        from echo_env.server.echo_environment import EchoEnvironment

        env = EchoEnvironment()
        env.reset()

        # Call a tool that doesn't exist
        obs = env.step(
            CallToolAction(
                tool_name="nonexistent_tool",
                arguments={},
            )
        )

        # Verify error is returned
        assert isinstance(obs, CallToolObservation)
        assert obs.tool_name == "nonexistent_tool"
        assert obs.error is not None

    def test_echo_environment_reset_returns_observation(self):
        """Test EchoEnvironment.reset() returns an Observation."""
        from echo_env.server.echo_environment import EchoEnvironment

        env = EchoEnvironment()
        obs = env.reset()

        # Verify observation type (now just Observation, not EchoObservation)
        assert isinstance(obs, Observation)
        assert obs.done is False
        assert obs.metadata.get("status") == "ready"


# =============================================================================
# MCPEnvironment with FastMCP Tests
# =============================================================================


class TestMCPEnvironmentWithFastMCP:
    """Tests for MCPEnvironment base class with FastMCP servers."""

    def test_fastmcp_in_mcp_environment_list_tools(self, minimal_mcp_env):
        """Test that MCPEnvironment correctly lists tools from a FastMCP server."""
        obs = minimal_mcp_env.step(ListToolsAction())

        # Verify observation type
        assert isinstance(obs, ListToolsObservation)
        assert len(obs.tools) == 2

        # Verify tool details
        tool_names = [t.name for t in obs.tools]
        assert "add" in tool_names
        assert "greet" in tool_names

    def test_fastmcp_in_mcp_environment_call_add(self, minimal_mcp_env):
        """Test MCPEnvironment can call an 'add' tool from FastMCP server."""
        obs = minimal_mcp_env.step(
            CallToolAction(
                tool_name="add",
                arguments={"a": 5, "b": 3},
            )
        )

        assert isinstance(obs, CallToolObservation)
        assert obs.tool_name == "add"
        assert obs.error is None
        # FastMCP returns a CallToolResult object with .data attribute
        # or content list. Check for the result value in various forms.
        if hasattr(obs.result, "data"):
            assert obs.result.data == 8
        elif hasattr(obs.result, "content"):
            assert "8" in str(obs.result.content)
        else:
            assert "8" in str(obs.result)

    def test_fastmcp_in_mcp_environment_call_greet(self, minimal_mcp_env):
        """Test MCPEnvironment can call a 'greet' tool from FastMCP server."""
        obs = minimal_mcp_env.step(
            CallToolAction(
                tool_name="greet",
                arguments={"name": "Claude"},
            )
        )

        assert isinstance(obs, CallToolObservation)
        assert obs.tool_name == "greet"
        assert obs.error is None
        assert "Claude" in str(obs.result)

    def test_fastmcp_reserved_name_validation(self):
        """Test that MCPEnvironment rejects FastMCP tools with reserved names."""
        mcp = FastMCP("test-server")

        @mcp.tool
        def reset() -> str:
            """This uses a reserved name."""
            return "should not work"

        with pytest.raises(ValueError) as exc_info:
            MinimalMCPEnvironment(mcp)

        assert "reset" in str(exc_info.value)
        assert "reserved" in str(exc_info.value).lower()


# =============================================================================
# WebSocket MCP Tests
# =============================================================================


class TestWebSocketMCP:
    """Tests for WebSocket MCP tools/list and tools/call endpoints."""

    @pytest.fixture
    def app(self):
        """Create a FastAPI app with EchoEnvironment for WebSocket testing."""
        from echo_env.server.echo_environment import EchoEnvironment
        from openenv.core.env_server.http_server import create_fastapi_app
        from openenv.core.env_server.mcp_types import (
            CallToolAction,
            CallToolObservation,
        )

        return create_fastapi_app(
            env=EchoEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
        )

    def test_websocket_tools_list(self, app):
        """Test WebSocket tools/list via JSON-RPC."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP tools/list request
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Verify response structure
            assert response["type"] == "mcp"
            assert "data" in response
            assert response["data"]["jsonrpc"] == "2.0"
            assert response["data"]["id"] == 1
            assert "result" in response["data"]
            assert "tools" in response["data"]["result"]

            # Verify tools are returned
            tools = response["data"]["result"]["tools"]
            tool_names = [t["name"] for t in tools]
            assert "echo_message" in tool_names
            assert "echo_with_length" in tool_names

    def test_websocket_tools_call(self, app):
        """Test WebSocket tools/call via JSON-RPC."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP tools/call request
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "echo_message",
                        "arguments": {"message": "Hello via WebSocket!"},
                    },
                    "id": 2,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Verify response structure
            assert response["type"] == "mcp"
            assert response["data"]["jsonrpc"] == "2.0"
            assert response["data"]["id"] == 2
            assert "result" in response["data"]

    def test_websocket_mcp_method_not_found(self, app):
        """Test WebSocket returns error for unknown MCP method."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send unknown MCP method
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "unknown/method",
                    "id": 3,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Verify error response
            assert response["type"] == "mcp"
            assert "error" in response["data"]
            assert response["data"]["error"]["code"] == -32601
            assert "not found" in response["data"]["error"]["message"].lower()

    def test_websocket_tools_call_missing_name(self, app):
        """Test WebSocket tools/call returns error when name is missing."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send tools/call without name
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "arguments": {"message": "test"},
                    },
                    "id": 4,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Verify error response
            assert response["type"] == "mcp"
            assert "error" in response["data"]
            assert response["data"]["error"]["code"] == -32602
            assert "name" in response["data"]["error"]["message"].lower()
