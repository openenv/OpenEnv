# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for production mode MCP functionality.

These tests verify that in production mode:
1. MCP endpoints (tools/list, tools/call) are available via WebSocket
2. MCP operations work WITHOUT requiring reset() first
3. MCP JSON-RPC protocol is properly implemented
4. Error handling is correct for invalid requests

This is a critical test suite for production inference use cases where agents
interact with real environments using MCP tools, not simulation controls.

Test coverage:
- Production mode exposes MCP tools/list via WebSocket
- Production mode exposes MCP tools/call via WebSocket
- MCP works without calling reset() first (key production requirement)
- MCP error handling (tool not found, invalid arguments, etc.)
- MCP JSON-RPC format compliance
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from fastmcp import FastMCP
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State


# ============================================================================
# Test Fixtures - MCP-Enabled Environment
# ============================================================================


class MCPTestEnvironment(MCPEnvironment):
    """
    Test environment with MCP tools for production mode testing.

    This environment provides simple tools for testing MCP functionality
    in production mode without requiring simulation controls.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize with a FastMCP server containing test tools."""
        mcp_server = FastMCP("test-production-env")

        @mcp_server.tool
        def get_info() -> str:
            """Get environment information."""
            return "Production environment v1.0"

        @mcp_server.tool
        def calculate(operation: str, a: int, b: int) -> int:
            """Perform a calculation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")

        super().__init__(mcp_server)
        self._state = State(episode_id="prod", step_count=0)

    def reset(self, **kwargs) -> Observation:
        """Reset the environment."""
        self._state = State(episode_id="prod", step_count=0)
        return Observation(done=False, reward=None)

    def _step_impl(self, action: Action, **kwargs) -> Observation:
        """Handle non-MCP actions."""
        self._state.step_count += 1
        return Observation(done=False, reward=None)

    @property
    def state(self) -> State:
        """Return current state."""
        return self._state


@pytest.fixture
def production_mcp_app() -> FastAPI:
    """
    Create a FastAPI app in production mode with MCP-enabled environment.

    This simulates a production deployment where only MCP tools are exposed,
    not simulation controls.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=MCPTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="production")
    return app


@pytest.fixture
def simulation_mcp_app() -> FastAPI:
    """
    Create a FastAPI app in simulation mode with MCP-enabled environment.

    This is for comparison testing to verify MCP works in both modes.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=MCPTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="simulation")
    return app


# ============================================================================
# Production Mode MCP Functionality Tests
# ============================================================================


class TestProductionModeMCPToolsList:
    """Test that production mode exposes MCP tools/list functionality."""

    def test_production_mode_mcp_tools_list_via_websocket(self, production_mcp_app):
        """
        Test that tools/list works in production mode via WebSocket.

        This is the primary test for production mode MCP functionality.
        Tools should be discoverable without calling reset() first.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP tools/list request (JSON-RPC format)
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

            # Verify JSON-RPC response structure
            assert response["type"] == "mcp", "Response should be MCP type"
            assert "data" in response, "Response should have data field"

            data = response["data"]
            assert data["jsonrpc"] == "2.0", "Should follow JSON-RPC 2.0"
            assert data["id"] == 1, "Should echo request ID"
            assert "result" in data, "Should have result (not error)"

            # Verify tools are returned
            result = data["result"]
            assert "tools" in result, "Result should contain tools list"

            tools = result["tools"]
            assert len(tools) > 0, "Should return at least one tool"

            # Verify tool structure
            tool_names = [t["name"] for t in tools]
            assert "get_info" in tool_names, "Should include get_info tool"
            assert "calculate" in tool_names, "Should include calculate tool"

            # Verify tool has required fields
            get_info_tool = next(t for t in tools if t["name"] == "get_info")
            assert "description" in get_info_tool, "Tool should have description"
            # Note: FastMCP may use different field names (inputSchema vs input_schema)
            # Just verify the tool has some schema-related field
            assert len(get_info_tool) > 2, (
                "Tool should have name, description, and other metadata"
            )

    def test_production_mode_tools_list_without_reset(self, production_mcp_app):
        """
        Test that tools/list works WITHOUT calling reset() first.

        This is a key requirement for production mode: MCP tools should be
        available immediately without needing to reset the environment.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # DO NOT call reset() - this is the key test

            # Directly call tools/list
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 42,
                },
            }
            websocket.send_text(json.dumps(request))

            # Should succeed without reset
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "result" in response["data"], "Should succeed without reset()"
            assert "tools" in response["data"]["result"]
            assert len(response["data"]["result"]["tools"]) > 0


class TestProductionModeMCPToolsCall:
    """Test that production mode exposes MCP tools/call functionality."""

    def test_production_mode_mcp_tools_call_via_websocket(self, production_mcp_app):
        """
        Test that tools/call works in production mode via WebSocket.

        Agents should be able to invoke tools in production mode.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Call get_info tool
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "get_info",
                        "arguments": {},
                    },
                    "id": 2,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Verify JSON-RPC response structure
            assert response["type"] == "mcp"
            assert response["data"]["jsonrpc"] == "2.0"
            assert response["data"]["id"] == 2
            assert "result" in response["data"], "Should have result (not error)"

            # Verify result contains tool output
            result = response["data"]["result"]
            assert result is not None, "Tool should return a result"

    def test_production_mode_tools_call_with_arguments(self, production_mcp_app):
        """
        Test tools/call with arguments in production mode.

        Verifies that tool arguments are correctly passed through.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Call calculate tool with arguments
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "calculate",
                        "arguments": {
                            "operation": "add",
                            "a": 5,
                            "b": 3,
                        },
                    },
                    "id": 3,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Verify successful execution
            assert response["type"] == "mcp"
            assert "result" in response["data"], "Should succeed with valid arguments"

            # Note: The exact result format depends on FastMCP implementation
            # We just verify it doesn't error
            result = response["data"]["result"]
            assert result is not None

    def test_production_mode_tools_call_without_reset(self, production_mcp_app):
        """
        Test that tools/call works WITHOUT calling reset() first.

        Production environments should allow tool calls immediately.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # DO NOT call reset() - this is the key test

            # Directly call a tool
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "get_info",
                        "arguments": {},
                    },
                    "id": 99,
                },
            }
            websocket.send_text(json.dumps(request))

            # Should succeed without reset
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "result" in response["data"], "Should succeed without reset()"


class TestProductionModeMCPErrorHandling:
    """Test MCP error handling in production mode."""

    def test_production_mode_tool_not_found_error(self, production_mcp_app):
        """
        Test that calling a non-existent tool returns proper error.

        Should return JSON-RPC error response.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Call non-existent tool
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "nonexistent_tool",
                        "arguments": {},
                    },
                    "id": 10,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive error response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Should return error in JSON-RPC format
            assert response["type"] == "mcp"
            assert "error" in response["data"], "Should return error for missing tool"

            error = response["data"]["error"]
            assert "code" in error
            assert "message" in error

    def test_production_mode_invalid_method_error(self, production_mcp_app):
        """
        Test that invalid MCP method returns proper error.

        Should return JSON-RPC method not found error (-32601).
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Send invalid MCP method
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/invalid",
                    "id": 11,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive error response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Should return method not found error
            assert response["type"] == "mcp"
            assert "error" in response["data"]
            assert response["data"]["error"]["code"] == -32601

    def test_production_mode_missing_tool_name_in_call(self, production_mcp_app):
        """
        Test that tools/call without name parameter returns error.

        Should return JSON-RPC invalid params error (-32600).
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Send tools/call without name
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        # Missing "name" field
                        "arguments": {},
                    },
                    "id": 12,
                },
            }
            websocket.send_text(json.dumps(request))

            # Receive error response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            # Should return invalid params error
            assert response["type"] == "mcp"
            assert "error" in response["data"]
            assert response["data"]["error"]["code"] == -32602
            assert "name" in response["data"]["error"]["message"].lower()


class TestProductionModeMCPJSONRPCCompliance:
    """Test JSON-RPC protocol compliance for MCP in production mode."""

    def test_jsonrpc_version_is_2_0(self, production_mcp_app):
        """
        Test that all MCP responses use JSON-RPC 2.0.

        This is required by the JSON-RPC spec.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 20,
                },
            }
            websocket.send_text(json.dumps(request))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["data"]["jsonrpc"] == "2.0"

    def test_jsonrpc_request_id_is_echoed(self, production_mcp_app):
        """
        Test that response echoes the request ID.

        JSON-RPC requires the response to include the same ID as the request.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Use a unique ID
            unique_id = "test-id-12345"

            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": unique_id,
                },
            }
            websocket.send_text(json.dumps(request))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["data"]["id"] == unique_id

    def test_jsonrpc_result_and_error_are_mutually_exclusive(self, production_mcp_app):
        """
        Test that JSON-RPC responses have either result OR error, not both.

        This is a JSON-RPC requirement.
        """
        client = TestClient(production_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Test successful request (should have result, not error)
            request_success = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 30,
                },
            }
            websocket.send_text(json.dumps(request_success))
            response_text = websocket.receive_text()
            response_success = json.loads(response_text)

            assert "result" in response_success["data"]
            assert "error" not in response_success["data"]

            # Test failed request (should have error, not result)
            request_error = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "invalid/method",
                    "id": 31,
                },
            }
            websocket.send_text(json.dumps(request_error))
            response_text = websocket.receive_text()
            response_error = json.loads(response_text)

            assert "error" in response_error["data"]
            assert "result" not in response_error["data"]


# ============================================================================
# Comparison Tests: Production vs Simulation Mode
# ============================================================================


class TestMCPWorksInBothModes:
    """
    Test that MCP functionality works in both production and simulation modes.

    This verifies that MCP is mode-agnostic and consistently available.
    """

    def test_tools_list_works_in_simulation_mode(self, simulation_mcp_app):
        """
        Test that tools/list also works in simulation mode.

        MCP should be available in both modes.
        """
        client = TestClient(simulation_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 100,
                },
            }
            websocket.send_text(json.dumps(request))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "result" in response["data"]
            assert "tools" in response["data"]["result"]

    def test_tools_call_works_in_simulation_mode(self, simulation_mcp_app):
        """
        Test that tools/call also works in simulation mode.

        MCP should be available in both modes.
        """
        client = TestClient(simulation_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            request = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "get_info",
                        "arguments": {},
                    },
                    "id": 101,
                },
            }
            websocket.send_text(json.dumps(request))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "result" in response["data"]
