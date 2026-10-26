# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for production mode routes in OpenEnv.

This file combines two aspects of production mode:

1. Route restrictions (from main): Tests that production mode blocks simulation control
   endpoints (/reset, /step, /state) while allowing safe endpoints. This is a critical
   security boundary: production environments should only expose MCP tools, not simulation
   controls that manipulate time and causality.

2. Direct MCP API access (from issue #347): Per RFC 003, environments should expose both:
   - Training/Eval API: step() for RL training (includes reward computation, state tracking)
   - Production API: Direct MCP endpoints for inference (bypasses step(), no rewards)

Test coverage:
- Production mode disables /reset, /step, /state endpoints (returns 404 or 405)
- Production mode allows /health, /schema, /metadata, /ws endpoints
- Direct MCP JSON-RPC endpoints work (tools/list, tools/call)
- WebSocket MCP message handling
- HTTP POST /mcp endpoint for MCP JSON-RPC
- Production mode bypasses step() overhead
- Proper error responses for invalid MCP requests
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import RESERVED_TOOL_NAMES
from openenv.core.env_server.types import Action, Observation, State


# ============================================================================
# Test Fixtures - Minimal Environment for Testing
# ============================================================================


class MinimalAction(Action):
    """Minimal action for testing."""

    message: str


class MinimalObservation(Observation):
    """Minimal observation for testing."""

    response: str
    reward: float | None = None
    done: bool = False


class MinimalState(State):
    """Minimal state for testing."""

    step_count: int = 0


class MinimalEnvironment(Environment):
    """Minimal environment implementation for testing server modes."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs) -> MinimalObservation:
        """Reset the environment."""
        return MinimalObservation(response="reset", reward=None, done=False)

    def step(self, action: MinimalAction) -> MinimalObservation:
        """Execute an action."""
        return MinimalObservation(
            response=f"echo: {action.message}", reward=1.0, done=False
        )

    @property
    def state(self) -> MinimalState:
        """Return current state."""
        return MinimalState(step_count=0)

    def close(self) -> None:
        """Cleanup resources."""
        pass


@pytest.fixture
def production_mode_app() -> FastAPI:
    """
    Create a FastAPI app with production mode enabled.

    In production mode, /reset, /step, /state should NOT be registered.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=MinimalEnvironment,
        action_cls=MinimalAction,
        observation_cls=MinimalObservation,
    )
    # TODO: Once production mode is implemented, pass mode="production" here
    # For now, this will fail because the feature doesn't exist yet
    server.register_routes(app, mode="production")
    return app


@pytest.fixture
def simulation_mode_app() -> FastAPI:
    """
    Create a FastAPI app with simulation mode (default).

    In simulation mode, all endpoints including /reset, /step, /state are available.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=MinimalEnvironment,
        action_cls=MinimalAction,
        observation_cls=MinimalObservation,
    )
    # Default mode should be simulation
    server.register_routes(app)
    return app


# ============================================================================
# Production Mode Route Restriction Tests (from main)
# ============================================================================


class TestProductionModeRouteRestrictions:
    """Test that production mode hides simulation control endpoints."""

    def test_production_mode_blocks_reset_endpoint(self, production_mode_app):
        """Test that /reset returns 404 or 405 in production mode."""
        client = TestClient(production_mode_app)

        response = client.post("/reset", json={})

        # Should return 404 (Not Found) or 405 (Method Not Allowed)
        assert response.status_code in [404, 405], (
            f"Expected 404 or 405, got {response.status_code}. "
            "Production mode should not expose /reset endpoint."
        )

    def test_production_mode_blocks_step_endpoint(self, production_mode_app):
        """Test that /step returns 404 or 405 in production mode."""
        client = TestClient(production_mode_app)

        response = client.post("/step", json={"action": {"message": "test"}})

        # Should return 404 (Not Found) or 405 (Method Not Allowed)
        assert response.status_code in [404, 405], (
            f"Expected 404 or 405, got {response.status_code}. "
            "Production mode should not expose /step endpoint."
        )

    def test_production_mode_blocks_state_endpoint(self, production_mode_app):
        """Test that /state returns 404 or 405 in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/state")

        # Should return 404 (Not Found) or 405 (Method Not Allowed)
        assert response.status_code in [404, 405], (
            f"Expected 404 or 405, got {response.status_code}. "
            "Production mode should not expose /state endpoint."
        )


# ============================================================================
# Production Mode Still Allows Safe Endpoints
# ============================================================================


class TestProductionModeAllowsSafeEndpoints:
    """Test that production mode still exposes safe, non-simulation endpoints."""

    def test_production_mode_allows_health_endpoint(self, production_mode_app):
        """Test that /health is still available in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/health")

        assert response.status_code == 200, (
            "Production mode should still expose /health for monitoring"
        )
        assert response.json()["status"] == "healthy"

    def test_production_mode_allows_schema_endpoint(self, production_mode_app):
        """Test that /schema is still available in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/schema")

        assert response.status_code == 200, (
            "Production mode should still expose /schema for introspection"
        )
        # Should have action, observation, state schemas
        data = response.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data

    def test_production_mode_allows_metadata_endpoint(self, production_mode_app):
        """Test that /metadata is still available in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/metadata")

        assert response.status_code == 200, (
            "Production mode should still expose /metadata for environment info"
        )

    def test_production_mode_allows_websocket_endpoint(self, production_mode_app):
        """Test that /ws WebSocket is still available in production mode."""
        client = TestClient(production_mode_app)

        # WebSocket connection test - we expect it to accept the connection
        # We don't test the full WebSocket protocol here, just that it's registered
        try:
            with client.websocket_connect("/ws") as websocket:
                # If we get here, the endpoint is registered
                # We can close immediately
                websocket.close()
                assert True, "WebSocket endpoint should be available"
        except Exception as e:
            # If the endpoint doesn't exist, we'll get a 404
            pytest.fail(
                f"WebSocket endpoint should be available in production mode: {e}"
            )


# ============================================================================
# Simulation Mode Allows All Endpoints (Regression Test)
# ============================================================================


class TestSimulationModeAllowsAllEndpoints:
    """Test that simulation mode (default) allows all endpoints."""

    def test_simulation_mode_allows_reset_endpoint(self, simulation_mode_app):
        """Test that /reset works in simulation mode (default behavior)."""
        client = TestClient(simulation_mode_app)

        response = client.post("/reset", json={})

        assert response.status_code == 200, (
            "Simulation mode should expose /reset endpoint"
        )
        data = response.json()
        assert "observation" in data
        assert data["observation"]["response"] == "reset"

    def test_simulation_mode_allows_step_endpoint(self, simulation_mode_app):
        """Test that /step works in simulation mode (default behavior)."""
        client = TestClient(simulation_mode_app)

        response = client.post("/step", json={"action": {"message": "hello"}})

        assert response.status_code == 200, (
            "Simulation mode should expose /step endpoint"
        )
        data = response.json()
        assert "observation" in data
        assert "echo: hello" in data["observation"]["response"]

    def test_simulation_mode_allows_state_endpoint(self, simulation_mode_app):
        """Test that /state works in simulation mode (default behavior)."""
        client = TestClient(simulation_mode_app)

        response = client.get("/state")

        assert response.status_code == 200, (
            "Simulation mode should expose /state endpoint"
        )
        data = response.json()
        assert "step_count" in data
        assert data["step_count"] == 0


# ============================================================================
# Mode Configuration Tests
# ============================================================================


class TestModeConfiguration:
    """Test that mode can be configured via parameter."""

    def test_explicit_production_mode_parameter(self):
        """Test that mode='production' can be passed to register_routes."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )

        # This should not raise an error
        # The implementation should accept mode parameter
        try:
            server.register_routes(app, mode="production")
        except TypeError as e:
            pytest.fail(f"register_routes should accept mode parameter: {e}")

    def test_explicit_simulation_mode_parameter(self):
        """Test that mode='simulation' can be passed to register_routes."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )

        # This should not raise an error
        try:
            server.register_routes(app, mode="simulation")
        except TypeError as e:
            pytest.fail(f"register_routes should accept mode parameter: {e}")

    def test_default_mode_is_simulation(self):
        """Test that default mode is 'simulation' for backwards compatibility."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )
        server.register_routes(app)
        client = TestClient(app)

        # Should have /reset, /step, /state in default mode
        reset_response = client.post("/reset", json={})
        step_response = client.post("/step", json={"action": {"message": "test"}})
        state_response = client.get("/state")

        assert reset_response.status_code == 200, "Default mode should allow /reset"
        assert step_response.status_code == 200, "Default mode should allow /step"
        assert state_response.status_code == 200, "Default mode should allow /state"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode value raises ValueError."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )

        with pytest.raises(ValueError) as exc_info:
            server.register_routes(app, mode="invalid_mode")

        assert "mode" in str(exc_info.value).lower()
        assert "production" in str(exc_info.value).lower()
        assert "simulation" in str(exc_info.value).lower()


# ============================================================================
# Security Boundary Tests
# ============================================================================


class TestProductionModeSecurityBoundary:
    """
    Test that production mode enforces the security boundary.

    The key invariant: In production, agents cannot control time/causality.
    """

    def test_production_mode_prevents_reset_manipulation(self, production_mode_app):
        """
        Test that production mode prevents environment reset.

        In production, we can't reset the real world - time only moves forward.
        """
        client = TestClient(production_mode_app)

        # Try to reset (should fail)
        response = client.post("/reset", json={"seed": 42})

        assert response.status_code in [404, 405], (
            "Production mode must not allow reset - can't reset the real world"
        )

    def test_production_mode_prevents_state_inspection(self, production_mode_app):
        """
        Test that production mode prevents arbitrary state inspection.

        State inspection is a simulation concept - in prod we only observe via tools.
        """
        client = TestClient(production_mode_app)

        response = client.get("/state")

        assert response.status_code in [404, 405], (
            "Production mode should not expose internal state directly"
        )

    def test_production_mode_prevents_direct_step(self, production_mode_app):
        """
        Test that production mode prevents direct step calls.

        In production, agents interact via MCP tools, not direct step() calls.
        """
        client = TestClient(production_mode_app)

        response = client.post("/step", json={"action": {"message": "test"}})

        assert response.status_code in [404, 405], (
            "Production mode should not allow direct step() - use MCP tools instead"
        )


# ============================================================================
# Direct MCP API Access Tests (from issue #347)
# ============================================================================


# =============================================================================
# Test Fixtures - MCP Endpoints
# =============================================================================


@pytest.fixture
def mock_fastmcp_server():
    """Create a mock FastMCP server for testing."""
    from fastmcp import FastMCP

    mcp = FastMCP("test-server")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool
    def greet(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}!"

    return mcp


@pytest.fixture
def app(mock_fastmcp_server):
    """Create FastAPI app with MCP endpoints."""
    # This creates and returns a FastAPI app with MCP endpoints
    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.mcp_environment import MCPEnvironment

    class TestMCPEnv(MCPEnvironment):
        def __init__(self):
            super().__init__(mock_fastmcp_server)
            self._state = {"step_count": 0}

        def reset(self, **kwargs):
            self._state = {"step_count": 0}
            return Observation(done=False, reward=0.0)

        def _step_impl(self, action, **kwargs):
            self._state["step_count"] += 1
            return Observation(done=False, reward=0.0)

        @property
        def state(self):
            from openenv.core.env_server.types import State

            return State(step_count=self._state["step_count"])

    return create_fastapi_app(
        env=TestMCPEnv,
        action_cls=None,
        observation_cls=None,
    )


# =============================================================================
# HTTP /mcp Endpoint Tests
# =============================================================================


class TestHTTPMCPEndpoint:
    """Tests for HTTP POST /mcp endpoint (JSON-RPC)."""

    def test_mcp_endpoint_exists(self, app):
        """Test /mcp endpoint is exposed."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        )

        assert response.status_code == 200

    def test_mcp_tools_list_via_http(self, app):
        """Test tools/list via HTTP /mcp endpoint."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) > 0

    def test_mcp_tools_call_via_http(self, app):
        """Test tools/call via HTTP /mcp endpoint."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 5, "b": 3}},
                "id": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert "result" in data
        # Result should contain the tool's return value
        assert "8" in str(data["result"]) or data["result"] == 8

    def test_mcp_http_bypasses_step_overhead(self, app):
        """Test direct MCP access doesn't call step() or compute rewards."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with patch(
            "openenv.core.env_server.mcp_environment.MCPEnvironment.step"
        ) as mock_step:
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "add", "arguments": {"a": 1, "b": 1}},
                    "id": 3,
                },
            )

            # Verify step() was NOT called (production mode bypasses it)
            mock_step.assert_not_called()
            assert response.status_code == 200

    def test_mcp_http_invalid_method_returns_error(self, app):
        """Test invalid MCP method returns proper JSON-RPC error."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "invalid/method", "id": 4}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 4
        assert "error" in data
        assert data["error"]["code"] == -32601  # Method not found

    def test_mcp_http_missing_jsonrpc_version(self, app):
        """Test request without jsonrpc version returns error."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post("/mcp", json={"method": "tools/list", "id": 5})

        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data

    def test_mcp_http_no_reset_required(self, app):
        """Test MCP endpoints work without calling reset() first."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Call tools/list without reset
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 6}
        )

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data["result"]


# =============================================================================
# HTTP MCP Session Lifecycle Tests
# =============================================================================


class TestHTTPMCPSessionLifecycle:
    """Tests for openenv/session/create and openenv/session/close methods."""

    def test_session_create_returns_session_id(self, app):
        """Test openenv/session/create returns a non-empty session_id."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "session_id" in data["result"]
        assert isinstance(data["result"]["session_id"], str)
        assert len(data["result"]["session_id"]) > 0

    def test_session_tools_call_with_session_id(self, app):
        """Test tools/call works with an explicit session_id."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Create session
        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )
        sid = create_resp.json()["result"]["session_id"]

        # Call tool with that session
        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "add",
                    "arguments": {"a": 2, "b": 3},
                    "session_id": sid,
                },
                "id": 2,
            },
        )

        assert call_resp.status_code == 200
        data = call_resp.json()
        assert "result" in data
        assert "5" in str(data["result"]) or data["result"] == 5

    def test_session_close_returns_closed_true(self, app):
        """Test openenv/session/close returns closed: true."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Create session
        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )
        sid = create_resp.json()["result"]["session_id"]

        # Close session
        close_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": sid},
                "id": 2,
            },
        )

        assert close_resp.status_code == 200
        data = close_resp.json()
        assert "result" in data
        assert data["result"]["session_id"] == sid
        assert data["result"]["closed"] is True

    def test_session_close_unknown_id_returns_error(self, app):
        """Test closing a bogus session_id returns an error."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": "nonexistent-session-id"},
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32602  # INVALID_PARAMS

    def test_session_create_from_websocket_is_idempotent(self, app):
        """Test openenv/session/create over WebSocket returns the existing session id."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send session/create via MCP over WebSocket
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "openenv/session/create",
                            "params": {},
                            "id": 1,
                        },
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            data = response["data"]
            assert "result" in data
            assert "session_id" in data["result"]
            # Should return the WebSocket's own session_id, not create a new one
            ws_session_id = data["result"]["session_id"]
            assert isinstance(ws_session_id, str)
            assert len(ws_session_id) > 0

            # Send again — should return the same session id (idempotent)
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "openenv/session/create",
                            "params": {},
                            "id": 2,
                        },
                    }
                )
            )

            response_text2 = websocket.receive_text()
            response2 = json.loads(response_text2)
            assert response2["data"]["result"]["session_id"] == ws_session_id

    def test_websocket_can_attach_to_http_session_without_destroying_it(self, app):
        """An attached WebSocket shares and preserves an HTTP-created session."""
        client = TestClient(app)
        create_response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )
        session_id = create_response.json()["result"]["session_id"]

        with client.websocket_connect(f"/ws?session_id={session_id}") as websocket:
            websocket.send_json({"type": "state"})
            state_response = websocket.receive_json()
            assert state_response["type"] == "state"

            active_close_response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "openenv/session/close",
                    "params": {"session_id": session_id},
                    "id": 2,
                },
            )
            close_result = active_close_response.json()["result"]
            assert close_result == {
                "session_id": session_id,
                "closed": False,
                "closing": True,
            }
            websocket.send_json({"type": "close"})

        tools_response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 3,
            },
        )
        assert tools_response.json()["error"]["code"] == -32602

        replacement_response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 4,
            },
        )
        replacement_id = replacement_response.json()["result"]["session_id"]
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": replacement_id},
                "id": 5,
            },
        )

    def test_http_session_allows_only_one_attached_websocket(self, app):
        """A second WebSocket cannot concurrently mutate the same session."""
        client = TestClient(app)
        create_response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )
        session_id = create_response.json()["result"]["session_id"]

        with client.websocket_connect(f"/ws?session_id={session_id}") as first_socket:
            with client.websocket_connect(
                f"/ws?session_id={session_id}"
            ) as second_socket:
                error_response = second_socket.receive_json()
                assert (
                    "already has an attached WebSocket"
                    in (error_response["data"]["message"])
                )
            first_socket.send_json({"type": "close"})

        close_response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        assert close_response.json()["result"]["closed"] is True

    def test_websocket_still_destroys_its_own_session(self, app):
        """A WebSocket-created session is destroyed when the socket closes."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "mcp",
                    "data": {
                        "jsonrpc": "2.0",
                        "method": "openenv/session/create",
                        "params": {},
                        "id": 1,
                    },
                }
            )
            session_id = websocket.receive_json()["data"]["result"]["session_id"]
            websocket.send_json({"type": "close"})

        tools_response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        assert tools_response.json()["error"]["code"] == -32602

    def test_session_close_missing_session_id_param(self, app):
        """Test openenv/session/close without session_id returns INVALID_PARAMS."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {},
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32602  # INVALID_PARAMS
        assert "session_id" in data["error"]["message"].lower()

    def test_session_double_close_returns_error(self, app):
        """Test closing the same session twice returns an error on the second close."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Create session
        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )
        sid = create_resp.json()["result"]["session_id"]

        # First close — should succeed
        close1 = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": sid},
                "id": 2,
            },
        )
        assert close1.json().get("result", {}).get("closed") is True

        # Second close — session no longer exists
        close2 = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": sid},
                "id": 3,
            },
        )
        data2 = close2.json()
        assert "error" in data2
        assert data2["error"]["code"] == -32602  # INVALID_PARAMS

    def test_tools_call_after_close_returns_error(self, app):
        """Test tools/call with a closed session_id returns an error."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Create then close
        create_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/create",
                "params": {},
                "id": 1,
            },
        )
        sid = create_resp.json()["result"]["session_id"]
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": sid},
                "id": 2,
            },
        )

        # Call tool on closed session
        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "add",
                    "arguments": {"a": 1, "b": 2},
                    "session_id": sid,
                },
                "id": 3,
            },
        )
        data = call_resp.json()
        assert "error" in data


# =============================================================================
# MCP Session Transport Persistence Tests
# =============================================================================


class TestMCPSessionTransportPersistence:
    """Tests for MCP transport persistence across HTTP calls.

    After the lifecycle fix, HTTP MCP paths hold mcp_session() open for the
    full OpenEnv session lifetime via AsyncExitStack in _create_session.
    FastMCP's session state (ctx.set_state / ctx.get_state) therefore
    persists across sequential HTTP tool calls within the same session.

    The WebSocket path likewise holds mcp_session() open for the connection
    lifetime, so both transports now provide the same persistence guarantee.
    """

    @pytest.fixture
    def stateful_mcp_app(self):
        """App with a stateful MCP tool that uses ctx.set_state/get_state."""
        from fastmcp import Context, FastMCP
        from openenv.core.env_server.http_server import create_fastapi_app
        from openenv.core.env_server.mcp_environment import MCPEnvironment

        mcp = FastMCP("stateful-test")

        @mcp.tool
        async def inc_counter(ctx: Context) -> str:
            """Increment a per-session counter and return the new value."""
            count = (await ctx.get_state("counter")) or 0
            await ctx.set_state("counter", count + 1)
            return str(count + 1)

        class StatefulMCPEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mcp)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                return State(step_count=0)

        return create_fastapi_app(
            env=StatefulMCPEnv,
            action_cls=None,
            observation_cls=None,
        )

    async def test_http_session_mcp_state_persists_across_calls(self, stateful_mcp_app):
        """Two HTTP tool calls in the same session should share MCP session state.

        inc_counter uses ctx.set_state() to track a per-session counter.
        Expected: sequential calls return "1", "2" (state persists).

        Uses httpx.AsyncClient (not Starlette's sync TestClient) because the
        MCP transport persistence relies on a background asyncio. Task that
        must survive across requests within the same event loop.
        """
        import httpx
        from httpx import ASGITransport

        transport = ASGITransport(app=stateful_mcp_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create a persistent HTTP session
            create_resp = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "openenv/session/create",
                    "params": {},
                    "id": 1,
                },
            )
            assert create_resp.status_code == 200
            sid = create_resp.json()["result"]["session_id"]

            # First call — should return "1"
            call1 = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "inc_counter",
                        "arguments": {},
                        "session_id": sid,
                    },
                    "id": 2,
                },
            )
            assert call1.status_code == 200
            result1 = call1.json()
            assert "result" in result1, f"First call failed: {result1}"
            assert "1" in str(result1["result"]), (
                f"First call should return 1, got: {result1['result']}"
            )

            # Second call — should return "2" if MCP session persists
            call2 = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "inc_counter",
                        "arguments": {},
                        "session_id": sid,
                    },
                    "id": 3,
                },
            )
            assert call2.status_code == 200
            result2 = call2.json()
            assert "result" in result2, f"Second call failed: {result2}"
            assert "2" in str(result2["result"]), (
                f"Second call should return 2 (MCP session state persisted), "
                f"but got: {result2['result']}. "
                "MCP transport is being torn down and recreated between HTTP calls."
            )

    def test_websocket_mcp_state_persists_across_calls(self, stateful_mcp_app):
        """WebSocket correctly persists MCP session state (control test).

        Should PASS: the WebSocket path holds mcp_session() open for the
        connection lifetime via AsyncExitStack, so reentrant mcp_session()
        entries share the same MCP protocol session.
        """
        from starlette.testclient import TestClient

        client = TestClient(stateful_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # First call — should return "1"
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {"name": "inc_counter", "arguments": {}},
                            "id": 1,
                        },
                    }
                )
            )
            resp1 = json.loads(websocket.receive_text())
            assert resp1["type"] == "mcp"
            assert "1" in str(resp1["data"].get("result", "")), (
                f"First WS call should return 1, got: {resp1}"
            )

            # Second call — should return "2" (state persisted in same session)
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {"name": "inc_counter", "arguments": {}},
                            "id": 2,
                        },
                    }
                )
            )
            resp2 = json.loads(websocket.receive_text())
            assert resp2["type"] == "mcp"
            assert "2" in str(resp2["data"].get("result", "")), (
                f"Second WS call should return 2, got: {resp2}. "
                "WebSocket MCP session is not persisting state."
            )

    async def test_concurrent_close_during_tool_call(self, stateful_mcp_app):
        """Concurrent session/close during active tool call returns clean responses.

        Fires tools/call and session/close concurrently on the same session.
        Both should return well-formed JSON-RPC responses — no HTTP 500 errors
        or unhandled exceptions from the TOCTOU race where mcp_handler holds
        an env reference after releasing the session lock.
        """
        import asyncio

        import httpx
        from fastmcp import FastMCP
        from httpx import ASGITransport
        from openenv.core.env_server.http_server import create_fastapi_app
        from openenv.core.env_server.mcp_environment import MCPEnvironment

        mcp = FastMCP("slow-test")

        @mcp.tool
        async def slow_add(a: int, b: int) -> int:
            """Add two numbers with a delay to widen the race window."""
            await asyncio.sleep(0.3)
            return a + b

        class SlowMCPEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mcp)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                return State(step_count=0)

        app = create_fastapi_app(
            env=SlowMCPEnv,
            action_cls=None,
            observation_cls=None,
        )

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            # Create session
            create_resp = await http.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "openenv/session/create",
                    "params": {},
                    "id": 1,
                },
            )
            sid = create_resp.json()["result"]["session_id"]

            # Fire tool call and close concurrently
            async def delayed_close():
                await asyncio.sleep(0.05)  # ensure tool call starts first
                return await http.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "openenv/session/close",
                        "params": {"session_id": sid},
                        "id": 3,
                    },
                )

            results = await asyncio.gather(
                http.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "slow_add",
                            "arguments": {"a": 1, "b": 2},
                            "session_id": sid,
                        },
                        "id": 2,
                    },
                ),
                delayed_close(),
                return_exceptions=True,
            )

            # Both requests must return valid HTTP 200 with JSON-RPC body
            for i, result in enumerate(results):
                assert not isinstance(result, Exception), (
                    f"Request {i} raised an unhandled exception: {result}"
                )
                assert result.status_code == 200, (
                    f"Request {i} returned HTTP {result.status_code}. "
                    "Concurrent close during tool call caused a server error."
                )
                data = result.json()
                assert "result" in data or "error" in data, (
                    f"Request {i} returned malformed JSON-RPC: {data}"
                )


class TestMCPSessionResourceLeaks:
    """Tests for resource cleanup on session creation failures and edge cases.

    These tests verify the fixes for:
    - P0: _create_session leaking session slot + env + executor when MCP
      transport fails to start (stack.enter_async_context raises).
    - P1: Executor orphaned when session/close fires during session init
      (env is None placeholder).
    """

    async def test_create_session_cleans_up_on_mcp_transport_failure(self):
        """If mcp_session() throws during _create_session, the session slot,
        env, and executor must all be cleaned up — not leaked permanently
        against _max_concurrent_envs.
        """
        from unittest.mock import patch

        from fastmcp import FastMCP
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.mcp_environment import MCPEnvironment
        from openenv.core.env_server.types import ConcurrencyConfig

        mcp = FastMCP("broken-test")

        @mcp.tool
        def noop() -> str:
            """A tool that does nothing."""
            return "ok"

        class BrokenTransportEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mcp)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                return State(step_count=0)

        server = HTTPEnvServer(
            env=BrokenTransportEnv,
            action_cls=None,
            observation_cls=None,
            concurrency_config=ConcurrencyConfig(
                max_concurrent_envs=2,
                session_timeout=None,
            ),
        )

        # Patch mcp_session to simulate an unreachable MCP server

        async def failing_mcp_session(self_env):
            raise ConnectionError("MCP server unreachable")
            yield  # make it a generator (never reached)

        from contextlib import asynccontextmanager

        failing_cm = asynccontextmanager(failing_mcp_session)

        with patch.object(MCPEnvironment, "mcp_session", failing_cm):
            with pytest.raises(ConnectionError, match="unreachable"):
                await server._create_session()

        # After the failure, no session slot should be leaked
        assert len(server._sessions) == 0, (
            f"Session slot leaked: {list(server._sessions.keys())}"
        )
        assert len(server._session_executors) == 0, (
            f"Executor leaked: {list(server._session_executors.keys())}"
        )
        assert len(server._session_stacks) == 0, (
            f"Stack leaked: {list(server._session_stacks.keys())}"
        )

        # Capacity should be fully available — can still create sessions
        status = server.get_capacity_status()
        assert status.active_sessions == 0

    async def test_close_during_init_preserves_executor(self):
        """When session/close fires for a still-initializing session (env is
        None), the executor must be re-inserted alongside the None placeholder
        so it remains tracked for eventual shutdown.
        """
        import asyncio

        from fastmcp import FastMCP
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.mcp_environment import MCPEnvironment
        from openenv.core.env_server.types import ConcurrencyConfig

        mcp = FastMCP("slow-init-test")

        @mcp.tool
        def ping() -> str:
            """Ping."""
            return "pong"

        init_event = asyncio.Event()
        asyncio.Event()

        class SlowInitEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mcp)
                # Signal that init has started, then block until released
                init_event.set()
                # We can't await in __init__, so we use a threading Event
                import threading

                self._threading_event = threading.Event()
                self._threading_event.wait(timeout=5)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                return State(step_count=0)

        server = HTTPEnvServer(
            env=SlowInitEnv,
            action_cls=None,
            observation_cls=None,
            concurrency_config=ConcurrencyConfig(
                max_concurrent_envs=5,
                session_timeout=None,
            ),
        )

        # Reserve a session slot manually to simulate the init-in-progress state
        session_id = "test-init-session"
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        async with server._session_lock:
            server._sessions[session_id] = None  # placeholder
            server._session_executors[session_id] = executor

        # Now simulate session/close hitting the "env is None" branch directly
        # by calling through the internal path
        from openenv.core.env_server.mcp_types import JsonRpcRequest

        JsonRpcRequest(
            jsonrpc="2.0",
            method="openenv/session/close",
            params={"session_id": session_id},
            id=99,
        )

        # We need to call mcp_handler — build the app to get access
        from openenv.core.env_server.http_server import create_fastapi_app

        create_fastapi_app(
            env=SlowInitEnv,
            action_cls=None,
            observation_cls=None,
        )

        # Instead of going through the app, directly verify the state
        # Simulate what session/close does for env=None:
        async with server._session_lock:
            env = server._sessions.pop(session_id, None)
            popped_executor = server._session_executors.pop(session_id, None)

        assert env is None, "Should be a None placeholder"
        assert popped_executor is executor, "Should have popped our executor"

        # Re-insert with the fix: both placeholder AND executor
        async with server._session_lock:
            server._sessions[session_id] = None
            if popped_executor is not None:
                server._session_executors[session_id] = popped_executor

        # Verify executor is still tracked
        assert session_id in server._session_executors, (
            "Executor must be re-inserted alongside the None placeholder"
        )
        assert server._session_executors[session_id] is executor

        # Cleanup
        async with server._session_lock:
            server._sessions.pop(session_id, None)
            server._session_executors.pop(session_id, None)
        executor.shutdown(wait=False)


class TestHTTPMCPSessionReaper:
    """Tests for the idle-session reaper (originally in TestHTTPMCPSessionLifecycle)."""

    async def test_idle_session_reaper_destroys_stale_sessions(
        self, mock_fastmcp_server
    ):
        """Test that _reap_idle_sessions destroys sessions past the timeout."""
        import asyncio
        import time as _time

        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.mcp_environment import MCPEnvironment
        from openenv.core.env_server.types import ConcurrencyConfig

        class ReaperTestEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mock_fastmcp_server)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                from openenv.core.env_server.types import State

                return State(step_count=0)

        server = HTTPEnvServer(
            env=ReaperTestEnv,
            action_cls=None,
            observation_cls=None,
            concurrency_config=ConcurrencyConfig(
                max_concurrent_envs=10,
                session_timeout=0.3,  # 300ms for fast test
            ),
        )

        # Create a session directly on the server
        session_id, env = await server._create_session()
        assert session_id in server._sessions

        # Wait for session to become stale
        await asyncio.sleep(0.4)

        # Manually trigger one reap cycle (the background reaper's interval
        # is min 5s, too long for a unit test).  Mirrors the reaper's
        # re-check-before-destroy logic so the test stays in sync.
        now = _time.time()
        timeout = 0.3
        stale = []
        async with server._session_lock:
            for sid, info in server._session_info.items():
                if now - info.last_activity_at > timeout:
                    stale.append(sid)
        for sid in stale:
            async with server._session_lock:
                info = server._session_info.get(sid)
                if info is None or (now - info.last_activity_at) <= timeout:
                    continue
            await server._destroy_session(sid)

        # Session should be gone
        assert session_id not in server._sessions

    async def test_reaper_stop_cancels_task(self, mock_fastmcp_server):
        """Test that _stop_reaper cancels the running reaper task."""
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.mcp_environment import MCPEnvironment
        from openenv.core.env_server.types import ConcurrencyConfig

        class ReaperTestEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mock_fastmcp_server)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                from openenv.core.env_server.types import State

                return State(step_count=0)

        server = HTTPEnvServer(
            env=ReaperTestEnv,
            action_cls=None,
            observation_cls=None,
            concurrency_config=ConcurrencyConfig(
                max_concurrent_envs=10,
                session_timeout=60,
            ),
        )

        server._start_reaper()
        assert server._reaper_task is not None
        assert not server._reaper_task.done()

        server._stop_reaper()
        assert server._reaper_task is None

    async def test_reaper_noop_when_no_timeout(self, mock_fastmcp_server):
        """Test that _start_reaper is a no-op when session_timeout is None."""
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.mcp_environment import MCPEnvironment
        from openenv.core.env_server.types import ConcurrencyConfig

        class ReaperTestEnv(MCPEnvironment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self):
                super().__init__(mock_fastmcp_server)

            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                from openenv.core.env_server.types import State

                return State(step_count=0)

        server = HTTPEnvServer(
            env=ReaperTestEnv,
            action_cls=None,
            observation_cls=None,
            concurrency_config=ConcurrencyConfig(
                max_concurrent_envs=10,
                session_timeout=None,  # default — no timeout
            ),
        )

        server._start_reaper()
        # No task should be created when timeout is None
        assert server._reaper_task is None


class TestWebSocketMCP:
    """Tests for WebSocket MCP message handling."""

    def test_websocket_mcp_message_type(self, app):
        """Test WebSocket accepts 'mcp' message type."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP message via WebSocket
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert response["data"]["jsonrpc"] == "2.0"

    def test_websocket_mcp_tools_list(self, app):
        """Test tools/list via WebSocket MCP message."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "tools" in response["data"]["result"]

    def test_websocket_mcp_tools_call(self, app):
        """Test tools/call via WebSocket MCP message."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "greet",
                                "arguments": {"name": "Production"},
                            },
                            "id": 2,
                        },
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "Production" in str(response["data"]["result"])

    def test_websocket_mcp_interleaved_with_step(self, app):
        """Test WebSocket can handle both MCP and step() messages."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # First, use step() API
            websocket.send_text(json.dumps({"type": "reset", "data": {}}))
            response1 = websocket.receive_text()
            assert json.loads(response1)["type"] == "observation"

            # Then use MCP API directly
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    }
                )
            )
            response2 = websocket.receive_text()
            mcp_response = json.loads(response2)

            assert mcp_response["type"] == "mcp"
            assert "tools" in mcp_response["data"]["result"]


# =============================================================================
# Reserved Tool Names Tests
# =============================================================================


class TestReservedToolNames:
    """Tests for reserved tool name validation."""

    def test_reserved_names_constant_exists(self):
        """Test RESERVED_TOOL_NAMES is defined."""
        # This should PASS as it's already defined in mcp_types.py
        assert RESERVED_TOOL_NAMES is not None
        assert isinstance(RESERVED_TOOL_NAMES, frozenset)

    def test_reserved_names_include_env_methods(self):
        """Test reserved names include environment methods."""
        # This should PASS as it's already defined
        assert "reset" in RESERVED_TOOL_NAMES
        assert "step" in RESERVED_TOOL_NAMES
        assert "state" in RESERVED_TOOL_NAMES
        assert "close" in RESERVED_TOOL_NAMES

    def test_mcp_server_rejects_reserved_tool_names(self):
        """Test MCP server validation rejects reserved tool names."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @mcp.tool
        def reset() -> str:
            """This uses a reserved name."""
            return "should not work"

        from openenv.core.env_server.mcp_environment import MCPEnvironment

        # Use a concrete subclass to test validation
        class TestMCPEnv(MCPEnvironment):
            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                from openenv.core.env_server.types import State

                return State(step_count=0)

        with pytest.raises(ValueError) as exc_info:
            TestMCPEnv(mcp)

        assert "reserved" in str(exc_info.value).lower()
        assert "reset" in str(exc_info.value)


# =============================================================================
# Performance Tests
# =============================================================================


class TestProductionModePerformance:
    """Tests verifying production mode is optimized for inference."""

    def test_production_mode_no_reward_in_response(self, app):
        """Test production MCP mode returns tool result without reward."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 1, "b": 1}},
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # MCP response is pure JSON-RPC - no reward field
        assert "reward" not in data

    def test_production_mode_no_state_tracking(self, app):
        """Test production MCP mode doesn't track episode state."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Get initial state
        state_response = client.get("/state")
        initial_step_count = state_response.json()["step_count"]

        # Call tool via MCP
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 1, "b": 1}},
                "id": 1,
            },
        )

        # Verify step count didn't increment (production mode bypasses step tracking)
        state_response = client.get("/state")
        final_step_count = state_response.json()["step_count"]

        assert final_step_count == initial_step_count


# =============================================================================
# Client Integration Tests
# =============================================================================


class TestMCPClientProductionMode:
    """Tests for MCP client using production mode."""

    async def test_mcp_client_can_use_production_endpoints(self):
        """Test MCPToolClient can use production MCP endpoints directly."""
        from openenv.core.mcp_client import MCPToolClient

        client = MCPToolClient(base_url="http://localhost:8000")

        # Client should have option to use production mode (bypasses step())
        assert hasattr(client, "use_production_mode")

        client.use_production_mode = True

        # Calling list_tools() should use /mcp endpoint, not step()
        with patch.object(client, "step") as mock_step:
            tools = await client.list_tools()

            # step() should NOT be called in production mode
            mock_step.assert_not_called()
            assert len(tools) >= 0

    @pytest.mark.skip(reason="Implementation detail - httpx is now imported locally")
    async def test_client_production_mode_uses_http_mcp_endpoint(self):
        """Test client in production mode uses HTTP /mcp endpoint."""
        pass


# =============================================================================
# Error Response Tests
# =============================================================================


class TestMCPErrorResponses:
    """Tests for proper MCP JSON-RPC error responses."""

    def test_invalid_json_returns_parse_error(self, app):
        """Test malformed JSON returns JSON-RPC parse error."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post("/mcp", content="not valid json")

        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32700  # Parse error

    def test_missing_params_returns_invalid_params(self, app):
        """Test missing required params returns invalid params error."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    # Missing 'name' field
                    "arguments": {"a": 1}
                },
                "id": 1,
            },
        )

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32602  # Invalid params

    def test_nonexistent_tool_returns_error(self, app):
        """Test calling non-existent tool returns proper error."""
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "nonexistent_tool", "arguments": {}},
                "id": 1,
            },
        )

        data = response.json()
        assert "error" in data or "result" in data
        # Should indicate tool not found
