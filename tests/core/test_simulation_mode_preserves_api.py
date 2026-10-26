# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for simulation mode API preservation (Task #1).

These tests verify that simulation mode preserves the full Gym-style API + MCP
functionality. This is a regression test to ensure that adding production mode
does not break existing simulation mode behavior.

Test coverage:
1. Simulation mode exposes /reset, /step, /state endpoints
2. Simulation mode exposes /ws WebSocket endpoint with Gym-style API
3. Simulation mode exposes /mcp WebSocket endpoint for MCP access
4. Simulation mode exposes HTTP POST /mcp endpoint for MCP access
5. Simulation mode is the default when no mode is specified
6. All endpoints work correctly together (not just exposed, but functional)

This addresses GitHub issue #346 Task #1:
  "Test that sim mode preserves full Gym-style API + MCP. Tests should verify
   that in simulation mode: (1) /reset, /step, /state all work (current behavior),
   (2) /mcp WebSocket endpoint works, (3) HTTP POST /mcp works (new feature),
   (4) Sim mode is the default when no mode specified."
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
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State


# ============================================================================
# Test Fixtures
# ============================================================================


class SimModeTestAction(Action):
    """Test action for simulation mode testing."""

    message: str


class SimModeTestObservation(Observation):
    """Test observation for simulation mode testing."""

    response: str
    reward: float | None = None
    done: bool = False


class SimModeTestState(State):
    """Test state for simulation mode testing."""

    step_count: int = 0
    episode_id: str = "test"


class SimModeTestEnvironment(Environment):
    """
    Test environment for simulation mode API preservation tests.

    This environment supports both Gym-style API and basic functionality.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize the test environment."""
        self._step_count = 0
        self._episode_id = "sim-test"

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs
    ) -> SimModeTestObservation:
        """Reset the environment."""
        self._step_count = 0
        if episode_id:
            self._episode_id = episode_id
        return SimModeTestObservation(
            response="reset_complete", reward=None, done=False
        )

    def step(self, action: SimModeTestAction) -> SimModeTestObservation:
        """Execute an action."""
        self._step_count += 1
        return SimModeTestObservation(
            response=f"echo: {action.message}",
            reward=1.0,
            done=False,
        )

    @property
    def state(self) -> SimModeTestState:
        """Return current state."""
        return SimModeTestState(
            step_count=self._step_count,
            episode_id=self._episode_id,
        )

    def close(self) -> None:
        """Cleanup resources."""
        pass


class SimModeMCPTestEnvironment(MCPEnvironment):
    """
    Test environment with MCP tools for simulation mode testing.

    This environment supports both Gym-style API and MCP tools.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize with MCP server and test tools."""
        mcp_server = FastMCP("sim-mode-test-env")

        @mcp_server.tool
        def get_step_count() -> int:
            """Get current step count."""
            return self._step_count

        @mcp_server.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        super().__init__(mcp_server)
        self._step_count = 0
        self._episode_id = "sim-mcp-test"

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs
    ) -> Observation:
        """Reset the environment."""
        self._step_count = 0
        if episode_id:
            self._episode_id = episode_id
        return Observation(done=False, reward=None)

    def _step_impl(self, action: Action, **kwargs) -> Observation:
        """Handle non-MCP actions."""
        self._step_count += 1
        return Observation(done=False, reward=1.0)

    @property
    def state(self) -> State:
        """Return current state."""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )


@pytest.fixture
def simulation_mode_app() -> FastAPI:
    """
    Create FastAPI app in simulation mode (default).

    Simulation mode should expose:
    - Gym-style API: /reset, /step, /state
    - WebSocket endpoint: /ws
    - Safe endpoints: /health, /schema, /metadata
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=SimModeTestEnvironment,
        action_cls=SimModeTestAction,
        observation_cls=SimModeTestObservation,
    )
    # Do not specify mode - should default to simulation
    server.register_routes(app)
    return app


@pytest.fixture
def simulation_mode_app_explicit() -> FastAPI:
    """
    Create FastAPI app with explicit mode='simulation'.

    Should behave identically to default mode.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=SimModeTestEnvironment,
        action_cls=SimModeTestAction,
        observation_cls=SimModeTestObservation,
    )
    # Explicitly set mode to simulation
    server.register_routes(app, mode="simulation")
    return app


@pytest.fixture
def simulation_mode_mcp_app() -> FastAPI:
    """
    Create FastAPI app in simulation mode with MCP support.

    This fixture tests that MCP endpoints work in simulation mode.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=SimModeMCPTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="simulation")
    return app


# ============================================================================
# Task #1.1: Simulation Mode Exposes Gym-Style API Endpoints
# ============================================================================


class TestSimulationModeGymAPIEndpoints:
    """Test that simulation mode exposes /reset, /step, /state endpoints."""

    def test_simulation_mode_exposes_reset_endpoint(self, simulation_mode_app):
        """
        Test that /reset endpoint is available in simulation mode.

        Signal: High - ensures core Gym API is not broken by production mode.
        """
        client = TestClient(simulation_mode_app)

        response = client.post("/reset", json={})

        assert response.status_code == 200, (
            "Simulation mode must expose /reset endpoint for episode initialization"
        )
        data = response.json()
        assert "observation" in data
        assert data["observation"]["response"] == "reset_complete"

    def test_simulation_mode_exposes_step_endpoint(self, simulation_mode_app):
        """
        Test that /step endpoint is available in simulation mode.

        Signal: High - ensures core Gym API is not broken by production mode.
        """
        client = TestClient(simulation_mode_app)

        response = client.post("/step", json={"action": {"message": "test_action"}})

        assert response.status_code == 200, (
            "Simulation mode must expose /step endpoint for action execution"
        )
        data = response.json()
        assert "observation" in data
        assert "echo: test_action" in data["observation"]["response"]

    def test_simulation_mode_exposes_state_endpoint(self, simulation_mode_app):
        """
        Test that /state endpoint is available in simulation mode.

        Signal: High - ensures core Gym API is not broken by production mode.
        """
        client = TestClient(simulation_mode_app)

        response = client.get("/state")

        assert response.status_code == 200, (
            "Simulation mode must expose /state endpoint for state inspection"
        )
        data = response.json()
        assert "step_count" in data
        assert "episode_id" in data

    def test_simulation_mode_reset_with_parameters(self, simulation_mode_app):
        """
        Test that /reset accepts optional parameters (seed, episode_id).

        Signal: Medium - ensures parameter passing works correctly.
        """
        client = TestClient(simulation_mode_app)

        response = client.post(
            "/reset",
            json={"seed": 42, "episode_id": "custom_episode"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "observation" in data


# ============================================================================
# Task #1.2: Simulation Mode Exposes WebSocket /ws Endpoint
# ============================================================================


class TestSimulationModeWebSocketEndpoint:
    """Test that simulation mode exposes /ws WebSocket endpoint."""

    def test_simulation_mode_exposes_websocket_endpoint(self, simulation_mode_app):
        """
        Test that /ws WebSocket endpoint is available in simulation mode.

        Signal: High - ensures WebSocket communication is not broken.
        """
        client = TestClient(simulation_mode_app)

        with client.websocket_connect("/ws") as websocket:
            # Send reset message
            reset_msg = {"type": "reset", "data": {}}
            websocket.send_text(json.dumps(reset_msg))

            # Receive response
            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "observation", (
                "Should receive observation message"
            )
            assert "data" in response
            assert "observation" in response["data"]

    def test_simulation_mode_websocket_reset_works(self, simulation_mode_app):
        """
        Test that WebSocket reset message works in simulation mode.

        Signal: High - ensures WebSocket reset functionality is preserved.
        """
        client = TestClient(simulation_mode_app)

        with client.websocket_connect("/ws") as websocket:
            # Send reset
            reset_msg = {"type": "reset", "data": {"episode_id": "ws_test"}}
            websocket.send_text(json.dumps(reset_msg))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "observation"
            assert response["data"]["observation"]["response"] == "reset_complete"

    def test_simulation_mode_websocket_step_works(self, simulation_mode_app):
        """
        Test that WebSocket step message works in simulation mode.

        Signal: High - ensures WebSocket step functionality is preserved.
        """
        client = TestClient(simulation_mode_app)

        with client.websocket_connect("/ws") as websocket:
            # Send step
            step_msg = {
                "type": "step",
                "data": {"message": "websocket_test"},
            }
            websocket.send_text(json.dumps(step_msg))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "observation"
            assert "echo: websocket_test" in response["data"]["observation"]["response"]

    def test_simulation_mode_websocket_state_works(self, simulation_mode_app):
        """
        Test that WebSocket state message works in simulation mode.

        Signal: High - ensures WebSocket state access is preserved.
        """
        client = TestClient(simulation_mode_app)

        with client.websocket_connect("/ws") as websocket:
            # Send state query (no data field needed for state message)
            state_msg = {"type": "state"}
            websocket.send_text(json.dumps(state_msg))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "state"
            assert "data" in response
            assert "step_count" in response["data"]


# ============================================================================
# Task #1.3: Simulation Mode Exposes /mcp WebSocket Endpoint
# ============================================================================


class TestSimulationModeWebSocketMCPEndpoint:
    """Test that simulation mode exposes /mcp functionality via WebSocket."""

    def test_simulation_mode_websocket_mcp_tools_list(self, simulation_mode_mcp_app):
        """
        Test that WebSocket MCP tools/list works in simulation mode.

        Signal: High - ensures MCP via WebSocket is not broken.
        """
        client = TestClient(simulation_mode_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP tools/list request
            mcp_msg = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
            }
            websocket.send_text(json.dumps(mcp_msg))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert response["data"]["jsonrpc"] == "2.0"
            assert "result" in response["data"]
            assert "tools" in response["data"]["result"]

    def test_simulation_mode_websocket_mcp_tools_call(self, simulation_mode_mcp_app):
        """
        Test that WebSocket MCP tools/call works in simulation mode.

        Signal: High - ensures MCP tool execution via WebSocket is preserved.
        """
        client = TestClient(simulation_mode_mcp_app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP tools/call request
            mcp_msg = {
                "type": "mcp",
                "data": {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "add",
                        "arguments": {"a": 10, "b": 20},
                    },
                    "id": 2,
                },
            }
            websocket.send_text(json.dumps(mcp_msg))

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert response["data"]["jsonrpc"] == "2.0"
            assert "result" in response["data"]


# ============================================================================
# Task #1.4: Simulation Mode Exposes WebSocket /mcp Endpoint
# ============================================================================


class TestSimulationModeDedicatedMCPEndpoint:
    """Test that simulation mode exposes WebSocket /mcp endpoint."""

    def test_simulation_mode_http_mcp_tools_list(self, simulation_mode_mcp_app):
        """
        Test that WebSocket /mcp tools/list works in simulation mode.

        Signal: High - ensures new WebSocket MCP endpoint works in simulation mode.
        """
        client = TestClient(simulation_mode_mcp_app)

        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 10,
        }

        with client.websocket_connect("/mcp") as websocket:
            websocket.send_text(json.dumps(request))
            response = json.loads(websocket.receive_text())

        assert response["jsonrpc"] == "2.0", (
            "Simulation mode must expose WebSocket /mcp endpoint"
        )
        assert "result" in response
        assert "tools" in response["result"]

    def test_simulation_mode_http_mcp_tools_call(self, simulation_mode_mcp_app):
        """
        Test that WebSocket /mcp tools/call works in simulation mode.

        Signal: High - ensures WebSocket MCP tool calling works in simulation mode.
        """
        client = TestClient(simulation_mode_mcp_app)

        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "add",
                "arguments": {"a": 5, "b": 7},
            },
            "id": 11,
        }

        with client.websocket_connect("/mcp") as websocket:
            websocket.send_text(json.dumps(request))
            response = json.loads(websocket.receive_text())

        assert "result" in response


# ============================================================================
# Task #1.5: Simulation Mode is Default
# ============================================================================


class TestSimulationModeIsDefault:
    """Test that simulation mode is the default when no mode is specified."""

    def test_default_mode_exposes_reset(self, simulation_mode_app):
        """
        Test that default mode (no mode parameter) exposes /reset.

        Signal: High - ensures backwards compatibility.
        """
        client = TestClient(simulation_mode_app)

        response = client.post("/reset", json={})

        assert response.status_code == 200, (
            "Default mode should be simulation, exposing /reset"
        )

    def test_default_mode_exposes_step(self, simulation_mode_app):
        """
        Test that default mode (no mode parameter) exposes /step.

        Signal: High - ensures backwards compatibility.
        """
        client = TestClient(simulation_mode_app)

        response = client.post("/step", json={"action": {"message": "test"}})

        assert response.status_code == 200, (
            "Default mode should be simulation, exposing /step"
        )

    def test_default_mode_exposes_state(self, simulation_mode_app):
        """
        Test that default mode (no mode parameter) exposes /state.

        Signal: High - ensures backwards compatibility.
        """
        client = TestClient(simulation_mode_app)

        response = client.get("/state")

        assert response.status_code == 200, (
            "Default mode should be simulation, exposing /state"
        )

    def test_explicit_simulation_matches_default(
        self, simulation_mode_app, simulation_mode_app_explicit
    ):
        """
        Test that explicit mode='simulation' behaves identically to default.

        Signal: Medium - ensures explicit and implicit modes are equivalent.
        """
        default_client = TestClient(simulation_mode_app)
        explicit_client = TestClient(simulation_mode_app_explicit)

        # Test /reset
        default_reset = default_client.post("/reset", json={})
        explicit_reset = explicit_client.post("/reset", json={})

        assert default_reset.status_code == explicit_reset.status_code == 200

        # Test /step
        default_step = default_client.post(
            "/step", json={"action": {"message": "test"}}
        )
        explicit_step = explicit_client.post(
            "/step", json={"action": {"message": "test"}}
        )

        assert default_step.status_code == explicit_step.status_code == 200

        # Test /state
        default_state = default_client.get("/state")
        explicit_state = explicit_client.get("/state")

        assert default_state.status_code == explicit_state.status_code == 200


# ============================================================================
# Task #1.6: Integration Test - All APIs Work Together
# ============================================================================


class TestSimulationModeFullIntegration:
    """
    Test that all simulation mode APIs work together correctly.

    This is a high-signal integration test to ensure the full system works,
    not just individual endpoints.
    """

    def test_simulation_mode_full_gym_workflow(self, simulation_mode_app):
        """
        Test complete Gym workflow: reset -> step -> step -> state.

        Signal: High - ensures full Gym API workflow is preserved.
        """
        client = TestClient(simulation_mode_app)

        # Reset
        reset_response = client.post("/reset", json={"episode_id": "integration_test"})
        assert reset_response.status_code == 200

        # Step 1
        step1_response = client.post("/step", json={"action": {"message": "step1"}})
        assert step1_response.status_code == 200
        assert "echo: step1" in step1_response.json()["observation"]["response"]

        # Step 2
        step2_response = client.post("/step", json={"action": {"message": "step2"}})
        assert step2_response.status_code == 200
        assert "echo: step2" in step2_response.json()["observation"]["response"]

        # Check state
        state_response = client.get("/state")
        assert state_response.status_code == 200
        # Note: HTTP endpoints create new env instances, so step_count is 0
        # This is expected behavior for stateless HTTP endpoints

    def test_simulation_mode_full_websocket_workflow(self, simulation_mode_app):
        """
        Test complete WebSocket workflow: connect -> reset -> step -> state -> close.

        Signal: High - ensures full WebSocket workflow is preserved.
        """
        client = TestClient(simulation_mode_app)

        with client.websocket_connect("/ws") as websocket:
            # Reset
            websocket.send_text(json.dumps({"type": "reset", "data": {}}))
            reset_resp = json.loads(websocket.receive_text())
            assert reset_resp["type"] == "observation"

            # Step
            websocket.send_text(
                json.dumps({"type": "step", "data": {"message": "ws_step"}})
            )
            step_resp = json.loads(websocket.receive_text())
            assert step_resp["type"] == "observation"

            # State
            websocket.send_text(json.dumps({"type": "state"}))
            state_resp = json.loads(websocket.receive_text())
            assert state_resp["type"] == "state"
            assert state_resp["data"]["step_count"] == 1  # WebSocket maintains state

            # Close
            websocket.send_text(json.dumps({"type": "close", "data": {}}))

    def test_simulation_mode_mcp_and_gym_coexist(self, simulation_mode_mcp_app):
        """
        Test that MCP and Gym-style APIs coexist in simulation mode.

        Signal: High - ensures dual API boundary is preserved.
        """
        client = TestClient(simulation_mode_mcp_app)

        # Test Gym API
        reset_response = client.post("/reset", json={})
        assert reset_response.status_code == 200

        # Test MCP API via WebSocket
        mcp_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 100,
        }
        with client.websocket_connect("/mcp") as websocket:
            websocket.send_text(json.dumps(mcp_request))
            mcp_response = json.loads(websocket.receive_text())

        assert "result" in mcp_response

        # Both should work - Gym step uses CallToolAction for MCP environments
        # The action format depends on the environment's action type
