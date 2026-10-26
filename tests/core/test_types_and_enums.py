# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for enums and Pydantic models in OpenEnv env_server.

This file tests the type-safe enums and JSON-RPC models added for
FastAPI/Pydantic best practices compliance.

Test coverage:
- ServerMode enum values and string conversion
- HealthStatus enum values and serialization
- WSErrorCode enum values
- JsonRpcErrorCode standard error codes
- McpMethod enum values
- JsonRpcRequest validation
- JsonRpcResponse serialization (result XOR error compliance)
- JsonRpcError factory methods
"""

import json

import pytest
from openenv.core.env_server.mcp_types import (
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    McpMethod,
)
from openenv.core.env_server.types import (
    HealthResponse,
    HealthStatus,
    ServerMode,
    WSErrorCode,
    WSErrorResponse,
)
from pydantic import ValidationError


# =============================================================================
# ServerMode Enum Tests
# =============================================================================


class TestServerModeEnum:
    """Tests for ServerMode enum."""

    def test_server_mode_values(self):
        """Test ServerMode enum has expected values."""
        assert ServerMode.SIMULATION.value == "simulation"
        assert ServerMode.PRODUCTION.value == "production"

    def test_server_mode_from_string(self):
        """Test ServerMode can be created from string."""
        assert ServerMode("simulation") == ServerMode.SIMULATION
        assert ServerMode("production") == ServerMode.PRODUCTION

    def test_server_mode_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ServerMode("invalid")

    def test_server_mode_is_str_subclass(self):
        """Test ServerMode values work as strings for comparison."""
        # Equality comparison works with strings
        assert ServerMode.SIMULATION == "simulation"
        assert ServerMode.PRODUCTION == "production"

        # For f-strings, use .value to get the string
        assert f"Mode: {ServerMode.SIMULATION.value}" == "Mode: simulation"

        # str() also gives the enum representation, not value
        # Use .value when you need the raw string
        assert ServerMode.SIMULATION.value == "simulation"

    def test_server_mode_case_sensitive(self):
        """Test ServerMode is case-sensitive (lowercase required)."""
        with pytest.raises(ValueError):
            ServerMode("SIMULATION")
        with pytest.raises(ValueError):
            ServerMode("Simulation")


# =============================================================================
# HealthStatus Enum Tests
# =============================================================================


class TestHealthStatusEnum:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum has expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_health_response_serialization(self):
        """Test HealthResponse serializes status enum correctly."""
        response = HealthResponse(status=HealthStatus.HEALTHY)
        data = response.model_dump()

        assert data["status"] == "healthy"

    def test_health_response_json_serialization(self):
        """Test HealthResponse JSON serialization."""
        response = HealthResponse(status=HealthStatus.DEGRADED)
        json_str = response.model_dump_json()
        data = json.loads(json_str)

        assert data["status"] == "degraded"


# =============================================================================
# WSErrorCode Enum Tests
# =============================================================================


class TestWSErrorCodeEnum:
    """Tests for WSErrorCode enum."""

    def test_ws_error_code_values(self):
        """Test WSErrorCode enum has expected values."""
        assert WSErrorCode.INVALID_JSON.value == "INVALID_JSON"
        assert WSErrorCode.UNKNOWN_TYPE.value == "UNKNOWN_TYPE"
        assert WSErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
        assert WSErrorCode.EXECUTION_ERROR.value == "EXECUTION_ERROR"
        assert WSErrorCode.CAPACITY_REACHED.value == "CAPACITY_REACHED"
        assert WSErrorCode.FACTORY_ERROR.value == "FACTORY_ERROR"
        assert WSErrorCode.SESSION_ERROR.value == "SESSION_ERROR"

    def test_ws_error_response_with_enum(self):
        """Test WSErrorResponse correctly serializes enum code."""
        response = WSErrorResponse(
            data={
                "message": "Test error",
                "code": WSErrorCode.INVALID_JSON,
            }
        )
        data = response.model_dump()

        assert data["type"] == "error"
        assert data["data"]["code"] == "INVALID_JSON"


# =============================================================================
# JsonRpcErrorCode Enum Tests
# =============================================================================


class TestJsonRpcErrorCodeEnum:
    """Tests for JsonRpcErrorCode enum with standard JSON-RPC 2.0 codes."""

    def test_standard_error_codes(self):
        """Test standard JSON-RPC 2.0 error codes are correct."""
        # Per https://www.jsonrpc.org/specification#error_object
        assert JsonRpcErrorCode.PARSE_ERROR.value == -32700
        assert JsonRpcErrorCode.INVALID_REQUEST.value == -32600
        assert JsonRpcErrorCode.METHOD_NOT_FOUND.value == -32601
        assert JsonRpcErrorCode.INVALID_PARAMS.value == -32602
        assert JsonRpcErrorCode.INTERNAL_ERROR.value == -32603
        assert JsonRpcErrorCode.SERVER_ERROR.value == -32000

    def test_error_codes_are_negative(self):
        """Test all JSON-RPC error codes are negative integers."""
        for code in JsonRpcErrorCode:
            assert code.value < 0


# =============================================================================
# McpMethod Enum Tests
# =============================================================================


class TestMcpMethodEnum:
    """Tests for McpMethod enum."""

    def test_mcp_method_values(self):
        """Test McpMethod enum has expected MCP method names."""
        assert McpMethod.TOOLS_LIST.value == "tools/list"
        assert McpMethod.TOOLS_CALL.value == "tools/call"

    def test_mcp_method_string_comparison(self):
        """Test McpMethod values work as strings for comparison."""
        assert McpMethod.TOOLS_LIST == "tools/list"
        assert McpMethod.TOOLS_CALL == "tools/call"


# =============================================================================
# JsonRpcError Model Tests
# =============================================================================


class TestJsonRpcError:
    """Tests for JsonRpcError Pydantic model."""

    def test_error_creation(self):
        """Test basic JsonRpcError creation."""
        error = JsonRpcError(code=-32600, message="Invalid Request")

        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert error.data is None

    def test_error_with_data(self):
        """Test JsonRpcError with additional data."""
        error = JsonRpcError(
            code=-32602, message="Invalid params", data={"field": "name"}
        )

        assert error.data == {"field": "name"}

    def test_from_code_factory(self):
        """Test JsonRpcError.from_code factory method."""
        error = JsonRpcError.from_code(JsonRpcErrorCode.PARSE_ERROR)

        assert error.code == -32700
        assert error.message == "Parse error"

    def test_from_code_with_custom_message(self):
        """Test from_code with custom message."""
        error = JsonRpcError.from_code(
            JsonRpcErrorCode.INTERNAL_ERROR, message="Custom error message"
        )

        assert error.code == -32603
        assert error.message == "Custom error message"

    def test_from_code_with_data(self):
        """Test from_code with additional data."""
        error = JsonRpcError.from_code(
            JsonRpcErrorCode.INVALID_PARAMS, data={"missing": ["name", "args"]}
        )

        assert error.data == {"missing": ["name", "args"]}


# =============================================================================
# JsonRpcRequest Model Tests
# =============================================================================


class TestJsonRpcRequest:
    """Tests for JsonRpcRequest Pydantic model."""

    def test_valid_request(self):
        """Test valid JSON-RPC request parsing."""
        request = JsonRpcRequest(jsonrpc="2.0", method="tools/list", id=1)

        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.id == 1
        assert request.params == {}

    def test_request_with_params(self):
        """Test request with params."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="tools/call",
            params={"name": "my_tool", "arguments": {"x": 1}},
            id="req-123",
        )

        assert request.params["name"] == "my_tool"
        assert request.id == "req-123"

    def test_request_requires_jsonrpc_2_0(self):
        """Test request must have jsonrpc='2.0'."""
        with pytest.raises(ValidationError):
            JsonRpcRequest(jsonrpc="1.0", method="test", id=1)

    def test_request_requires_method(self):
        """Test request must have method."""
        with pytest.raises(ValidationError):
            JsonRpcRequest(jsonrpc="2.0", id=1)

    def test_request_id_can_be_string_or_int(self):
        """Test request ID can be string or integer."""
        req1 = JsonRpcRequest(jsonrpc="2.0", method="test", id=42)
        req2 = JsonRpcRequest(jsonrpc="2.0", method="test", id="string-id")

        assert req1.id == 42
        assert req2.id == "string-id"

    def test_request_id_can_be_none(self):
        """Test request ID can be None (notification)."""
        request = JsonRpcRequest(jsonrpc="2.0", method="test")

        assert request.id is None


# =============================================================================
# JsonRpcResponse Model Tests
# =============================================================================


class TestJsonRpcResponse:
    """Tests for JsonRpcResponse Pydantic model."""

    def test_success_response(self):
        """Test success response creation."""
        response = JsonRpcResponse.success(result={"tools": []}, request_id=1)

        assert response.result == {"tools": []}
        assert response.error is None
        assert response.id == 1

    def test_error_response(self):
        """Test error response creation."""
        response = JsonRpcResponse.error_response(
            JsonRpcErrorCode.METHOD_NOT_FOUND,
            message="Method not found: foo",
            request_id=2,
        )

        assert response.result is None
        assert response.error is not None
        assert response.error.code == -32601
        assert response.id == 2

    def test_model_dump_excludes_result_on_error(self):
        """Test model_dump excludes 'result' when there's an error (JSON-RPC compliance)."""
        response = JsonRpcResponse.error_response(
            JsonRpcErrorCode.PARSE_ERROR, request_id=3
        )
        data = response.model_dump()

        assert "error" in data
        assert "result" not in data
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 3

    def test_model_dump_excludes_error_on_success(self):
        """Test model_dump excludes 'error' when there's a result (JSON-RPC compliance)."""
        response = JsonRpcResponse.success(result="ok", request_id=4)
        data = response.model_dump()

        assert "result" in data
        assert "error" not in data
        assert data["result"] == "ok"

    def test_model_dump_json(self):
        """Test model_dump_json produces valid JSON."""
        response = JsonRpcResponse.success(result={"value": 42}, request_id=5)
        json_str = response.model_dump_json()
        data = json.loads(json_str)

        assert data["jsonrpc"] == "2.0"
        assert data["result"] == {"value": 42}
        assert data["id"] == 5
        assert "error" not in data

    def test_success_with_null_result(self):
        """Test success response with null result is still valid."""
        response = JsonRpcResponse.success(result=None, request_id=6)
        data = response.model_dump()

        # Per JSON-RPC spec, result can be null for success
        assert "result" in data
        assert data["result"] is None
        assert "error" not in data

    def test_response_preserves_string_id(self):
        """Test response preserves string request ID."""
        response = JsonRpcResponse.success(result={}, request_id="test-uuid-123")
        data = response.model_dump()

        assert data["id"] == "test-uuid-123"

    def test_response_with_none_id(self):
        """Test response with None ID (notification response)."""
        response = JsonRpcResponse.success(result={}, request_id=None)
        data = response.model_dump()

        assert data["id"] is None


# =============================================================================
# Integration Tests: Enums with HTTP Server
# =============================================================================


class TestEnumIntegrationWithHTTPServer:
    """Tests for enum integration with HTTPEnvServer."""

    def test_register_routes_accepts_enum(self):
        """Test register_routes accepts ServerMode enum."""
        from fastapi import FastAPI
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Action, Observation, State

        class TestAction(Action):
            message: str

        class TestObservation(Observation):
            response: str

        class TestEnvironment(Environment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def reset(self, **kwargs):
                return TestObservation(response="reset", done=False)

            def step(self, action):
                return TestObservation(response="step", done=False)

            @property
            def state(self):
                return State(step_count=0)

            def close(self):
                pass

        app = FastAPI()
        server = HTTPEnvServer(TestEnvironment, TestAction, TestObservation)

        # Should work with enum
        server.register_routes(app, mode=ServerMode.SIMULATION)

        # Verify routes are registered
        routes = [route.path for route in app.routes]
        assert "/reset" in routes
        assert "/step" in routes

    def test_register_routes_accepts_string(self):
        """Test register_routes still accepts string (backwards compatibility)."""
        from fastapi import FastAPI
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Action, Observation, State

        class TestAction(Action):
            message: str

        class TestObservation(Observation):
            response: str

        class TestEnvironment(Environment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def reset(self, **kwargs):
                return TestObservation(response="reset", done=False)

            def step(self, action):
                return TestObservation(response="step", done=False)

            @property
            def state(self):
                return State(step_count=0)

            def close(self):
                pass

        app = FastAPI()
        server = HTTPEnvServer(TestEnvironment, TestAction, TestObservation)

        # Should work with string
        server.register_routes(app, mode="production")

        # Verify simulation routes are NOT registered in production mode
        routes = [route.path for route in app.routes]
        assert "/reset" not in routes
        assert "/step" not in routes
        assert "/health" in routes

    def test_health_endpoint_returns_enum_value(self):
        """Test /health endpoint returns correct enum value as string."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from openenv.core.env_server.http_server import HTTPEnvServer
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Action, Observation, State

        class TestAction(Action):
            message: str

        class TestObservation(Observation):
            response: str

        class TestEnvironment(Environment):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def reset(self, **kwargs):
                return TestObservation(response="reset", done=False)

            def step(self, action):
                return TestObservation(response="step", done=False)

            @property
            def state(self):
                return State(step_count=0)

            def close(self):
                pass

        app = FastAPI()
        server = HTTPEnvServer(TestEnvironment, TestAction, TestObservation)
        server.register_routes(app)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
