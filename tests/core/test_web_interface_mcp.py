# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MCP action handling in the OpenEnv Gradio web interface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import State
from openenv.core.env_server.web_interface import create_web_interface_app

pytest.importorskip("gradio", reason="gradio is not installed")


class _FakeMCPState(State):
    """Minimal state for exercising the web wrapper."""

    step_count: int = 0


class FakeMCPEnvironment(Environment):
    """Minimal environment that accepts CallToolAction for web testing."""

    def __init__(self):
        super().__init__()
        self._state = _FakeMCPState()

    def reset(self) -> CallToolObservation:
        self._state = _FakeMCPState(step_count=0)
        return CallToolObservation(tool_name="init", result="ready")

    def step(self, action: CallToolAction) -> CallToolObservation:
        self._state.step_count += 1
        return CallToolObservation(
            tool_name=action.tool_name,
            result={
                "echoed_arguments": action.arguments,
                "is_dict": isinstance(action.arguments, dict),
            },
        )

    @property
    def state(self) -> _FakeMCPState:
        return self._state

    def close(self) -> None:
        pass


def test_web_step_call_tool_parses_json_string_arguments() -> None:
    """POST /web/step should decode JSON string arguments for CallToolAction."""
    app = create_web_interface_app(
        FakeMCPEnvironment,
        CallToolAction,
        CallToolObservation,
    )
    client = TestClient(app)

    reset_response = client.post("/web/reset")
    assert reset_response.status_code == 200

    step_response = client.post(
        "/web/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": "echo",
                "arguments": '{"message": "hello"}',
            }
        },
    )
    assert step_response.status_code == 200
    step_json = step_response.json()
    result = step_json["observation"]["result"]
    assert result["echoed_arguments"] == {"message": "hello"}
    assert result["is_dict"] is True
