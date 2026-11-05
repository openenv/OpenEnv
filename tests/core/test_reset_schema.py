# SPDX-License-Identifier: BSD-3-Clause

"""Tests that `/schema` publishes an environment's reset parameters without enforcing them."""

from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import (
    create_app,
    create_fastapi_app,
    HTTPEnvServer,
)
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, ResetRequest, State
from pydantic import Field


class EchoAction(Action):
    """Action for the fixture environment."""

    message: str = ""


class EchoObservation(Observation):
    """Observation for the fixture environment."""

    response: str = ""


class TaskResetRequest(ResetRequest):
    """Reset parameters declared by the fixture environment."""

    task_id: int | None = Field(default=None, description="Task to load")


class RecordingEnvironment(Environment[EchoAction, EchoObservation, State]):
    """Environment that records the kwargs each `reset()` call receives.

    The HTTP `/reset` handler builds a fresh environment per request, so the
    record lives on the class.
    """

    reset_calls: list[dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Any = None,
        **kwargs: Any,
    ) -> EchoObservation:
        RecordingEnvironment.reset_calls.append(
            {"seed": seed, "episode_id": episode_id, "task_id": task_id, **kwargs}
        )
        return EchoObservation(response="")

    def step(self, action: EchoAction) -> EchoObservation:
        return EchoObservation(response=action.message)

    @property
    def state(self) -> State:
        return State()


@pytest.fixture(autouse=True)
def clear_reset_calls():
    RecordingEnvironment.reset_calls.clear()


@pytest.fixture
def declared_client() -> TestClient:
    """Client for an app that declares its `ResetRequest` subclass."""
    app = create_fastapi_app(
        RecordingEnvironment,
        EchoAction,
        EchoObservation,
        env_name="echo_env",
        reset_cls=TaskResetRequest,
    )
    return TestClient(app)


@pytest.fixture
def default_client() -> TestClient:
    """Client for an app that declares no `ResetRequest` subclass."""
    app = create_fastapi_app(
        RecordingEnvironment,
        EchoAction,
        EchoObservation,
        env_name="echo_env",
    )
    return TestClient(app)


class TestDeclaredResetClass:
    """An environment that passes `reset_cls` gets it published on `/schema`."""

    @pytest.mark.parametrize("web_enabled", [False, True])
    def test_create_app_publishes_reset_schema_in_both_modes(
        self, monkeypatch, web_enabled
    ):
        monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true" if web_enabled else "false")
        app = create_app(
            RecordingEnvironment,
            EchoAction,
            EchoObservation,
            env_name="echo_env",
            reset_cls=TaskResetRequest,
        )
        client = TestClient(app)

        assert "task_id" in client.get("/schema").json()["reset"]["properties"]

    def test_schema_publishes_subclass_and_base_fields(self, declared_client):
        response = declared_client.get("/schema")

        assert response.status_code == 200
        properties = response.json()["reset"]["properties"]
        assert set(properties) == {"seed", "episode_id", "task_id"}

    def test_other_schemas_are_unaffected(self, declared_client):
        payload = declared_client.get("/schema").json()

        assert "message" in payload["action"]["properties"]
        assert "response" in payload["observation"]["properties"]
        assert set(payload["state"]["properties"]) == {"episode_id", "step_count"}

    def test_reset_is_not_validated_against_reset_cls(self, declared_client):
        response = declared_client.post(
            "/reset", json={"task_id": "not-an-int", "extra": "x"}
        )

        assert response.status_code == 200
        [received] = RecordingEnvironment.reset_calls
        assert received["task_id"] == "not-an-int"
        assert received["extra"] == "x"

    def test_websocket_reset_is_not_validated_against_reset_cls(self, declared_client):
        with declared_client.websocket_connect("/ws") as ws:
            ws.send_json(
                {"type": "reset", "data": {"task_id": "not-an-int", "extra": "x"}}
            )
            response = ws.receive_json()
        assert response["type"] == "observation"
        [received] = RecordingEnvironment.reset_calls
        assert received["task_id"] == "not-an-int"
        assert received["extra"] == "x"


class TestDefaultResetClass:
    """Omitting `reset_cls` publishes the base `ResetRequest` schema."""

    def test_schema_falls_back_to_base_reset_request(self, default_client):
        reset_schema = default_client.get("/schema").json()["reset"]

        assert reset_schema == ResetRequest.model_json_schema()

    def test_server_defaults_to_base_reset_request(self):
        server = HTTPEnvServer(RecordingEnvironment, EchoAction, EchoObservation)

        assert server.reset_cls is ResetRequest
