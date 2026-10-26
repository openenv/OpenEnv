"""Test fixture faults over the production OpenEnv WebSocket endpoint."""

import importlib.util
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from starlette.testclient import TestClient


FIXTURE = Path(__file__).parents[2] / "fixtures/validation/runtime/served_probe/app.py"
spec = importlib.util.spec_from_file_location("served_probe_app", FIXTURE)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def observation(ws, request):
    ws.send_json(request)
    result = ws.receive_json()
    assert result["type"] == "observation"
    return result["data"]


def test_real_session_retains_counter_and_requested_identity():
    with TestClient(fixture.make_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/schema").json()["observation"]["properties"]["counter"]
        with client.websocket_connect("/ws") as ws:
            initial = observation(
                ws, {"type": "reset", "data": {"episode_id": "test", "seed": 42}}
            )
            assert initial["observation"]["counter"] == 0
            for index in (1, 2):
                result = observation(ws, {"type": "step", "data": {"increment": 1}})
                assert result["observation"]["counter"] == index
                assert result["done"] is (index == 2)
                ws.send_json({"type": "state"})
                assert ws.receive_json()["data"] == {
                    "episode_id": "test",
                    "step_count": index,
                }
        with client.websocket_connect("/ws") as ws:
            result = observation(ws, {"type": "reset", "data": {"episode_id": "fresh"}})
            assert result["observation"]["counter"] == 0


@pytest.mark.parametrize(
    "mode", ["bad_reward", "bad_observation", "missing_done", "bad_state"]
)
def test_fault_is_visible_in_raw_wire_response(mode):
    with TestClient(fixture.make_app(mode)) as client:
        with client.websocket_connect("/ws") as ws:
            result = observation(ws, {"type": "reset", "data": {"episode_id": "test"}})
            if mode == "bad_reward":
                assert result["reward"] is True
            elif mode == "bad_observation":
                assert result["observation"]["counter"] == "not-an-integer"
            elif mode == "missing_done":
                assert "done" not in result
            else:
                ws.send_json({"type": "state"})
                assert ws.receive_json()["data"]["episode_id"] == "wrong-episode"


def test_failed_start_is_explicit():
    with pytest.raises(RuntimeError, match="Deliberate startup failure"):
        fixture.make_app("startup_failure")


@pytest.mark.parametrize("legacy_schema", [False, True])
def test_real_echo_reset_schema_and_unscored_rewards(legacy_schema, monkeypatch):
    # The lab deliberately clears PYTHONPATH and tests the installed core wheel.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "envs"))
    from echo_env.server.echo_environment import EchoEnvironment
    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Observation
    from openenv.validation.graders.runtime import (
        ObservationSchemaGrader,
        RewardWellFormedGrader,
        StateContractGrader,
    )
    from openenv.validation.runtime.collector import collect_runtime_evidence
    from openenv.validation.runtime.contracts import RuntimePlan

    app = create_fastapi_app(
        EchoEnvironment,
        CallToolAction,
        CallToolObservation,
        reset_observation_cls=Observation,
    )

    async def served_app(scope, receive, send):
        async def legacy_send(message):
            # Reproduce an older server's actual single-schema HTTP response.
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        (key, value)
                        for key, value in message["headers"]
                        if key != b"content-length"
                    ],
                }
            elif message["type"] == "http.response.body":
                schema = json.loads(message["body"])
                schema.pop("reset_observation")
                message = {**message, "body": json.dumps(schema).encode()}
            await send(message)

        await app(
            scope,
            receive,
            legacy_send if legacy_schema and scope.get("path") == "/schema" else send,
        )

    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "echo-contract", "seed": 42},
            "actions": [
                {
                    "tool_name": "echo_message",
                    "arguments": {"message": "real contract probe"},
                }
            ],
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(served_app, log_level="error", lifespan="on")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            evidence = collect_runtime_evidence(
                f"http://127.0.0.1:{port}", plan, episode_timeout_s=5
            )
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert evidence.failure_reason is None
    assert (evidence.reset_observation_schema_json is None) is legacy_schema
    step = next(row for row in evidence.exchanges if row.operation == "step")
    response = json.loads(step.response_json)["data"]
    assert response["reward"] is None and response["done"] is False
    assert "real contract probe" in json.dumps(response["observation"]["result"])
    subject = SimpleNamespace(
        runtime_evidence=evidence,
        manifest=SimpleNamespace(reward=SimpleNamespace(range=(0, 1))),
    )
    assert RewardWellFormedGrader().run(subject).status.value == "fail"
    assert StateContractGrader().run(subject).status.value == "pass"
    assert ObservationSchemaGrader().run(subject).status.value == (
        "fail" if legacy_schema else "pass"
    )
