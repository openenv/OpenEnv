"""Exercise the real replay WebSocket and its separate production MCP boundary."""

import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import FastMCP
from openenv.core.env_server.http_server import create_app, HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Rubric, WeightedSum
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan

TOKEN = "per-run-validation-test-token-000000000000"


class ValueAction(Action):
    value: float


class ValueScore(Rubric):
    def forward(self, action, observation):
        return action.value

    def validation_config(self):
        return {}


class SessionEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__(rubric=WeightedSum([ValueScore(), ValueScore()], [0.25, 0.75]))
        self.count = 0
        self.seed = None
        self.mcp_server = FastMCP("telemetry-test")

    def reset(self, seed=None):
        self.seed, self.count = seed, 0
        return Observation(reward=0.0, metadata={"seed": seed})

    def step(self, action):
        self.count += 1
        observation = Observation(metadata={"count": self.count})
        observation.reward = self.rubric(action, observation)
        return observation

    @property
    def state(self):
        return State(episode_id="same-session", step_count=self.count)


class DropsSeed(SessionEnv):
    def reset(self):
        return super().reset()


class KwargSeed(SessionEnv):
    def reset(self, **kwargs):
        return super().reset(seed=kwargs.get("seed"))


class AsyncSeed(SessionEnv):
    async def reset_async(self, seed=None):
        return super().reset(seed=seed)


class AsyncDropsSeed(SessionEnv):
    async def reset_async(self):
        return super().reset()


@pytest.mark.parametrize("closed_before_send", [False, True])
def test_factory_error_survives_real_telemetry_handshake(
    monkeypatch, closed_before_send
):
    class BrokenFactory(SessionEnv):
        def __init__(self):
            # Fail before base initialization to exercise factory errors.
            raise RuntimeError(TOKEN)

    monkeypatch.setenv("OPENENV_VALIDATION_TOKEN", TOKEN)
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    app = create_app(BrokenFactory, ValueAction, Observation)
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.01)
            if closed_before_send:
                real_connect = collector.connect

                def connect_after_close(*args, **kwargs):
                    connection = real_connect(*args, **kwargs)
                    deadline = time.monotonic() + 2
                    while connection.close_code is None:
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    return connection

                monkeypatch.setattr(collector, "connect", connect_after_close)
            plan = RuntimePlan.model_validate(
                {
                    "plan_schema_version": "1",
                    "reset": {"episode_id": "factory-failure", "seed": 1},
                    "actions": [{"value": 0.5}],
                }
            )
            evidence = collector.collect_runtime_evidence(
                f"http://127.0.0.1:{port}",
                plan,
                episode_timeout_s=3,
                validation_token=TOKEN,
            )
            assert evidence.failure_reason == "validation_open failed (FACTORY_ERROR)"
            assert evidence.exchanges == ()
            assert evidence.telemetry_json is None
            assert TOKEN not in repr(evidence)
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


def app_for(monkeypatch, env=SessionEnv, *, enabled=True, mode="simulation"):
    if enabled:
        monkeypatch.setenv("OPENENV_VALIDATION_TOKEN", TOKEN)
    else:
        monkeypatch.delenv("OPENENV_VALIDATION_TOKEN", raising=False)
    app = FastAPI()
    HTTPEnvServer(env, ValueAction, Observation, max_concurrent_envs=4).register_routes(
        app, mode=mode
    )
    return app


def exchange(ws, request):
    ws.send_json(request)
    return ws.receive_json()


def authorize(ws):
    response = exchange(
        ws, {"type": "validation_open", "data": {"schema_version": 1, "token": TOKEN}}
    )
    assert response["type"] == "validation_open", response
    return response["data"]["capability"]


def read(ws, capability):
    return exchange(
        ws,
        {
            "type": "validation_read",
            "data": {"schema_version": 1, "capability": capability},
        },
    )


@pytest.mark.parametrize("invalid_action", [{}, {"value": -1}])
def test_failed_operation_marks_subject_record_incomplete(monkeypatch, invalid_action):
    class FailingStep(SessionEnv):
        def step(self, action):
            if action.value < 0:
                raise RuntimeError("step failed")
            return super().step(action)

    with (
        TestClient(app_for(monkeypatch, FailingStep)) as client,
        client.websocket_connect("/ws") as ws,
    ):
        capability = authorize(ws)
        exchange(ws, {"type": "reset", "data": {"seed": 1}})
        exchange(ws, {"type": "step", "data": {"value": 0.5}})
        assert exchange(ws, {"type": "step", "data": invalid_action})["type"] == "error"
        assert exchange(ws, {"type": "state"})["type"] == "state"
        trajectory = read(ws, capability)["data"]["trajectory"]
        assert trajectory["complete"] is False
        assert trajectory["reason"] == "Session operation failed"
        assert len(trajectory["records"]) == 2


@pytest.mark.parametrize(
    "env,accepted",
    [
        (SessionEnv, True),
        (DropsSeed, False),
        (AsyncSeed, True),
        (AsyncDropsSeed, False),
        (KwargSeed, True),
    ],
)
def test_same_session_seed_scores_and_subject_record(monkeypatch, env, accepted):
    with TestClient(app_for(monkeypatch, env)) as client:
        with client.websocket_connect("/ws") as ws:
            capability = authorize(ws)
            requests = [
                {"type": "reset", "data": {"seed": 42}},
                {"type": "state"},
                {"type": "step", "data": {"value": 0.6}},
                {"type": "state"},
            ]
            responses = [exchange(ws, request) for request in requests]
            snapshot = read(ws, capability)["data"]
            assert snapshot["seed"] == {
                "requested": True,
                "value": 42,
                "accepted": accepted,
            }
            assert responses[-1]["data"]["step_count"] == 1
            assert responses[0]["data"]["observation"]["metadata"]["seed"] == (
                42 if accepted else None
            )
            assert snapshot["trajectory"]["complete"] is True
            assert snapshot["trajectory"]["source"] == "openenv-server"
            assert snapshot["trajectory"]["records"] == [
                {"operation": req["type"], "request": req, "response": resp}
                for req, resp in zip(requests, responses)
            ]
            assert len(snapshot["attribution"]) == 1
            assert snapshot["attribution"][0]["step_index"] == 0
            assert all(node["evaluated"] for node in snapshot["rubric"])
            assert snapshot["rubric"][0]["score"] == pytest.approx(0.6)
            # Editing the collector's copy cannot modify the subject's retained record.
            responses[-1]["data"]["step_count"] = 999
            assert (
                read(ws, capability)["data"]["trajectory"]["records"][-1]["response"][
                    "data"
                ]["step_count"]
                == 1
            )
            assert TOKEN not in json.dumps(snapshot)
            assert capability not in json.dumps(snapshot)


@pytest.mark.parametrize("enabled,mode", [(False, "simulation"), (True, "production")])
def test_validation_is_opt_in_and_simulation_only(monkeypatch, enabled, mode):
    with TestClient(app_for(monkeypatch, enabled=enabled, mode=mode)) as client:
        with client.websocket_connect("/ws") as ws:
            denied = exchange(
                ws,
                {
                    "type": "validation_open",
                    "data": {"schema_version": 1, "token": TOKEN},
                },
            )
            assert denied["type"] == "error"
            assert TOKEN not in json.dumps(denied)
            # Ordinary replay clients do not need to know about telemetry.
            assert (
                exchange(ws, {"type": "reset", "data": {"seed": 4}})["type"]
                == "observation"
            )


def test_missing_wrong_malformed_cross_socket_and_expired_capabilities(monkeypatch):
    with TestClient(app_for(monkeypatch)) as client:
        with (
            client.websocket_connect("/ws") as first,
            client.websocket_connect("/ws") as second,
        ):
            assert read(first, "x" * 32)["type"] == "error"
            for data in (
                {},
                {"schema_version": 1, "token": "wrong-" * 8},
                {"schema_version": 999, "token": TOKEN},
            ):
                denied = exchange(first, {"type": "validation_open", "data": data})
                assert denied["type"] == "error"
                assert TOKEN not in json.dumps(denied)
                assert "wrong-" not in json.dumps(denied)
            first_cap, second_cap = authorize(first), authorize(second)
            assert first_cap != second_cap
            assert read(second, first_cap)["type"] == "error"
            assert read(first, second_cap)["type"] == "error"
            assert read(first, first_cap)["type"] == "validation"
        with client.websocket_connect("/ws") as fresh:
            authorize(fresh)
            assert read(fresh, first_cap)["type"] == "error"


def test_late_open_cannot_discard_prior_replay_operations(monkeypatch):
    with (
        TestClient(app_for(monkeypatch)) as client,
        client.websocket_connect("/ws") as ws,
    ):
        exchange(ws, {"type": "reset", "data": {"seed": 0}})
        assert (
            exchange(
                ws,
                {
                    "type": "validation_open",
                    "data": {"schema_version": 1, "token": TOKEN},
                },
            )["type"]
            == "error"
        )


def test_production_mcp_has_no_telemetry_or_reset_tools(monkeypatch):
    with TestClient(app_for(monkeypatch, mode="production")) as client:
        assert client.post("/reset", json={}).status_code == 404
        with client.websocket_connect("/mcp") as ws:
            listed = exchange(
                ws, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            )
            assert listed["result"]["tools"] == []
            for name in ("reset", "validation_open", "validation_read"):
                result = exchange(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": {}},
                    },
                )
                assert "error" in result


@pytest.mark.parametrize("broken", [False, True])
def test_absent_or_uninspectable_rubric_does_not_hide_subject_record(
    monkeypatch, broken
):
    class NoRubricEnv(SessionEnv):
        def __init__(self):
            super().__init__()
            if broken:
                # Exact stock containers have known config; use a custom leaf.
                self.rubric = ValueScore()
                self.rubric.validation_config = lambda: {"private": object()}
            else:
                self.rubric = None

        def step(self, action):
            self.count += 1
            return Observation(reward=0.5)

    with (
        TestClient(app_for(monkeypatch, NoRubricEnv)) as client,
        client.websocket_connect("/ws") as ws,
    ):
        capability = authorize(ws)
        exchange(ws, {"type": "reset", "data": {"seed": 1}})
        exchange(ws, {"type": "step", "data": {"value": 0.5}})
        snapshot = read(ws, capability)["data"]
        assert snapshot["trajectory"]["complete"] is True
        assert len(snapshot["trajectory"]["records"]) == 2
        assert snapshot["rubric"] == []
        assert bool(snapshot["rubric_error"]) is broken
