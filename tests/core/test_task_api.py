# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ORS-compatible task and split endpoints."""

import functools

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_app, HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State


class TaskAction(Action):
    value: str = ""


class TaskObservation(Observation):
    message: str = ""


class TaskEnvironment(Environment):
    def reset(self, **kwargs) -> TaskObservation:
        return TaskObservation(message="ready")

    def step(self, action: TaskAction, **kwargs) -> TaskObservation:
        return TaskObservation(message=action.value, reward=1.0)

    @property
    def state(self) -> State:
        return State()

    def list_splits(self) -> list[str]:
        return ["train", "holdout"]

    def list_tasks(self, split: str) -> list[dict[str, str]]:
        return [{"id": f"{split}-0"}, {"id": f"{split}-1"}]

    def num_tasks(self, split: str) -> int:
        return 2

    def get_task(self, split: str, index: int) -> dict[str, str | int]:
        return {"id": f"{split}-{index}", "index": index}

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, str | int]]:
        start = 0 if start is None else start
        stop = 2 if stop is None else stop
        return [{"id": f"{split}-{i}", "index": i} for i in range(start, stop)]


class UnsupportedTaskEnvironment(Environment):
    def reset(self, **kwargs) -> TaskObservation:
        return TaskObservation(message="ready")

    def step(self, action: TaskAction, **kwargs) -> TaskObservation:
        return TaskObservation(message=action.value)

    @property
    def state(self) -> State:
        return State()


class PartialStepAsyncEnvironment(Environment):
    def __init__(self):
        async def step_async(action: CallToolAction, **kwargs) -> CallToolObservation:
            return CallToolObservation(tool_name=action.tool_name, result={"ok": True})

        self.step_async = functools.partial(step_async)  # type: ignore[method-assign]

    def reset(self, **kwargs) -> TaskObservation:
        return TaskObservation(message="ready")

    def step(self, action: CallToolAction, **kwargs) -> CallToolObservation:
        return CallToolObservation(tool_name=action.tool_name, result={"sync": True})

    @property
    def state(self) -> State:
        return State()


class PartialMcpStepAsyncEnvironment(Environment):
    def __init__(self):
        async def step_async(action: Action, **kwargs) -> Observation:
            if isinstance(action, ListToolsAction):
                return ListToolsObservation(tools=[])
            return CallToolObservation(tool_name="unknown", result=None)

        self.step_async = functools.partial(step_async)  # type: ignore[method-assign]

    def reset(self, **kwargs) -> TaskObservation:
        return TaskObservation(message="ready")

    def step(self, action: Action, **kwargs) -> Observation:
        return TaskObservation(message="sync")

    @property
    def state(self) -> State:
        return State()


class WrongMcpStyleEnvironment(Environment):
    def reset(self, **kwargs) -> TaskObservation:
        return TaskObservation(message="ready")

    def step(self, action: Action, **kwargs) -> Observation:
        return TaskObservation(message=f"wrong:{type(action).__name__}")

    @property
    def state(self) -> State:
        return State()


def test_task_routes_expose_ors_compatible_shapes() -> None:
    app = FastAPI()
    server = HTTPEnvServer(
        env=TaskEnvironment,
        action_cls=TaskAction,
        observation_cls=TaskObservation,
        env_name="task_env",
    )
    server.register_routes(app)
    client = TestClient(app)

    assert client.get("/list_environments").json() == ["task_env"]
    assert client.get("/task_env/splits").json() == [
        {"name": "train", "type": "train"},
        {"name": "holdout", "type": "validation"},
    ]
    assert client.post("/task_env/tasks", json={"split": "train"}).json() == {
        "tasks": [{"id": "train-0"}, {"id": "train-1"}],
        "env_name": "task_env",
    }
    assert client.post("/task_env/num_tasks", json={"split": "train"}).json() == {
        "num_tasks": 2
    }
    assert client.post(
        "/task_env/task", json={"split": "train", "index": 1}
    ).json() == {"task": {"id": "train-1", "index": 1}}
    assert client.post(
        "/task_env/task_range",
        json={"split": "train", "start": 0, "stop": 2},
    ).json() == {
        "tasks": [{"id": "train-0", "index": 0}, {"id": "train-1", "index": 1}]
    }


def test_task_routes_reject_unknown_environment_name() -> None:
    app = FastAPI()
    server = HTTPEnvServer(
        env=TaskEnvironment,
        action_cls=TaskAction,
        observation_cls=TaskObservation,
        env_name="task_env",
    )
    server.register_routes(app)
    client = TestClient(app)

    response = client.get("/other_env/splits")

    assert response.status_code == 404


def test_task_routes_return_501_when_environment_does_not_support_tasks() -> None:
    app = FastAPI()
    server = HTTPEnvServer(
        env=UnsupportedTaskEnvironment,
        action_cls=TaskAction,
        observation_cls=TaskObservation,
        env_name="plain_env",
    )
    server.register_routes(app)
    client = TestClient(app)

    response = client.get("/plain_env/splits")

    assert response.status_code == 501


def test_create_app_threads_env_name_to_task_routes() -> None:
    app = create_app(
        TaskEnvironment,
        TaskAction,
        TaskObservation,
        env_name="created_env",
    )
    client = TestClient(app)

    assert client.get("/list_environments").json() == ["created_env"]
    assert client.get("/created_env/splits").status_code == 200


def test_step_route_handles_partial_step_async() -> None:
    app = create_app(
        PartialStepAsyncEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="partial_env",
    )
    client = TestClient(app)

    response = client.post(
        "/step",
        json={"action": {"tool_name": "submit", "arguments": {}}},
    )

    assert response.status_code == 200
    assert response.json()["observation"]["result"] == {"ok": True}


def test_mcp_style_step_handles_partial_step_async() -> None:
    app = create_app(
        PartialMcpStepAsyncEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="partial_mcp_env",
    )
    client = TestClient(app)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
    )

    assert response.status_code == 200
    assert response.json()["result"] == {"tools": []}


def test_mcp_style_tools_list_reports_wrong_observation_type() -> None:
    app = create_app(
        WrongMcpStyleEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="wrong_mcp_env",
    )
    client = TestClient(app)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
    )

    assert response.status_code == 200
    assert "expected ListToolsObservation" in response.json()["error"]["message"]


def test_mcp_style_tools_call_reports_wrong_observation_type() -> None:
    app = create_app(
        WrongMcpStyleEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="wrong_mcp_env",
    )
    client = TestClient(app)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "submit", "arguments": {}},
            "id": 1,
        },
    )

    assert response.status_code == 200
    assert "expected CallToolObservation" in response.json()["error"]["message"]
