# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import shutil
import tempfile

import httpx
import pytest
from openenv.core.openenvd.contract import Phase
from openenv.core.openenvd.phases import PhaseError
from openenv.core.openenvd.relays import serve_asgi_on_unix, stop_server
from openenv.core.openenvd.surfaces import serve_surfaces, surfaces_app
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect, unix_connect
from websockets.exceptions import InvalidStatus

ORCH = "orch-token"


class FakeUnit:
    def __init__(self):
        self.phase = Phase.RUNNING
        self.calls = []
        self.activity = 0
        self.fail = None

    def note_activity(self):
        self.activity += 1

    async def on_episode_done(self):
        self.calls.append("done")

    async def _control(self, name):
        self.calls.append(name)
        if self.fail is not None:
            raise self.fail
        return {"did": name}

    async def info(self):
        return await self._control("info")

    async def inspect(self):
        return await self._control("inspect")

    async def resume(self):
        return await self._control("resume")

    async def reset(self):
        return await self._control("reset")

    async def end_episode(self):
        return await self._control("end_episode")

    def verdict_info(self):
        self.calls.append("verdict_info")
        if self.fail is not None:
            raise self.fail
        return {"did": "verdict_info"}


seen: list = []


async def env_http(request: Request):
    seen.append(
        {
            "path": request.url.path,
            "query": request.url.query,
            "authorization": request.headers.get("authorization"),
        }
    )
    if request.url.path == "/step":
        return JSONResponse({"observation": {}, "reward": 1.0, "done": True})
    return JSONResponse({"path": request.url.path})


async def env_ws(websocket: WebSocket):
    seen.append({"path": "/ws", "query": websocket.url.query})
    await websocket.accept()
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        frame = json.loads(message["text"])
        done = frame["type"] == "step" and frame["data"].get("finish", False)
        await websocket.send_text(
            json.dumps({"type": "observation", "data": {"done": done}})
        )


fake_env = Starlette(
    routes=[
        Route("/{p:path}", env_http, methods=["GET", "POST"]),
        WebSocketRoute("/ws", env_ws),
    ]
)


@pytest.fixture
def sockdir():
    d = tempfile.mkdtemp(prefix="oe", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def unit():
    return FakeUnit()


async def _serve(sockdir, unit, env_app):
    env_path = os.path.join(sockdir, "env.sock")
    env_server = await serve_asgi_on_unix(env_app, env_path)
    app = surfaces_app(unit, env_path, orchestrator_token=ORCH)
    path = os.path.join(sockdir, "surfaces.sock")
    server = await serve_asgi_on_unix(app, path)
    return path, env_path, [server, env_server], app


@pytest.fixture
async def surfaces(sockdir, unit):
    seen.clear()
    path, _, servers, app = await _serve(sockdir, unit, fake_env)
    yield path
    for server in servers:
        await stop_server(server)
    await app.state.relay.aclose()


def client(path, token=None):
    headers = {"authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=path),
        base_url="http://unit",
        headers=headers,
    )


async def test_agent_paths_need_no_token_and_never_see_one(surfaces):
    async with client(surfaces, token=ORCH) as c:
        response = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "x"})
    assert response.json() == {"path": "/mcp"}
    assert seen[-1]["authorization"] is None


@pytest.mark.parametrize("path", ["/reset", "/step", "/state"])
@pytest.mark.parametrize("token", [None, "wrong"])
async def test_simulation_controls_need_the_orchestrator_token(surfaces, path, token):
    async with client(surfaces, token=token) as c:
        response = await c.post(path, json={})
    assert response.status_code == 401
    assert response.json() == {"error": {"code": "not_permitted"}}
    assert seen == []


async def test_http_step_with_done_ends_the_episode(surfaces, unit):
    async with client(surfaces, token=ORCH) as c:
        response = await c.post("/step", json={"action": {}})
    assert response.json()["done"] is True
    assert unit.calls == ["done"]


async def test_gym_websocket_rejects_callers_without_the_token(surfaces):
    for headers in ({}, {"authorization": "Bearer wrong"}):
        with pytest.raises(InvalidStatus):
            async with unix_connect(
                surfaces, uri="ws://unit/ws", additional_headers=headers
            ):
                pass
    assert seen == []


async def test_gym_websocket_with_token_relays_and_detects_done(surfaces, unit):
    async with unix_connect(
        surfaces, uri=f"ws://unit/ws?session_id=s1&token={ORCH}"
    ) as ws:
        await ws.send(json.dumps({"type": "step", "data": {}}))
        assert json.loads(await ws.recv())["data"]["done"] is False
        assert unit.calls == []
        await ws.send(json.dumps({"type": "step", "data": {"finish": True}}))
        assert json.loads(await ws.recv())["data"]["done"] is True
    assert unit.calls == ["done"]
    assert seen[0]["query"] == "session_id=s1"
    assert unit.activity == 2


async def test_env_paths_are_unavailable_outside_live_phases(surfaces, unit):
    unit.phase = Phase.FROZEN
    async with client(surfaces) as c:
        response = await c.post("/mcp", json={})
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "episode_unavailable"}}
    with pytest.raises(InvalidStatus):
        async with unix_connect(
            surfaces,
            uri="ws://unit/ws",
            additional_headers={"authorization": f"Bearer {ORCH}"},
        ):
            pass
    assert seen == []


@pytest.mark.parametrize(
    "method,path,name",
    [
        ("GET", "/info", "info"),
        ("POST", "/inspect", "inspect"),
        ("POST", "/resume", "resume"),
        ("POST", "/reset_unit", "reset"),
        ("POST", "/end_episode", "end_episode"),
        ("GET", "/verdict", "verdict_info"),
    ],
)
async def test_unit_controls(surfaces, unit, method, path, name):
    async with client(surfaces) as c:
        denied = await c.request(method, path)
    async with client(surfaces, token="wrong") as c:
        denied_wrong = await c.request(method, path)
    async with client(surfaces, token=ORCH) as c:
        ok = await c.request(method, path)
        unit.fail = PhaseError("frozen cannot move to /secret/path")
        refused = await c.request(method, path)
        unit.fail = RuntimeError("crun failed at /run/openenvd/abc123")
        broken = await c.request(method, path)
    assert denied.status_code == denied_wrong.status_code == 401
    assert ok.json() == {"did": name}
    assert (refused.status_code, refused.json()) == (
        409,
        {"error": {"code": "not_permitted"}},
    )
    assert (broken.status_code, broken.json()) == (
        500,
        {"error": {"code": "infrastructure_error"}},
    )
    assert seen == []


def test_the_orchestrator_token_is_required(unit):
    with pytest.raises(ValueError):
        surfaces_app(unit, "/x", orchestrator_token="")


async def test_serves_the_real_env_server_on_tcp(sockdir, unit):
    from echo_env.server.app import app as echo_app

    env_path = os.path.join(sockdir, "env.sock")
    env_server = await serve_asgi_on_unix(echo_app, env_path)
    app = surfaces_app(unit, env_path, orchestrator_token=ORCH)
    server = await serve_surfaces(app, "127.0.0.1", 0)
    port = server.servers[0].sockets[0].getsockname()[1]
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    try:
        async with client(env_path) as direct:
            expected = await direct.post("/mcp", json=call)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            relayed = await c.post("/mcp", json=call)
        async with connect(
            f"ws://127.0.0.1:{port}/ws",
            additional_headers={"authorization": f"Bearer {ORCH}"},
        ) as ws:
            await ws.send(json.dumps({"type": "reset", "data": {}}))
            reset = json.loads(await ws.recv())
    finally:
        await stop_server(server)
        await stop_server(env_server)
        await app.state.relay.aclose()
    assert relayed.status_code == expected.status_code == 200
    assert relayed.content == expected.content
    assert reset["type"] == "observation"
    assert unit.activity == 2
