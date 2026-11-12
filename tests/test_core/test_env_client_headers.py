# SPDX-License-Identifier: BSD-3-Clause

"""Clients send auth headers; the Sandbox proxy keeps credentials off redirects."""

import httpx
import pytest
from fastapi import FastAPI, Request, Response, WebSocket
from openenv.core.containers.runtime.hf_sandbox_provider import _LocalAuthProxy
from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

echo_app = pytest.importorskip("echo_env.server.app").app

HEADERS = {"Authorization": "Bearer secret"}
AUTH_HEADER = (b"authorization", b"Bearer secret")


def require_token(app):
    """Reject HTTP requests and WebSocket upgrades without the token, like a private Space."""

    async def guarded(scope, receive, send):
        if scope["type"] == "lifespan" or AUTH_HEADER in scope["headers"]:
            await app(scope, receive, send)
        elif scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    return guarded


@pytest.fixture
def server_url(serve):
    return serve(require_token(echo_app))


async def test_websocket_without_headers_is_rejected(server_url):
    with pytest.raises(ConnectionError):
        async with GenericEnvClient(base_url=server_url) as env:
            await env.reset()


async def test_generic_client_sends_headers(server_url):
    async with GenericEnvClient(base_url=server_url, headers=HEADERS) as env:
        await env.reset()
        child = await env.new_session()
        await child.reset()


async def test_mcp_tool_client_sends_headers(server_url):
    async with MCPToolClient(base_url=server_url, headers=HEADERS) as env:
        await env.reset()
        assert await env.list_tools()


@pytest.mark.parametrize(
    "cross_origin", [False, True], ids=["same-origin", "cross-origin"]
)
def test_http_redirect_does_not_forward_sandbox_token(serve, cross_origin):
    upstream = FastAPI()
    destination = FastAPI()
    initial_requests = []
    redirected_tokens = []

    async def collect(request: Request):
        redirected_tokens.append(request.headers.get("x-sandbox-token"))
        return Response("redirect followed")

    upstream.add_api_route("/destination", collect, methods=["POST"])
    destination.add_api_route("/destination", collect, methods=["POST"])
    location = f"{serve(destination)}/destination" if cross_origin else "/destination"

    @upstream.post("/start")
    async def redirect(request: Request):
        initial_requests.append(
            (
                request.headers.get("x-sandbox-token"),
                request.url.query,
                await request.body(),
                request.headers.get("content-type"),
            )
        )
        return Response(
            "redirect body", status_code=307, headers={"Location": location}
        )

    proxy = _LocalAuthProxy(
        target_url=serve(upstream), headers={"X-Sandbox-Token": "sandbox-only"}
    )
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                f"{proxy.start()}/start?mode=test",
                content=b"payload",
                headers={"Content-Type": "application/octet-stream"},
                follow_redirects=False,
                timeout=5,
            )
        assert initial_requests == [
            ("sandbox-only", "mode=test", b"payload", "application/octet-stream")
        ]
        assert redirected_tokens == []
        assert response.status_code == 502
        assert "location" not in response.headers
        assert response.text == "upstream HF job redirected; rejected"
    finally:
        proxy.stop()


def test_http_proxy_preserves_direct_response(serve):
    upstream = FastAPI()

    @upstream.get("/schema")
    async def schema(request: Request):
        assert request.headers["x-sandbox-token"] == "sandbox-only"
        return Response(
            b'{"ok":true}',
            status_code=201,
            media_type="application/json",
            headers={"X-Upstream": "direct"},
        )

    proxy = _LocalAuthProxy(
        target_url=serve(upstream), headers={"X-Sandbox-Token": "sandbox-only"}
    )
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.get(f"{proxy.start()}/schema", timeout=5)
        assert response.status_code == 201
        assert response.json() == {"ok": True}
        assert response.headers["content-type"] == "application/json"
        assert response.headers["x-upstream"] == "direct"
    finally:
        proxy.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cross_origin", [False, True], ids=["same-origin", "cross-origin"]
)
async def test_websocket_redirect_does_not_forward_sandbox_token(serve, cross_origin):
    upstream = FastAPI()
    destination = FastAPI()
    initial_tokens = []
    redirected_tokens = []

    async def collect(websocket: WebSocket):
        redirected_tokens.append(websocket.headers.get("x-sandbox-token"))
        await websocket.accept()
        await websocket.close()

    upstream.add_api_websocket_route("/destination", collect)
    destination.add_api_websocket_route("/destination", collect)
    location = (
        f"{serve(destination).replace('http://', 'ws://')}/destination"
        if cross_origin
        else "/destination"
    )

    @upstream.websocket("/start")
    async def redirect(websocket: WebSocket):
        initial_tokens.append(websocket.headers.get("x-sandbox-token"))
        await websocket.send_denial_response(
            Response(status_code=307, headers={"Location": location})
        )

    proxy = _LocalAuthProxy(
        target_url=serve(upstream), headers={"X-Sandbox-Token": "sandbox-only"}
    )
    connection = None
    try:
        try:
            connection = await connect(
                f"{proxy.start().replace('http://', 'ws://')}/start",
                proxy=None,
                open_timeout=8,
            )
        except InvalidStatus:
            # Redirect rejection is the expected handshake outcome.
            pass
        else:
            await connection.close()
        assert initial_tokens and set(initial_tokens) == {"sandbox-only"}
        assert redirected_tokens == []
        assert connection is None
    finally:
        proxy.stop()


@pytest.mark.asyncio
async def test_websocket_proxy_preserves_direct_connection(serve):
    upstream = FastAPI()
    received = []

    @upstream.websocket("/ws")
    async def echo(websocket: WebSocket):
        received.append((websocket.headers.get("x-sandbox-token"), websocket.url.query))
        await websocket.accept()
        await websocket.send_text(await websocket.receive_text())
        await websocket.send_bytes(b"binary reply")
        await websocket.close()

    proxy = _LocalAuthProxy(
        target_url=serve(upstream), headers={"X-Sandbox-Token": "sandbox-only"}
    )
    try:
        async with connect(
            f"{proxy.start().replace('http://', 'ws://')}/ws?mode=test",
            proxy=None,
            open_timeout=5,
        ) as websocket:
            await websocket.send("hello")
            assert await websocket.recv() == "hello"
            assert await websocket.recv() == b"binary reply"
        assert received == [("sandbox-only", "mode=test")]
    finally:
        proxy.stop()
