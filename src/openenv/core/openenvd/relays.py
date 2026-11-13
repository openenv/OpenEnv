# SPDX-License-Identifier: BSD-3-Clause

"""Relays: every call out of the agent zone crosses one of these.

Three relays share one HTTP proxy core:

- the env relay forwards any HTTP request or websocket to the env server's Unix
  socket, unchanged;
- the model proxy swaps the harness's placeholder key for the real one, so the
  real key never enters a zone;
- a service relay forwards to a hidden service's Unix socket and strips
  headers that would reveal what is behind it.

Callers see the upstream's status, headers (minus hop-by-hop ones) and bytes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

HOOK_BODY_LIMIT = 64 * 1024
"""Bodies passed to `on_http_response` are read up to this many bytes."""

HTTP_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

# Set by the client library or by the websocket handshake, never forwarded.
_REQUEST_DROP = HOP_BY_HOP | {"host", "content-length"}
_WS_DROP = _REQUEST_DROP | {
    "authorization",
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-extensions",
    "sec-websocket-protocol",
}
_INFRA_HEADERS = frozenset({"server", "via", "x-powered-by"})

JsonHook = Callable[[dict], Awaitable[None]]
ActivityHook = Callable[[], None]


def episode_unavailable(**headers: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": "episode_unavailable"}}, status_code=503, headers=headers
    )


def _clean(headers: Iterable[tuple[bytes, bytes]], drop: frozenset[str] | set[str]):
    return [(k, v) for k, v in headers if k.decode("latin-1").lower() not in drop]


def _parse_json(payload: str | bytes) -> Any:
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


def _target(scope: dict) -> str:
    """The request target exactly as the caller sent it."""
    raw = scope.get("raw_path") or scope["path"].encode()
    target = raw.decode("latin-1")
    if scope.get("query_string"):
        target += "?" + scope["query_string"].decode("latin-1")
    return target


class _Tee:
    """Async iterator over an upstream body that keeps a bounded copy for a hook."""

    def __init__(
        self,
        upstream: httpx.Response,
        done: Callable[[bytes, bool], Awaitable[None]] | None,
    ):
        self._upstream = upstream
        self._done = done
        self._buf = bytearray()
        self._truncated = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._upstream.aiter_raw():
                if self._done is not None:
                    if len(self._buf) < HOOK_BODY_LIMIT:
                        self._buf += chunk[: HOOK_BODY_LIMIT - len(self._buf)]
                    if len(self._buf) >= HOOK_BODY_LIMIT:
                        self._truncated = True
                yield chunk
        finally:
            await self._upstream.aclose()
            if self._done is not None:
                await self._done(bytes(self._buf), self._truncated)


async def _forward(
    client: httpx.AsyncClient,
    request: Request,
    body: bytes,
    *,
    drop: frozenset[str] | set[str],
    add: list[tuple[bytes, bytes]] = (),
    strip_response: frozenset[str] | set[str] = HOP_BY_HOP,
    done: Callable[[int, bytes, bool], Awaitable[None]] | None = None,
) -> Response:
    """Send `request` upstream and stream the answer back."""
    headers = _clean(request.headers.raw, drop) + list(add)
    upstream_request = client.build_request(
        request.method, _target(request.scope), headers=headers, content=body
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        return JSONResponse(
            {"error": {"code": "upstream_unavailable"}}, status_code=502
        )
    status = upstream.status_code

    async def finished(buf: bytes, truncated: bool) -> None:
        await done(status, buf, truncated)

    tee = _Tee(upstream, finished if done is not None else None)
    response = StreamingResponse(tee, status_code=status)
    response.raw_headers = _clean(upstream.headers.raw, strip_response)
    return response


def _upstream_client(uds: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=uds),
        base_url="http://env",
        timeout=httpx.Timeout(None, connect=10.0),
    )


class EnvRelay:
    """The env relay's endpoints, reusable by apps that put a guard in front.

    Args:
        upstream_uds (`str`):
            The env server's Unix socket.
        on_activity (`Callable[[], None]`, *optional*):
            Called on every request and every frame a caller sends.
        on_ws_out (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object the env server sends on `/ws`.
        on_ws_in (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object a caller sends on `/ws`.
        on_http_response (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called after every non-MCP request with `method`, `path`, `status`
            and the JSON `body` when it parses.
    """

    def __init__(
        self,
        upstream_uds: str,
        *,
        on_activity: ActivityHook | None = None,
        on_ws_out: JsonHook | None = None,
        on_ws_in: JsonHook | None = None,
        on_http_response: JsonHook | None = None,
    ):
        self.upstream_uds = upstream_uds
        self.on_activity = on_activity
        self.on_ws_out = on_ws_out
        self.on_ws_in = on_ws_in
        self.on_http_response = on_http_response
        self.client = _upstream_client(upstream_uds)

    def _active(self) -> None:
        if self.on_activity is not None:
            self.on_activity()

    async def http(self, request: Request) -> Response:
        body = await request.body()
        path = request.url.path
        self._active()
        is_mcp = request.method == "POST" and path.rstrip("/") == "/mcp"
        done = None
        if not is_mcp and self.on_http_response is not None:
            hook = self.on_http_response

            async def done(status: int, buf: bytes, truncated: bool) -> None:
                parsed = None if truncated else _parse_json(buf)
                await hook(
                    {
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "body": parsed,
                    }
                )

        return await _forward(
            self.client,
            request,
            body,
            drop=_REQUEST_DROP | {"authorization"},
            done=done,
        )

    async def websocket(self, websocket: WebSocket) -> None:
        from websockets.asyncio.client import unix_connect
        from websockets.exceptions import ConnectionClosed, InvalidHandshake

        path = websocket.url.path
        is_gym = path.rstrip("/") == "/ws"
        protocols = [
            p.strip()
            for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if p.strip()
        ]
        headers = [
            (k.decode("latin-1"), v.decode("latin-1"))
            for k, v in _clean(websocket.headers.raw, _WS_DROP)
        ]
        try:
            upstream = await unix_connect(
                self.upstream_uds,
                uri="ws://env" + _target(websocket.scope),
                additional_headers=headers,
                subprotocols=protocols or None,
                max_size=None,
                compression=None,
                open_timeout=10,
                ping_interval=None,
            )
        except (OSError, InvalidHandshake, asyncio.TimeoutError):
            await websocket.close(code=1011)
            return

        await websocket.accept(subprotocol=upstream.subprotocol)

        async def client_to_upstream() -> None:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    await upstream.close(code=message.get("code") or 1000)
                    return
                payload = message.get("text")
                if payload is None:
                    payload = message.get("bytes") or b""
                self._active()
                if is_gym and self.on_ws_in is not None:
                    parsed = _parse_json(payload)
                    if isinstance(parsed, dict):
                        await self.on_ws_in(parsed)
                await upstream.send(payload)

        async def upstream_to_client() -> None:
            try:
                async for payload in upstream:
                    if isinstance(payload, str):
                        await websocket.send_text(payload)
                    else:
                        await websocket.send_bytes(payload)
                    if is_gym and self.on_ws_out is not None:
                        parsed = _parse_json(payload)
                        if isinstance(parsed, dict):
                            await self.on_ws_out(parsed)
            except ConnectionClosed:
                pass
            code = upstream.close_code or 1000
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=code, reason=upstream.close_reason or "")

        tasks = [
            asyncio.ensure_future(client_to_upstream()),
            asyncio.ensure_future(upstream_to_client()),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(
                    exc, (ConnectionClosed, WebSocketDisconnect, RuntimeError)
                ):
                    raise exc
        finally:
            await upstream.close()

    async def aclose(self) -> None:
        await self.client.aclose()


def env_relay_app(
    upstream_uds: str,
    *,
    on_activity: ActivityHook | None = None,
    on_ws_out: JsonHook | None = None,
    on_ws_in: JsonHook | None = None,
    on_http_response: JsonHook | None = None,
) -> Starlette:
    """Relay any request or websocket to the env server, unchanged.

    The caller's `Authorization` header is never forwarded.

    Args:
        upstream_uds (`str`):
            The env server's Unix socket.
        on_activity (`Callable[[], None]`, *optional*):
            Called on every request and every frame a caller sends.
        on_ws_out (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object the env server sends on `/ws`, for
            example to notice `done: true`.
        on_ws_in (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called with every JSON object a caller sends on `/ws`.
        on_http_response (`Callable[[dict], Awaitable[None]]`, *optional*):
            Called after every non-MCP request with its method, path, status and
            parsed JSON body.

    Returns:
        `Starlette`: The relay. `app.state.relay.aclose()` closes its client.
    """
    relay = EnvRelay(
        upstream_uds,
        on_activity=on_activity,
        on_ws_out=on_ws_out,
        on_ws_in=on_ws_in,
        on_http_response=on_http_response,
    )
    app = Starlette(
        routes=[
            Route("/{path:path}", relay.http, methods=HTTP_METHODS),
            WebSocketRoute("/{path:path}", relay.websocket),
        ]
    )
    app.state.relay = relay
    return app


# --- model proxy ------------------------------------------------------------


def model_proxy_app(
    upstream_base_url: str,
    real_key: str,
    *,
    provider: str = "anthropic",
    placeholder_key: str = "openenvd-placeholder",
) -> Starlette:
    """Proxy model calls, swapping the harness's placeholder key for the real one.

    A request must carry `placeholder_key` (as `x-api-key` or a bearer token);
    anything else gets a 401 and never reaches the provider. Responses stream
    back as they arrive. The real key never leaves the control plane.

    Args:
        upstream_base_url (`str`):
            The provider's base URL, for example `https://api.anthropic.com`.
        real_key (`str`):
            The provider credential.
        provider (`str`, *optional*, defaults to `"anthropic"`):
            `anthropic` (sent as `x-api-key`) or `openai` (sent as a bearer token).
        placeholder_key (`str`, *optional*, defaults to `"openenvd-placeholder"`):
            The only key the harness holds.

    Returns:
        `Starlette`: The proxy. `app.state.aclose()` closes its client.
    """
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"unknown model provider {provider!r}")
    client = httpx.AsyncClient(
        base_url=upstream_base_url, timeout=httpx.Timeout(None, connect=10.0)
    )
    if provider == "anthropic":
        auth = [(b"x-api-key", real_key.encode())]
    else:
        auth = [(b"authorization", b"Bearer " + real_key.encode())]
    drop = _REQUEST_DROP | {"authorization", "x-api-key"}
    expected = placeholder_key.encode()

    def presented(request: Request) -> bytes:
        key = request.headers.get("x-api-key")
        if key is None:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            key = token if scheme.lower() == "bearer" else ""
        return key.encode()

    async def proxy(request: Request) -> Response:
        if not hmac.compare_digest(presented(request), expected):
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "invalid key"}},
                status_code=401,
            )
        body = await request.body()
        return await _forward(client, request, body, drop=drop, add=auth)

    app = Starlette(routes=[Route("/{path:path}", proxy, methods=HTTP_METHODS)])
    app.state.aclose = client.aclose
    return app


# --- service relay ----------------------------------------------------------


def service_relay_app(upstream_uds: str) -> Starlette:
    """Relay HTTP calls to a hidden service.

    Response headers that could reveal infrastructure (`server`, `via`,
    `x-powered-by`, `x-forwarded-*`) are dropped.

    Args:
        upstream_uds (`str`):
            The service's Unix socket.

    Returns:
        `Starlette`: The relay. `app.state.aclose()` closes its client.
    """
    client = _upstream_client(upstream_uds)

    class _Strip(frozenset):
        def __contains__(self, name: object) -> bool:
            return (
                super().__contains__(name)
                or isinstance(name, str)
                and name.startswith("x-forwarded-")
            )

    strip = _Strip(HOP_BY_HOP | _INFRA_HEADERS)

    async def proxy(request: Request) -> Response:
        body = await request.body()
        return await _forward(
            client,
            request,
            body,
            drop=_REQUEST_DROP | {"authorization"},
            strip_response=strip,
        )

    app = Starlette(routes=[Route("/{path:path}", proxy, methods=HTTP_METHODS)])
    app.state.aclose = client.aclose
    return app


# --- serving ----------------------------------------------------------------


class _Server(uvicorn.Server):
    """A uvicorn server that leaves signal handling to openenvd."""

    serve_task: asyncio.Task | None = None

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def _start(config: uvicorn.Config) -> _Server:
    server = _Server(config)
    server.serve_task = asyncio.ensure_future(server.serve())
    while not server.started:
        if server.serve_task.done():
            try:
                server.serve_task.result()
            except SystemExit as exc:
                raise OSError(f"server failed to start ({exc.code})") from None
            raise OSError("server exited before it started")
        await asyncio.sleep(0.005)
    return server


async def serve_asgi_on_unix(app, path: str) -> uvicorn.Server:
    """Start serving `app` on a Unix socket in the running loop.

    Args:
        app (`ASGIApp`):
            The application.
        path (`str`):
            Socket path. A stale socket there is replaced. Mode is set to `0o666`.

    Returns:
        `uvicorn.Server`: The started server. Stop it with [`stop_server`].
    """
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    config = uvicorn.Config(
        app,
        uds=path,
        log_level="warning",
        lifespan="off",
        server_header=False,
        date_header=False,
    )
    server = await _start(config)
    os.chmod(path, 0o666)
    return server


async def serve_asgi_on_tcp(app, host: str, port: int) -> uvicorn.Server:
    """Start serving `app` on a TCP port in the running loop.

    Args:
        app (`ASGIApp`):
            The application.
        host (`str`):
            Address to bind.
        port (`int`):
            Port to bind; `0` picks a free one (see `server.servers[0].sockets`).

    Returns:
        `uvicorn.Server`: The started server. Stop it with [`stop_server`].
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        lifespan="off",
        server_header=False,
        date_header=False,
    )
    return await _start(config)


async def stop_server(server: uvicorn.Server) -> None:
    """Ask a server started here to exit and wait until it has."""
    server.should_exit = True
    task = getattr(server, "serve_task", None)
    if task is not None:
        await task
