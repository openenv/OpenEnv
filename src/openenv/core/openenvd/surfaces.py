# SPDX-License-Identifier: BSD-3-Clause

"""The unit's single external listener.

One TCP port carries every principal:

| Path | Who | Token |
|------|-----|-------|
| `/mcp` and other env paths | agent | none; byte-identical to the env server |
| `/ws`, `/reset`, `/step`, `/state`, `/harness` | orchestrator | orchestrator |
| `GET /info`, `GET /verdict`, `POST /inspect`, `POST /resume`, `POST /reset_unit`, `POST /end_episode` | orchestrator | orchestrator |

Tokens travel as `Authorization: Bearer <token>` or a `token` query parameter,
which is removed before anything is forwarded. Env paths are relayed only while
agent-zone containers are live (`ready`, `running`). Error bodies carry a stable
code and nothing else.
"""

from __future__ import annotations

import hmac
import inspect
import logging
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from .contract import AGENT_PHASES, Phase
from .phases import PhaseError
from .relays import EnvRelay, episode_unavailable, HTTP_METHODS, serve_asgi_on_tcp

logger = logging.getLogger(__name__)

ORCHESTRATOR_ENV_PATHS = frozenset({"/ws", "/reset", "/step", "/state", "/harness"})
"""Env server paths that control the simulation; never reachable by the agent."""

_TOKEN_KEY = "openenvd.token"


class UnitControl(Protocol):
    """What the surfaces need from the unit.

    Attributes:
        phase (`Phase`):
            The unit's current phase.
    """

    phase: Phase

    def note_activity(self) -> None:
        """The orchestrator or the agent sent something: the episode is running."""

    async def on_episode_done(self) -> None:
        """The environment reported `done: true` on a step."""

    async def info(self) -> dict:
        """Isolation state, phase and verdict. No ids, paths or backends."""

    async def inspect(self) -> dict:
        """Freeze the agent zone for inspection."""

    async def resume(self) -> dict:
        """Thaw the agent zone after an inspection."""

    async def reset(self) -> dict:
        """Tear the unit down and provision a fresh episode."""

    async def end_episode(self) -> dict:
        """Stop the agent zone and grade, as if the environment reported done."""

    def verdict_info(self) -> dict:
        """The last episode's grading results."""


def _error(code: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code}}, status_code=status)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class _TokenFromQuery:
    """Move a `token` query parameter out of the URL before routing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope.get("query_string"):
            pairs = parse_qsl(scope["query_string"].decode("latin-1"), True)
            token = [v for k, v in pairs if k == "token"]
            if token:
                rest = [(k, v) for k, v in pairs if k != "token"]
                scope = {
                    **scope,
                    "query_string": urlencode(rest).encode("latin-1"),
                    _TOKEN_KEY: token[-1],
                }
        await self.app(scope, receive, send)


def surfaces_app(
    unit: UnitControl,
    env_uds: str,
    *,
    orchestrator_token: str,
):
    """Build the ASGI app for the unit's external listener.

    Args:
        unit (`UnitControl`):
            The unit: its phase and lifecycle controls.
        env_uds (`str`):
            The env server's Unix socket.
        orchestrator_token (`str`):
            Bearer token for the orchestrator surface.

    Returns:
        `ASGIApp`: The app. `app.state.relay.aclose()` closes the env client.
    """
    if not orchestrator_token:
        raise ValueError("the orchestrator surface needs a token")

    def token_of(conn: Request | WebSocket) -> str | None:
        scheme, _, value = conn.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and value:
            return value
        return conn.scope.get(_TOKEN_KEY)

    def holds(conn: Request | WebSocket, *tokens: str | None) -> bool:
        presented = (token_of(conn) or "").encode()
        ok = False
        for token in tokens:
            if token:
                ok |= hmac.compare_digest(presented, token.encode())
        return ok

    async def episode_done() -> None:
        try:
            await _maybe_await(unit.on_episode_done())
        except Exception:
            logger.exception("on_episode_done failed")

    async def on_ws_out(frame: dict) -> None:
        data = frame.get("data")
        if frame.get("type") == "observation" and isinstance(data, dict):
            if data.get("done") is True:
                await episode_done()

    async def on_http_response(data: dict) -> None:
        body = data.get("body")
        if (
            data.get("method") == "POST"
            and data.get("path", "").rstrip("/") == "/step"
            and data.get("status") == 200
            and isinstance(body, dict)
            and body.get("done") is True
        ):
            await episode_done()

    relay = EnvRelay(
        env_uds,
        on_activity=getattr(unit, "note_activity", None),
        on_ws_out=on_ws_out,
        on_http_response=on_http_response,
    )

    def orchestrator_only(path: str) -> bool:
        return path.rstrip("/") in ORCHESTRATOR_ENV_PATHS

    async def env_http(request: Request) -> Response:
        path = request.url.path
        if orchestrator_only(path) and not holds(request, orchestrator_token):
            return _error("not_permitted", 401)
        if unit.phase not in AGENT_PHASES:
            return episode_unavailable()
        return await relay.http(request)

    async def env_ws(websocket: WebSocket) -> None:
        path = websocket.url.path
        if orchestrator_only(path) and not holds(websocket, orchestrator_token):
            await websocket.close(code=1008, reason="not_permitted")
            return
        if unit.phase not in AGENT_PHASES:
            await websocket.close(code=1013, reason="episode_unavailable")
            return
        await relay.websocket(websocket)

    def control(method: str):
        async def endpoint(request: Request) -> Response:
            if not holds(request, orchestrator_token):
                return _error("not_permitted", 401)
            try:
                result = await _maybe_await(getattr(unit, method)())
            except PhaseError:
                return _error("not_permitted", 409)
            except Exception:
                logger.exception("unit.%s failed", method)
                return _error("infrastructure_error", 500)
            return JSONResponse(result if result is not None else {})

        return endpoint

    app = Starlette(
        routes=[
            Route("/info", control("info"), methods=["GET"]),
            Route("/inspect", control("inspect"), methods=["POST"]),
            Route("/resume", control("resume"), methods=["POST"]),
            Route("/reset_unit", control("reset"), methods=["POST"]),
            Route("/end_episode", control("end_episode"), methods=["POST"]),
            Route("/verdict", control("verdict_info"), methods=["GET"]),
            Route("/{path:path}", env_http, methods=HTTP_METHODS),
            WebSocketRoute("/{path:path}", env_ws),
        ]
    )
    app.state.relay = relay
    wrapped = _TokenFromQuery(app)
    wrapped.state = app.state
    return wrapped


async def serve_surfaces(
    app, host: str = "0.0.0.0", port: int = 8100
) -> uvicorn.Server:
    """Start the external listener in the running loop.

    Args:
        app (`ASGIApp`):
            From [`surfaces_app`].
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Address to bind.
        port (`int`, *optional*, defaults to `8100`):
            Port to bind.

    Returns:
        `uvicorn.Server`: Stop it with [`~openenv.core.openenvd.relays.stop_server`].
    """
    return await serve_asgi_on_tcp(app, host, port)
