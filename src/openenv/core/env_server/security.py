# SPDX-License-Identifier: BSD-3-Clause

"""Optional fastapi-guard security middleware for environment servers.

Opt-in via the `OPENENV_GUARD_*` environment variables. When disabled (the
default), `attach_guard` is a no-op and environment servers behave exactly as
before.

Install the extra with `pip install "openenv[security]"` and enable with
`OPENENV_GUARD_ENABLED=1` to get IP block/allow lists, rate limiting,
user-agent blocking, penetration-attempt detection with auto-ban, and optional
Redis-backed distributed state (fastapi-guard).

Health and schema endpoints are excluded from detection checks so
orchestrators can probe the server; global IP allow/block lists are still
enforced on those paths.

WebSocket connections (`/ws`, `/mcp`, `/harness`, `/ws/ui`) get the same
checks on the handshake: ban, IP lists, a separate rate-limit bucket and
penetration detection on the path, query and headers. A refused handshake is
closed before `accept()`, so the client gets HTTP 403 on the upgrade and no
socket. Messages on an accepted connection are not screened, for the same
reason request bodies are not. With Redis configured, the guard initializes at
startup; while Redis is unreachable a handshake is closed with 1013 and HTTP
requests are refused.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from starlette import status
from starlette.exceptions import WebSocketException
from starlette.websockets import WebSocket

DEFAULT_EXCLUDED_PATHS = [
    "/health",
    "/schema",
    "/metadata",
    "/docs",
    "/redoc",
    "/openapi.json",
]
# Empty by default on purpose: trusting private-range peers would let
# any of them spoof X-Forwarded-For past the blocklist and rate limits.
# Behind a reverse proxy, list the proxy's addresses in
# OPENENV_GUARD_TRUSTED_PROXIES.
DEFAULT_TRUSTED_PROXIES = ""


def _env_list(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return False
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    msg = f"{name} must be a boolean (1/true/yes/on or 0/false/no/off), got {raw!r}"
    raise ValueError(msg)


def _build_security_config() -> Any:
    from guard import SecurityConfig

    kwargs: dict[str, Any] = {
        "enable_rate_limiting": True,
        "rate_limit": int(os.environ.get("OPENENV_GUARD_RATE_LIMIT", "100")),
        "rate_limit_window": int(
            os.environ.get("OPENENV_GUARD_RATE_LIMIT_WINDOW", "60")
        ),
        "enable_ip_banning": True,
        "auto_ban_threshold": int(
            os.environ.get("OPENENV_GUARD_AUTO_BAN_THRESHOLD", "10")
        ),
        "auto_ban_duration": int(
            os.environ.get("OPENENV_GUARD_AUTO_BAN_DURATION", "300")
        ),
        # Environment/request bodies are model input (observations, prompts,
        # completions), not commands this server executes: signature-scanning
        # them false-positives on ordinary content. URL, query, and header
        # screening stay on.
        "detection_scan_body": False,
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable at localhost.
        "enable_redis": False,
        "blacklist": tuple(_env_list("OPENENV_GUARD_BLOCKED_IPS")),
        "blocked_user_agents": _env_list("OPENENV_GUARD_BLOCKED_USER_AGENTS"),
        "trusted_proxies": tuple(
            _env_list("OPENENV_GUARD_TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES)
        ),
        "trusted_proxy_depth": int(
            os.environ.get("OPENENV_GUARD_TRUSTED_PROXY_DEPTH", "1")
        ),
        "exclude_paths": DEFAULT_EXCLUDED_PATHS,
        # fastapi-guard sends security headers unless this is None, and its
        # X-Frame-Options: SAMEORIGIN stops a Space page from framing the app.
        "security_headers": None,
    }

    allowed_ips = _env_list("OPENENV_GUARD_ALLOWED_IPS")
    if allowed_ips:
        kwargs["whitelist"] = tuple(allowed_ips)

    redis_url = os.environ.get("OPENENV_GUARD_REDIS_URL")
    if redis_url:
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url

    return SecurityConfig(**kwargs)


class _WebSocketGuardMiddleware:
    # SecurityMiddleware is a BaseHTTPMiddleware, which never sees websocket
    # scopes, so every handshake goes through fastapi-guard's guard_websocket.
    def __init__(self, app: Any, config: Any) -> None:
        from guard import guard_websocket
        from guard.lifespan import _find_security_redis_handler, guard_startup

        self.app = app
        self.config = config
        self.guard_websocket = guard_websocket
        # guard_websocket reads Redis through this; None means not connected.
        self.find_redis = _find_security_redis_handler
        self.guard_startup = guard_startup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":
            websocket = WebSocket(scope, receive, send)
            try:
                if not await self._redis_ready(scope["app"]):
                    # The HTTP side refuses too until Redis is reachable.
                    raise WebSocketException(
                        code=status.WS_1013_TRY_AGAIN_LATER,
                        reason="Service temporarily unavailable",
                    )
                await self.guard_websocket(websocket)
            except WebSocketException as exc:
                await websocket.close(code=exc.code, reason=exc.reason)
                return
        await self.app(scope, receive, send)

    async def _redis_ready(self, app: Any) -> bool:
        if not self.config.enable_redis or self.find_redis(app) is not None:
            return True
        # Retry on each handshake, like the HTTP side's first request.
        await self.guard_startup(app)
        return self.find_redis(app) is not None


def attach_guard(app: Any) -> None:
    """Add the fastapi-guard middleware to `app` when enabled via env.

    No-op unless OPENENV_GUARD_ENABLED is set to 1/true/yes/on.
    """
    if not _env_bool("OPENENV_GUARD_ENABLED"):
        return
    if getattr(app.state, "guard_attached", False):
        return
    try:
        from guard import SecurityMiddleware
        from guard.lifespan import guard_lifespan
    except ImportError as exc:
        raise ImportError(
            "OPENENV_GUARD_ENABLED requires fastapi-guard. "
            'Install it with: pip install "openenv[security]"'
        ) from exc

    config = _build_security_config()
    app.add_middleware(SecurityMiddleware, config=config)
    app.add_middleware(_WebSocketGuardMiddleware, config=config)
    # Initialize the guard (Redis included) at startup, so a WebSocket-only
    # worker never runs on in-memory state.
    existing_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: Any):
        async with guard_lifespan(app):
            async with existing_lifespan(app) as state:
                yield state

    app.router.lifespan_context = lifespan
    app.state.guard_attached = True
