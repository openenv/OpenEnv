# SPDX-License-Identifier: BSD-3-Clause

"""Optional fastapi-guard security middleware for environment servers.

Opt-in via the `OPENENV_GUARD_*` environment variables. When disabled (the
default), `attach_guard` is a no-op and environment servers behave exactly as
before.

Install the extra with `pip install "openenv[security]"` and enable with
`OPENENV_GUARD_ENABLED=1` to get IP block/allow lists, rate limiting with
auto-ban, user-agent blocking, penetration-attempt detection, and optional
Redis-backed distributed state (fastapi-guard).

Health and schema endpoints are excluded from detection checks by default so
orchestrators can probe the server; global IP allow/block lists are still
enforced on those paths.

WebSocket connections (`/ws`, `/mcp`, `/harness`, `/ws/ui`) get the same
checks on the handshake: ban, IP lists, a separate rate-limit bucket and
penetration detection on the path, query and headers. A refused handshake is
closed before `accept()`, so the client gets HTTP 403 on the upgrade and no
socket. Messages on an accepted connection are not screened, for the same
reason request bodies are not by default. In passive mode a handshake that
would be refused is logged and accepted. With Redis configured, the guard
initializes at startup; while Redis is unreachable a handshake is closed with
1013, as an HTTP request gets 503.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from starlette import status
from starlette.exceptions import WebSocketException
from starlette.websockets import WebSocket

DEFAULT_EXCLUDED_PATHS = "/health,/schema,/metadata,/docs,/redoc,/openapi.json"
# Empty by default on purpose: trusting private-range peers would let
# any of them spoof X-Forwarded-For past the blocklist and rate limits.
# Behind a reverse proxy, list the proxy's addresses in
# OPENENV_GUARD_TRUSTED_PROXIES.
DEFAULT_TRUSTED_PROXIES = ""


def _env_list(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
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
        # On by default: repeated rate-limit violations feed the ban engine (OPENENV_GUARD_RATE_LIMIT_AUTO_BAN=0 keeps 429-only enforcement).
        "enable_rate_limit_auto_ban": _env_bool(
            "OPENENV_GUARD_RATE_LIMIT_AUTO_BAN", default=True
        ),
        # Environment/request bodies are model input (observations, prompts,
        # completions), not commands this server executes: signature-scanning
        # them false-positives on ordinary content. URL, query, and header
        # screening stay on; body scanning is opt-in.
        "detection_scan_body": _env_bool("OPENENV_GUARD_SCAN_BODY"),
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
        "exclude_paths": _env_list(
            "OPENENV_GUARD_EXCLUDED_PATHS", DEFAULT_EXCLUDED_PATHS
        ),
        "passive_mode": _env_bool("OPENENV_GUARD_PASSIVE_MODE"),
        "custom_log_file": os.environ.get("OPENENV_GUARD_LOG_FILE") or None,
        "log_format": os.environ.get("OPENENV_GUARD_LOG_FORMAT", "text"),
        "security_headers": (
            {
                "enabled": True,
                "hsts": {"max_age": 31536000, "include_subdomains": True},
                "frame_options": "SAMEORIGIN",
                "content_type_options": "nosniff",
                "referrer_policy": "strict-origin-when-cross-origin",
            }
            if _env_bool("OPENENV_GUARD_SECURITY_HEADERS")
            else None
        ),
        "enforce_https": _env_bool("OPENENV_GUARD_ENFORCE_HTTPS"),
    }

    # Behind a TLS-terminating proxy, enforce_https must read the forwarded
    # scheme or every request looks like plain HTTP. Follow enforce_https
    # unless explicitly overridden.
    explicit_xfp = _env_str("OPENENV_GUARD_TRUST_X_FORWARDED_PROTO")
    if explicit_xfp is not None:
        kwargs["trust_x_forwarded_proto"] = _env_bool(
            "OPENENV_GUARD_TRUST_X_FORWARDED_PROTO"
        )
    else:
        kwargs["trust_x_forwarded_proto"] = _env_bool("OPENENV_GUARD_ENFORCE_HTTPS")

    if blocked_countries := _env_list("OPENENV_GUARD_BLOCKED_COUNTRIES"):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _env_list("OPENENV_GUARD_ALLOWED_COUNTRIES"):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _env_list("OPENENV_GUARD_BLOCK_CLOUD_PROVIDERS"):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if ipinfo_token := os.environ.get("IPINFO_TOKEN"):
        kwargs["ipinfo_token"] = ipinfo_token

    allowed_ips = _env_list("OPENENV_GUARD_ALLOWED_IPS")
    if allowed_ips:
        kwargs["whitelist"] = tuple(allowed_ips)

    redis_url = os.environ.get("OPENENV_GUARD_REDIS_URL")
    if redis_url:
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = os.environ.get(
            "OPENENV_GUARD_REDIS_PREFIX", "openenv_guard:"
        )

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
            if not await self._redis_ready(scope["app"]):
                # The HTTP side answers 503 until Redis is reachable.
                await websocket.close(
                    code=status.WS_1013_TRY_AGAIN_LATER,
                    reason="Service temporarily unavailable",
                )
                return
            try:
                await self.guard_websocket(websocket)
            except WebSocketException as exc:
                if not self.config.passive_mode:
                    await websocket.close(code=exc.code, reason=exc.reason)
                    return
                logging.getLogger(__name__).warning(
                    "Passive mode: WebSocket %s from %s would be refused: %s",
                    websocket.url.path,
                    getattr(websocket.state, "client_ip", None),
                    exc.reason,
                )
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
