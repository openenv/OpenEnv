# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the optional fastapi-guard security middleware wiring."""

import asyncio
import itertools
import json
import os
from contextlib import asynccontextmanager

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.security import _build_security_config, attach_guard
from openenv.core.env_server.types import Action, Observation, State
from starlette.websockets import WebSocketDisconnect

# Rate limiting state in guard-core is process-wide, so every scenario gets
# its own TEST-NET client IP to stay hermetic.
_IP = itertools.count(1)


@pytest.fixture(autouse=True)
def _optional_security_dependency(request):
    if request.node.originalname in (
        "test_disabled_by_default",
        "test_websocket_unguarded_when_disabled",
        "test_invalid_enabled_flag_raises",
    ):
        return
    pytest.importorskip("guard", reason="requires the optional security extra")


def test_invalid_enabled_flag_raises(monkeypatch):
    with pytest.raises(ValueError, match="OPENENV_GUARD_ENABLED"):
        _make_app(monkeypatch, OPENENV_GUARD_ENABLED="sometimes")


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


class GuardAction(Action):
    value: str = ""


class GuardObservation(Observation):
    message: str = ""


class GuardEnvironment(Environment):
    def reset(self, **kwargs) -> GuardObservation:
        return GuardObservation(message="ready")

    def step(self, action: GuardAction, **kwargs) -> GuardObservation:
        return GuardObservation(message=action.value, reward=1.0)

    @property
    def state(self) -> State:
        return State()


def _make_app(monkeypatch, **env):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return create_fastapi_app(GuardEnvironment, GuardAction, GuardObservation)


async def _request(app, path, client_ip=None):
    transport = httpx.ASGITransport(app=app, client=(client_ip or _unique_ip(), 50000))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        return await client.get(path)


async def test_disabled_by_default(monkeypatch):
    app = _make_app(monkeypatch)
    assert len(app.user_middleware) == 0
    response = await _request(app, "/health")
    assert response.status_code == 200


@pytest.mark.parametrize("enabled", ["1", "true", "yes", "on"])
async def test_enabled_adds_middleware(monkeypatch, enabled):
    app = _make_app(monkeypatch, OPENENV_GUARD_ENABLED=enabled)
    # SecurityMiddleware for HTTP plus the websocket handshake guard.
    assert len(app.user_middleware) == 2
    response = await _request(app, "/health")
    assert response.status_code == 200


async def test_blocked_ip_is_rejected(monkeypatch):
    blocked = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_BLOCKED_IPS=blocked,
    )
    response = await _request(app, "/health", client_ip=blocked)
    assert response.status_code == 403


async def test_ip_lists_enforced_on_excluded_paths(monkeypatch):
    blocked = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_BLOCKED_IPS=blocked,
    )
    # /health is in the default exclusion list, but global IP lists are
    # enforced there too.
    response = await _request(app, "/health", client_ip=blocked)
    assert response.status_code == 403


def _ws_reset(app, client_ip):
    with TestClient(app, client=(client_ip, 50000)) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "reset", "data": {}}))
            return websocket.receive_json()["type"]


def _ws_refusal_code(app, client_ip, path="/ws"):
    with TestClient(app, client=(client_ip, 50000)) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(path):
                pass
    return closed.value.code


def test_websocket_handshake_refused_for_blocked_ip(monkeypatch):
    blocked = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_BLOCKED_IPS=blocked,
    )
    for path in ("/ws", "/mcp"):
        assert _ws_refusal_code(app, blocked, path) == 1008
    assert _ws_reset(app, _unique_ip()) == "observation"


def test_websocket_handshake_is_rate_limited(monkeypatch):
    ip = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_RATE_LIMIT="1",
        OPENENV_GUARD_RATE_LIMIT_AUTO_BAN="0",
    )
    assert _ws_reset(app, ip) == "observation"
    assert _ws_refusal_code(app, ip) == 1008


def test_websocket_unguarded_when_disabled(monkeypatch):
    app = _make_app(monkeypatch, OPENENV_GUARD_BLOCKED_IPS="198.51.100.1")
    assert _ws_reset(app, "198.51.100.1") == "observation"


def test_websocket_passive_mode_logs_and_accepts(monkeypatch, caplog):
    blocked = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_PASSIVE_MODE="1",
        OPENENV_GUARD_BLOCKED_IPS=blocked,
    )
    assert _ws_reset(app, blocked) == "observation"
    assert "would be refused" in caplog.text


def test_guard_initializes_at_startup(monkeypatch, caplog):
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_REDIS_URL="redis://127.0.0.1:1",
    )
    with TestClient(app):
        assert "Redis unavailable during startup initialization" in caplog.text


def test_guard_preserves_existing_lifespan_state_and_cleanup(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENENV_GUARD_ENABLED", "1")
    events = []
    resource = object()

    @asynccontextmanager
    async def lifespan(app):
        events.append("startup")
        try:
            yield {"resource": resource}
        finally:
            events.append("shutdown")

    app = FastAPI(lifespan=lifespan)
    attach_guard(app)
    attach_guard(app)
    with TestClient(app) as client:
        assert events == ["startup"]
        assert client.app_state["resource"] is resource
    assert events == ["startup", "shutdown"]


def test_websocket_refused_while_redis_unreachable(monkeypatch):
    ip = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_REDIS_URL="redis://127.0.0.1:1",
    )
    # A WebSocket as the first request must not fall back to in-memory state.
    assert _ws_refusal_code(app, ip) == 1013
    with TestClient(app, client=(ip, 50000)) as client:
        assert client.get("/health").status_code == 503


def test_passive_mode_admits_while_redis_unreachable(monkeypatch, caplog):
    ip = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_PASSIVE_MODE="1",
        OPENENV_GUARD_REDIS_URL="redis://127.0.0.1:1",
    )
    assert _ws_reset(app, ip) == "observation"
    assert "would be refused" in caplog.text
    with TestClient(app, client=(ip, 50000)) as client:
        assert client.get("/health").status_code == 200


async def test_full_bundle_mapping(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENENV_GUARD_ENABLED", "1")
    monkeypatch.setenv("OPENENV_GUARD_PASSIVE_MODE", "1")
    monkeypatch.setenv("OPENENV_GUARD_SECURITY_HEADERS", "1")
    monkeypatch.setenv("OPENENV_GUARD_ENFORCE_HTTPS", "1")
    monkeypatch.setenv("OPENENV_GUARD_BLOCKED_COUNTRIES", "RU")
    monkeypatch.setenv("IPINFO_TOKEN", "test-token")
    monkeypatch.setenv("OPENENV_GUARD_BLOCK_CLOUD_PROVIDERS", "AWS")
    monkeypatch.setenv("OPENENV_GUARD_LOG_FILE", "guard.log")
    config = _build_security_config()
    assert config.passive_mode is True
    assert config.redis_fail_open is True
    assert config.enforce_https is True
    assert config.security_headers["enabled"] is True
    assert config.blocked_countries == frozenset({"RU"})
    assert config.block_cloud_providers == frozenset({"AWS"})
    assert config.custom_log_file == "guard.log"
    assert config.enable_redis is False


async def test_rate_limit_returns_429(monkeypatch):
    ip = _unique_ip()
    app = _make_app(
        monkeypatch,
        OPENENV_GUARD_ENABLED="1",
        OPENENV_GUARD_RATE_LIMIT="2",
        OPENENV_GUARD_RATE_LIMIT_WINDOW="60",
    )
    assert (await _request(app, "/metadata", client_ip=ip)).status_code == 200
    assert (await _request(app, "/metadata", client_ip=ip)).status_code == 200
    assert (await _request(app, "/metadata", client_ip=ip)).status_code == 429


async def test_default_config_zero_trust(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    config = _build_security_config()
    # No implicit trust in RFC1918 peers: an unlisted proxy means
    # X-Forwarded-For is not honored, so LAN clients cannot spoof their
    # way past the blocklist or rate limits.
    assert config.trusted_proxies == ()
    # Request bodies are model input, not commands: scanning them is opt-in.
    assert config.detection_scan_body is False
    # Repeated rate-limit violations feed the ban engine by default.
    assert config.enable_rate_limit_auto_ban is True


async def test_trusted_proxies_from_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENENV_GUARD_TRUSTED_PROXIES", "10.0.0.1,10.0.0.2")
    config = _build_security_config()
    assert config.trusted_proxies == ("10.0.0.1", "10.0.0.2")


async def test_invalid_boolean_env_raises(monkeypatch):
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENENV_GUARD_SCAN_BODY", "sometimes")
    with pytest.raises(ValueError, match="OPENENV_GUARD_SCAN_BODY"):
        _build_security_config()


@pytest.mark.parametrize("startup_fails", [False, True])
def test_guard_lifespan_exit_preserves_task_and_unwinds(
    monkeypatch, tmp_path, startup_fails
):
    import guard.lifespan

    monkeypatch.setenv("OPENENV_GUARD_ENABLED", "1")
    for name in list(os.environ):
        if name.startswith("OPENENV_GUARD_") and name != "OPENENV_GUARD_ENABLED":
            monkeypatch.delenv(name)
    original_guard_lifespan = guard.lifespan.guard_lifespan
    events = []
    resource = None
    owner = None

    @asynccontextmanager
    async def guard_context(app):
        nonlocal resource, owner
        owner = asyncio.current_task()
        # A real task group detects exit in a different task/cancel scope.
        async with anyio.create_task_group():
            with (tmp_path / "resource").open("w") as resource:
                async with original_guard_lifespan(app):
                    events.append("guard startup")
                    try:
                        yield
                    finally:
                        assert asyncio.current_task() is owner
                        events.append("guard shutdown")

    @asynccontextmanager
    async def app_context(app):
        assert asyncio.current_task() is owner
        assert not resource.closed
        events.append("app startup")
        if startup_fails:
            raise RuntimeError("app startup failed")
        try:
            yield {"resource": resource}
        finally:
            assert asyncio.current_task() is owner
            assert not resource.closed
            events.append("app shutdown")

    monkeypatch.setattr(guard.lifespan, "guard_lifespan", guard_context)
    app = FastAPI(lifespan=app_context)
    attach_guard(app)
    if startup_fails:
        with pytest.raises(Exception, match="unhandled errors") as caught:
            with TestClient(app):
                pytest.fail("startup should fail")
        assert len(caught.value.exceptions) == 1
        assert isinstance(caught.value.exceptions[0], RuntimeError)
        assert str(caught.value.exceptions[0]) == "app startup failed"
        assert events == ["guard startup", "app startup", "guard shutdown"]
    else:
        with TestClient(app) as client:
            assert client.app_state["resource"] is resource
            assert not resource.closed
        assert events == [
            "guard startup",
            "app startup",
            "app shutdown",
            "guard shutdown",
        ]
    assert resource.closed
