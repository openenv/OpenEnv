# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for AgentSandboxProvider. No cluster needed: the SDK is faked,
and a loopback FastAPI app stands in for the sandbox-router."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import RedirectResponse
from openenv.core.containers.runtime.agent_sandbox_provider import (
    _find_available_port,
    _LocalAuthProxy,
    AgentSandboxProvider,
)
from websockets.asyncio.client import connect as ws_connect


def test_requires_exactly_one_of_gateway_or_router_url():
    with pytest.raises(ValueError, match="exactly one"):
        AgentSandboxProvider(warmpool="openenv-echo-pool")
    with pytest.raises(ValueError, match="exactly one"):
        AgentSandboxProvider(
            warmpool="openenv-echo-pool",
            gateway_name="external-http-gateway",
            router_url="https://sandbox.example.com",
        )


def test_router_headers_reject_reserved_names():
    with pytest.raises(ValueError, match="reserved"):
        AgentSandboxProvider(
            warmpool="openenv-echo-pool",
            gateway_name="external-http-gateway",
            router_headers={"X-Sandbox-ID": "evil"},
        )


def test_constructor_defaults():
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    assert provider._warmpool == "openenv-echo-pool"
    assert provider._namespace == "default"
    assert provider._gateway_namespace == "default"
    assert provider._shutdown_after_seconds == 3600
    assert provider._sandbox_ready_timeout == 180
    assert provider._allow_insecure is False
    assert provider._router_headers == {}
    assert provider._client is None
    assert provider._sandbox is None
    assert provider._proxy is None


class _FakeRouter:
    """Minimal sandbox-router stand-in: records headers, answers every path,
    echoes one WebSocket text frame, and can emit HTTP/WS redirects to test
    that the proxy never follows them."""

    def __init__(self):
        self.headers: dict[str, str] = {}
        self.ws_headers: dict[str, str] = {}
        self.redirect_target = "http://127.0.0.1:1/"
        self.ws_redirect_target = "ws://127.0.0.1:1/"
        self.port = _find_available_port()
        app = FastAPI()

        @app.websocket("/ws-redirect")
        async def ws_redirect(websocket: WebSocket):
            # Answer the handshake with a plain HTTP 302 (Starlette's message
            # types for HTTP responses inside a websocket scope).
            await websocket.send(
                {
                    "type": "websocket.http.response.start",
                    "status": 302,
                    "headers": [
                        (b"location", self.ws_redirect_target.encode()),
                    ],
                }
            )
            await websocket.send({"type": "websocket.http.response.body", "body": b""})

        @app.api_route("/{path:path}", methods=["GET", "POST"])
        async def handle(path: str, request: Request):
            self.headers = dict(request.headers)
            if path == "health":
                return {"status": "ok"}
            if path == "redirect":
                return RedirectResponse(url=self.redirect_target, status_code=307)
            body = await request.body()
            return {"path": path, "body": body.decode()}

        @app.websocket("/{path:path}")
        async def ws(websocket: WebSocket):
            self.ws_headers = dict(websocket.headers)
            await websocket.accept()
            message = await websocket.receive_text()
            await websocket.send_text(f"echo:{message}")
            await websocket.close()

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self._thread.start()
        while not self._server.started:
            time.sleep(0.05)

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


@pytest.fixture
def fake_router():
    router = _FakeRouter()
    router.start()
    yield router
    router.stop()


def test_proxy_injects_headers_on_http(fake_router):
    proxy = _LocalAuthProxy(
        target_url=fake_router.url,
        headers={
            "X-Sandbox-ID": "sbx-1",
            "X-Sandbox-Namespace": "openenv",
            "X-Sandbox-Port": "8000",
            "Authorization": "Bearer t",
        },
    )
    base = proxy.start()
    try:
        assert base == f"http://127.0.0.1:{proxy.port}"
        response = requests.get(f"{base}/health", timeout=10)
        assert response.status_code == 200
        assert fake_router.headers["x-sandbox-id"] == "sbx-1"
        assert fake_router.headers["x-sandbox-namespace"] == "openenv"
        assert fake_router.headers["x-sandbox-port"] == "8000"
        assert fake_router.headers["authorization"] == "Bearer t"
    finally:
        proxy.stop()


def test_proxy_forwards_method_path_and_body(fake_router):
    proxy = _LocalAuthProxy(target_url=fake_router.url, headers={})
    base = proxy.start()
    try:
        response = requests.post(f"{base}/step?x=1", data=b'{"a":2}', timeout=10)
        assert response.status_code == 200
        assert response.json()["path"] == "step"
        assert response.json()["body"] == '{"a":2}'
    finally:
        proxy.stop()


def test_proxy_bridges_websocket_with_headers(fake_router):
    proxy = _LocalAuthProxy(
        target_url=fake_router.url, headers={"X-Sandbox-ID": "sbx-9"}
    )
    base = proxy.start()
    try:

        async def roundtrip():
            async with ws_connect(f"{base.replace('http://', 'ws://')}/ws") as ws:
                await ws.send("ping")
                return await ws.recv()

        assert asyncio.run(roundtrip()) == "echo:ping"
        assert fake_router.ws_headers["x-sandbox-id"] == "sbx-9"
    finally:
        proxy.stop()


def _make_sandbox(ingress: str = "http://203.0.113.7"):
    sandbox = MagicMock()
    sandbox.sandbox_id = "sbx-1"
    sandbox.namespace = "openenv"
    sandbox.claim_name = "sandbox-claim-abc12345"
    # A property on the real Sandbox handle, not a method.
    sandbox.is_active = True
    sandbox.connector.connect.return_value = ingress
    return sandbox


@pytest.fixture
def fake_sdk():
    """Install a fake ``k8s_agent_sandbox`` package. ``sandboxes`` records every
    handle create_sandbox returned; ``INGRESS`` is what connector.connect() reports."""
    sdk = types.ModuleType("k8s_agent_sandbox")
    models = types.ModuleType("k8s_agent_sandbox.models")

    # Strict signatures: reject kwargs the real SDK does not accept
    # (k8s-agent-sandbox 1.0.6).
    class _GatewayConfig:
        def __init__(
            self,
            gateway_name: str,
            gateway_namespace: str = "default",
            server_port: int = 8888,
        ):
            self.gateway_name = gateway_name
            self.gateway_namespace = gateway_namespace
            self.server_port = server_port

    class _DirectConfig:
        def __init__(self, api_url: str, server_port: int = 8888):
            self.api_url = api_url
            self.server_port = server_port

    models.SandboxGatewayConnectionConfig = _GatewayConfig
    models.SandboxDirectConnectionConfig = _DirectConfig
    sdk.models = models
    sdk.INGRESS = "https://gw.example.com"
    sdk.sandboxes = []

    allowed_create_kwargs = {
        "warmpool",
        "namespace",
        "sandbox_ready_timeout",
        "claim_name",
        "shutdown_after_seconds",
        "env",
    }

    def _create(**kwargs):
        unexpected = set(kwargs) - allowed_create_kwargs
        assert not unexpected, f"unsupported create_sandbox kwargs: {unexpected}"
        sandbox = _make_sandbox(sdk.INGRESS)
        sdk.sandboxes.append(sandbox)
        return sandbox

    sdk.SandboxClient = MagicMock()
    sdk.SandboxClient.return_value.create_sandbox.side_effect = _create
    with patch.dict(
        sys.modules,
        {"k8s_agent_sandbox": sdk, "k8s_agent_sandbox.models": models},
    ):
        yield sdk


def test_gateway_mode_happy_path(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        namespace="openenv",
        gateway_name="external-http-gateway",
        gateway_namespace="gateway-ns",
    )
    url = provider.start_container()
    try:
        assert url.startswith("http://127.0.0.1:")
        client_kwargs = fake_sdk.SandboxClient.call_args.kwargs
        assert client_kwargs["cleanup"] is True
        assert client_kwargs["api_client"] is None
        config = client_kwargs["connection_config"]
        assert isinstance(config, fake_sdk.models.SandboxGatewayConnectionConfig)
        assert config.gateway_name == "external-http-gateway"
        assert config.gateway_namespace == "gateway-ns"
        assert config.server_port == 8000
        create_kwargs = (
            fake_sdk.SandboxClient.return_value.create_sandbox.call_args.kwargs
        )
        assert create_kwargs["warmpool"] == "openenv-echo-pool"
        assert create_kwargs["namespace"] == "openenv"
        assert create_kwargs["sandbox_ready_timeout"] == 180
        assert create_kwargs["shutdown_after_seconds"] == 3600
        assert create_kwargs["env"] is None
        assert re.fullmatch(r"openenv-[0-9a-f]{20}", create_kwargs["claim_name"])
        # An https ingress passes through untouched.
        assert provider._proxy.target_url == "https://gw.example.com"
        assert provider._proxy.headers["X-Sandbox-ID"] == "sbx-1"
        assert provider._proxy.headers["X-Sandbox-Namespace"] == "openenv"
        assert provider._proxy.headers["X-Sandbox-Port"] == "8000"
    finally:
        provider.stop_container()


def test_router_url_mode_uses_direct_config(fake_sdk):
    fake_sdk.INGRESS = "https://sandbox.example.com"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", router_url="https://sandbox.example.com"
    )
    provider.start_container()
    try:
        config = fake_sdk.SandboxClient.call_args.kwargs["connection_config"]
        assert isinstance(config, fake_sdk.models.SandboxDirectConnectionConfig)
        assert config.api_url == "https://sandbox.example.com"
        assert config.server_port == 8000
        assert provider._proxy.target_url == "https://sandbox.example.com"
    finally:
        provider.stop_container()


def test_env_vars_merge_and_forward(fake_sdk):
    fake_sdk.INGRESS = "https://r.example.com"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        router_url="https://r.example.com",
        env_vars={"TOKEN": "t", "A": "1"},
    )
    provider.start_container(env_vars={"A": "2"})
    try:
        create_kwargs = (
            fake_sdk.SandboxClient.return_value.create_sandbox.call_args.kwargs
        )
        assert create_kwargs["env"] == {"TOKEN": "t", "A": "2"}
    finally:
        provider.stop_container()


def test_empty_env_vars_forwarded_as_none(fake_sdk):
    fake_sdk.INGRESS = "https://r.example.com"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", router_url="https://r.example.com"
    )
    provider.start_container(env_vars={})
    try:
        create_kwargs = (
            fake_sdk.SandboxClient.return_value.create_sandbox.call_args.kwargs
        )
        assert create_kwargs["env"] is None
    finally:
        provider.stop_container()


def test_custom_ttl_and_ready_timeout_forwarded(fake_sdk):
    fake_sdk.INGRESS = "https://r.example.com"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        router_url="https://r.example.com",
        shutdown_after_seconds=60,
        sandbox_ready_timeout=30,
    )
    provider.start_container()
    try:
        create_kwargs = (
            fake_sdk.SandboxClient.return_value.create_sandbox.call_args.kwargs
        )
        assert create_kwargs["shutdown_after_seconds"] == 60
        assert create_kwargs["sandbox_ready_timeout"] == 30
    finally:
        provider.stop_container()


def test_router_headers_merged_under_routing_headers(fake_sdk):
    fake_sdk.INGRESS = "https://r.example.com"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        router_url="https://r.example.com",
        router_headers={"Authorization": "Bearer t"},
    )
    provider.start_container()
    try:
        assert provider._proxy.headers["Authorization"] == "Bearer t"
        assert provider._proxy.headers["X-Sandbox-ID"] == "sbx-1"
    finally:
        provider.stop_container()


def test_missing_sdk_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "k8s_agent_sandbox", None)
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", router_url="https://r.example.com"
    )
    with pytest.raises(ImportError, match=r"openenv\[agent-sandbox\]"):
        provider.start_container()


def test_rejects_image_argument(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with pytest.raises(ValueError, match="warmpool"):
        provider.start_container("echo-env:latest")
    fake_sdk.SandboxClient.return_value.create_sandbox.assert_not_called()


def test_rejects_bad_port(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with pytest.raises(ValueError, match="port 8000"):
        provider.start_container(port=9000)


def test_rejects_unknown_kwargs(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        provider.start_container(typo=True)


def test_accepts_autoenv_wait_timeout(fake_sdk):
    fake_sdk.INGRESS = "https://r.example.com"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", router_url="https://r.example.com"
    )
    assert provider.start_container(wait_timeout=30.0).startswith("http://127.0.0.1:")
    provider.stop_container()


def test_second_start_is_rejected(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    provider.start_container()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            provider.start_container()
    finally:
        provider.stop_container()


def test_allow_insecure_keeps_http_gateway(fake_sdk):
    # Gateway discovery reports the raw http:// gateway address.
    fake_sdk.INGRESS = "http://203.0.113.7"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        gateway_name="external-http-gateway",
        allow_insecure=True,
    )
    provider.start_container()
    try:
        assert provider._proxy.target_url == "http://203.0.113.7"
    finally:
        provider.stop_container()


def test_http_gateway_without_allow_insecure_is_rejected(fake_sdk):
    # The raw gateway IP cannot be rewritten to https: the certificate is
    # issued for a hostname, so an https://<IP> dial fails verification.
    fake_sdk.INGRESS = "http://203.0.113.7"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with pytest.raises(RuntimeError, match="not HTTPS") as exc:
        provider.start_container()
    # Guidance instead of a broken rewrite; the URL must not leak.
    assert "router_url" in str(exc.value)
    assert "allow_insecure" in str(exc.value)
    assert "203.0.113.7" not in str(exc.value)
    fake_sdk.sandboxes[-1].terminate.assert_called_once()
    assert provider._sandbox is None
    assert provider._proxy is None


def test_router_url_requires_https(fake_sdk):
    fake_sdk.INGRESS = "http://insecure-router"
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", router_url="http://insecure-router"
    )
    with pytest.raises(RuntimeError, match="not HTTPS") as exc:
        provider.start_container()
    # The ingress URL is a bearer capability: it must not leak into the error.
    assert "insecure-router" not in str(exc.value)
    fake_sdk.sandboxes[-1].terminate.assert_called_once()
    assert provider._sandbox is None
    assert provider._proxy is None


def test_failed_start_cleans_up_claim_and_proxy(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with patch.object(
        AgentSandboxProvider, "_secure_ingress_url", side_effect=OSError("boom")
    ):
        with pytest.raises(OSError, match="boom"):
            provider.start_container()
    sandbox = fake_sdk.sandboxes[-1]
    sandbox.terminate.assert_called_once()
    assert provider._proxy is None


def test_cleanup_failure_does_not_hide_start_error(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    original = fake_sdk.SandboxClient.return_value.create_sandbox.side_effect

    def _create(**kwargs):
        sandbox = original(**kwargs)
        sandbox.terminate.side_effect = ConnectionError("apiserver down")
        return sandbox

    fake_sdk.SandboxClient.return_value.create_sandbox.side_effect = _create
    with patch.object(
        AgentSandboxProvider, "_secure_ingress_url", side_effect=OSError("boom")
    ):
        # The start error surfaces, not the cleanup error, and the handle is
        # kept so stop_container() can retry the terminate.
        with pytest.raises(OSError, match="boom"):
            provider.start_container()
    assert provider._sandbox is not None
    provider._sandbox = None  # unblock fixture teardown


def test_e2e_headers_reach_router(fake_sdk, fake_router):
    fake_sdk.INGRESS = fake_router.url
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        namespace="openenv",
        router_url=fake_router.url,
        allow_insecure=True,
        router_headers={"Authorization": "Bearer t"},
    )
    base = provider.start_container()
    try:
        response = requests.get(f"{base}/health", timeout=10)
        assert response.status_code == 200
        assert fake_router.headers["x-sandbox-id"] == "sbx-1"
        assert fake_router.headers["x-sandbox-namespace"] == "openenv"
        assert fake_router.headers["x-sandbox-port"] == "8000"
        assert fake_router.headers["authorization"] == "Bearer t"
    finally:
        provider.stop_container()


def test_stop_terminates_once_and_is_idempotent(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    provider.start_container()
    sandbox = fake_sdk.sandboxes[-1]

    provider.stop_container()
    provider.stop_container()

    sandbox.terminate.assert_called_once()


def test_failed_stop_is_retried_on_same_handle(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    provider.start_container()
    sandbox = fake_sdk.sandboxes[-1]
    sandbox.terminate.side_effect = [ConnectionError("apiserver down"), None]

    with pytest.raises(ConnectionError):
        provider.stop_container()
    # Still active: a new start is refused until the retry succeeds.
    with pytest.raises(RuntimeError, match="already active"):
        provider.start_container()

    provider.stop_container()
    provider.stop_container()

    assert sandbox.terminate.call_count == 2


def test_stop_stops_proxy(fake_sdk, fake_router):
    fake_sdk.INGRESS = fake_router.url
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        router_url=fake_router.url,
        allow_insecure=True,
    )
    base = provider.start_container()
    assert requests.get(f"{base}/health", timeout=10).status_code == 200

    provider.stop_container()

    assert provider._proxy is None
    with pytest.raises(requests.RequestException):
        requests.get(f"{base}/health", timeout=2)


def test_context_manager_exit_stops_sandbox(fake_sdk):
    with AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    ) as provider:
        provider.start_container()

    fake_sdk.sandboxes[-1].terminate.assert_called_once()


def test_wait_for_ready_returns_on_healthy(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    url = provider.start_container()
    try:
        with patch("requests.get", return_value=MagicMock(status_code=200)) as get:
            provider.wait_for_ready(url, timeout_s=5)
        get.assert_called_once_with(f"{url}/health", timeout=5.0)
    finally:
        provider.stop_container()


def test_wait_for_ready_times_out(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    url = provider.start_container()
    try:
        with (
            patch("requests.get", side_effect=requests.ConnectionError),
            patch("time.sleep"),
        ):
            with pytest.raises(TimeoutError):
                provider.wait_for_ready(url, timeout_s=0.2)
    finally:
        provider.stop_container()


def test_wait_for_ready_detects_inactive_sandbox(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    url = provider.start_container()
    try:
        fake_sdk.sandboxes[-1].is_active = False
        with patch("requests.get", side_effect=requests.ConnectionError):
            with pytest.raises(RuntimeError, match="no longer active"):
                provider.wait_for_ready(url, timeout_s=5)
    finally:
        provider.stop_container()


def test_wait_for_ready_survives_transient_health_errors(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    url = provider.start_container()
    try:
        responses = [
            requests.ConnectionError,
            MagicMock(status_code=502),
            MagicMock(status_code=200),
        ]
        with patch("requests.get", side_effect=responses), patch("time.sleep"):
            provider.wait_for_ready(url, timeout_s=5)
    finally:
        provider.stop_container()


# --- Review fixes: redirect rejection, sanitized handshake failures,
# --- proxy-thread retention, and claim-identity ownership.


def _ws_roundtrip(base: str, path: str = "/ws"):
    async def attempt():
        async with ws_connect(f"{base.replace('http://', 'ws://')}{path}") as ws:
            await ws.send("ping")
            return await ws.recv()

    return asyncio.run(attempt())


def test_http_redirect_is_rejected_not_followed(fake_router):
    attacker = _FakeRouter()
    attacker.start()
    try:
        fake_router.redirect_target = f"{attacker.url}/steal?token=s3cret"
        proxy = _LocalAuthProxy(
            target_url=fake_router.url,
            headers={"Authorization": "Bearer t", "X-Router-Key": "k1"},
        )
        base = proxy.start()
        try:
            response = requests.get(f"{base}/redirect", timeout=10)
            assert response.status_code == 502
            # The redirect target must never receive the replayed request.
            assert attacker.headers == {}
        finally:
            proxy.stop()
    finally:
        attacker.stop()


def test_ws_redirect_is_not_followed(fake_router):
    attacker = _FakeRouter()
    attacker.start()
    try:
        fake_router.ws_redirect_target = (
            attacker.url.replace("http://", "ws://") + "/steal?token=s3cret"
        )
        proxy = _LocalAuthProxy(
            target_url=fake_router.url, headers={"Authorization": "Bearer t"}
        )
        base = proxy.start()
        try:
            with pytest.raises(Exception):
                _ws_roundtrip(base, "/ws-redirect")
            # websockets would replay additional_headers on a cross-origin
            # redirect; the proxy must refuse to follow it at all.
            assert attacker.ws_headers == {}
            assert attacker.headers == {}
        finally:
            proxy.stop()
    finally:
        attacker.stop()


def test_ws_handshake_failure_is_sanitized(fake_router, caplog):
    from websockets.exceptions import SecurityError

    proxy = _LocalAuthProxy(
        target_url=fake_router.url, headers={"Authorization": "Bearer t"}
    )
    base = proxy.start()
    secret = "ws://127.0.0.1:9/session?token=review-url-secret"
    try:
        with (
            patch(
                "openenv.core.containers.runtime.agent_sandbox_provider.ws_connect",
                side_effect=SecurityError(
                    f"cannot follow redirect to non-secure URI {secret}"
                ),
            ),
            caplog.at_level(
                logging.WARNING,
                logger="openenv.core.containers.runtime.agent_sandbox_provider",
            ),
        ):
            with pytest.raises(Exception):
                _ws_roundtrip(base)
        # The route must handle the failure itself with a generic warning:
        # the upstream exception chain (which embeds the URL) must not be
        # logged or propagated (RFC 002 S4, INVARIANTS credential rule).
        assert "handshake failed" in caplog.text
        assert "review-url-secret" not in caplog.text
    finally:
        proxy.stop()


def test_stop_keeps_handles_when_thread_does_not_exit(fake_router):
    proxy = _LocalAuthProxy(target_url=fake_router.url, headers={})
    proxy.start()
    real_thread = proxy._thread

    class _StuckThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    proxy._thread = _StuckThread()
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            proxy.stop()
        # Handles are retained so a later stop() can retry.
        assert proxy._thread is not None
        assert proxy._server is not None
    finally:
        proxy._thread = real_thread
        proxy.stop()
    assert proxy._thread is None


def test_stop_container_terminates_claim_even_if_proxy_stop_fails(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    provider.start_container()
    sandbox = fake_sdk.sandboxes[-1]

    with patch.object(
        _LocalAuthProxy, "stop", side_effect=[RuntimeError("proxy did not stop"), None]
    ):
        with pytest.raises(RuntimeError, match="did not stop"):
            provider.stop_container()
        # The claim was terminated even though the proxy shutdown failed.
        sandbox.terminate.assert_called_once()
        provider.stop_container()

    assert provider._proxy is None
    assert provider._sandbox is None


def test_claim_identity_is_recorded_and_cleared(fake_sdk):
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", namespace="openenv", gateway_name="g"
    )
    provider.start_container()
    create_kwargs = fake_sdk.SandboxClient.return_value.create_sandbox.call_args.kwargs
    assert re.fullmatch(r"openenv-[0-9a-f]{20}", create_kwargs["claim_name"])
    assert provider._claim == (create_kwargs["claim_name"], "openenv")

    provider.stop_container()
    assert provider._claim is None


def test_failed_provision_keeps_claim_identity_for_retry(fake_sdk):
    create = fake_sdk.SandboxClient.return_value.create_sandbox
    original = create.side_effect
    create.side_effect = TimeoutError("claim not ready")
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with pytest.raises(TimeoutError):
        provider.start_container()

    # Another start is refused while the orphaned claim needs cleanup.
    with pytest.raises(RuntimeError, match="stop_container"):
        provider.start_container()

    helper = fake_sdk.SandboxClient.return_value.k8s_helper
    provider.stop_container()
    claim_name, namespace = helper.delete_sandbox_claim.call_args.args
    assert re.fullmatch(r"openenv-[0-9a-f]{20}", claim_name)
    assert namespace == "default"

    # Cleanup done: provisioning may proceed again.
    create.side_effect = original
    url = provider.start_container()
    assert url.startswith("http://127.0.0.1:")
    provider.stop_container()


def test_pending_claim_delete_failure_is_retried(fake_sdk):
    create = fake_sdk.SandboxClient.return_value.create_sandbox
    original = create.side_effect
    create.side_effect = TimeoutError("claim not ready")
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool", gateway_name="external-http-gateway"
    )
    with pytest.raises(TimeoutError):
        provider.start_container()

    helper = fake_sdk.SandboxClient.return_value.k8s_helper
    helper.delete_sandbox_claim.side_effect = [
        ConnectionError("apiserver down"),
        None,
    ]
    with pytest.raises(ConnectionError):
        provider.stop_container()
    # Identity retained: the delete is retried, not dropped.
    assert provider._claim is not None
    provider.stop_container()
    assert helper.delete_sandbox_claim.call_count == 2
    assert provider._claim is None
    create.side_effect = original
