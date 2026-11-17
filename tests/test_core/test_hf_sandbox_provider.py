# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for HFSandboxProvider and its local auth proxy. No Hugging Face
account or network needed: the sandbox pool is faked, and loopback FastAPI
apps stand in for the HF sandbox proxy endpoint."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import RedirectResponse
from openenv.core.containers.runtime import hf_sandbox_provider as hf
from openenv.core.containers.runtime.hf_sandbox_provider import (
    _find_available_port,
    _LocalAuthProxy,
    HFSandboxProvider,
)
from websockets.asyncio.client import connect as ws_connect


class _FakeUpstream:
    """Loopback stand-in for the HF sandbox proxy endpoint: records headers,
    answers every path, echoes one WebSocket text frame, and can emit
    HTTP/WS redirects to test that the proxy never follows them."""

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
def fake_upstream():
    upstream = _FakeUpstream()
    upstream.start()
    yield upstream
    upstream.stop()


def _ws_roundtrip(base: str, path: str = "/ws"):
    async def attempt():
        async with ws_connect(f"{base.replace('http://', 'ws://')}{path}") as ws:
            await ws.send("ping")
            return await ws.recv()

    return asyncio.run(attempt())


# --- proxy behavior ---


def test_proxy_injects_headers_on_http(fake_upstream):
    proxy = _LocalAuthProxy(
        target_url=fake_upstream.url,
        headers={"Authorization": "Bearer hf-secret"},
    )
    base = proxy.start()
    try:
        response = requests.get(f"{base}/health", timeout=10)
        assert response.status_code == 200
        assert fake_upstream.headers["authorization"] == "Bearer hf-secret"
    finally:
        proxy.stop()


def test_proxy_bridges_websocket_with_headers(fake_upstream):
    proxy = _LocalAuthProxy(
        target_url=fake_upstream.url,
        headers={"Authorization": "Bearer hf-secret"},
    )
    base = proxy.start()
    try:
        assert _ws_roundtrip(base) == "echo:ping"
        assert fake_upstream.ws_headers["authorization"] == "Bearer hf-secret"
    finally:
        proxy.stop()


def test_http_redirect_is_rejected_not_followed(fake_upstream):
    attacker = _FakeUpstream()
    attacker.start()
    try:
        fake_upstream.redirect_target = f"{attacker.url}/steal?token=s3cret"
        proxy = _LocalAuthProxy(
            target_url=fake_upstream.url,
            headers={"Authorization": "Bearer hf-secret"},
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


def test_ws_redirect_is_not_followed(fake_upstream):
    attacker = _FakeUpstream()
    attacker.start()
    try:
        fake_upstream.ws_redirect_target = (
            attacker.url.replace("http://", "ws://") + "/steal?token=s3cret"
        )
        proxy = _LocalAuthProxy(
            target_url=fake_upstream.url,
            headers={"Authorization": "Bearer hf-secret"},
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


def test_ws_handshake_failure_is_sanitized(fake_upstream, caplog):
    from websockets.exceptions import SecurityError

    proxy = _LocalAuthProxy(
        target_url=fake_upstream.url,
        headers={"Authorization": "Bearer hf-secret"},
    )
    base = proxy.start()
    secret = "ws://127.0.0.1:9/session?token=review-url-secret"
    try:
        with (
            patch(
                "openenv.core.containers.runtime.hf_sandbox_provider.ws_connect",
                side_effect=SecurityError(
                    f"cannot follow redirect to non-secure URI {secret}"
                ),
            ),
            caplog.at_level(logging.WARNING, logger=hf.__name__),
        ):
            with pytest.raises(Exception):
                _ws_roundtrip(base)
        # The route must handle the failure itself with a generic warning:
        # the upstream exception chain (which embeds the URL) must not be
        # logged or propagated.
        assert "handshake failed" in caplog.text
        assert "review-url-secret" not in caplog.text
    finally:
        proxy.stop()


def test_stop_keeps_handles_when_thread_does_not_exit(fake_upstream):
    proxy = _LocalAuthProxy(target_url=fake_upstream.url, headers={})
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


# --- provider lifecycle ---


@pytest.fixture
def fake_pool(fake_upstream):
    """Fake the huggingface_hub sandbox pool; the sandbox proxy points at the
    loopback fake upstream."""
    sandbox = MagicMock()
    sandbox.proxy_url_for.return_value = fake_upstream.url
    sandbox.proxy_headers = {"Authorization": "Bearer hf-secret"}
    pool = MagicMock()
    pool.create.return_value = sandbox
    with patch.object(hf, "_get_pool", return_value=pool):
        yield sandbox


def test_start_container_runs_server_and_proxies(fake_pool, fake_upstream):
    provider = HFSandboxProvider(image="hf.co/spaces/openenv/echo_env")
    base = provider.start_container(env_vars={"A": "1"})
    try:
        assert base.startswith("http://127.0.0.1:")
        fake_pool.run.assert_called_once()
        response = requests.get(f"{base}/health", timeout=10)
        assert response.status_code == 200
        assert fake_upstream.headers["authorization"] == "Bearer hf-secret"
    finally:
        provider.stop_container()
    fake_pool.kill.assert_called_once()


def test_stop_container_kills_sandbox_even_if_proxy_stop_fails(fake_pool):
    provider = HFSandboxProvider(image="hf.co/spaces/openenv/echo_env")
    provider.start_container()

    with patch.object(
        _LocalAuthProxy, "stop", side_effect=[RuntimeError("proxy did not stop"), None]
    ):
        with pytest.raises(RuntimeError, match="did not stop"):
            provider.stop_container()
        # The sandbox was killed even though the proxy shutdown failed.
        fake_pool.kill.assert_called_once()
        provider.stop_container()

    assert provider._proxy is None


def test_start_refused_while_proxy_cleanup_pending(fake_pool):
    provider = HFSandboxProvider(image="hf.co/spaces/openenv/echo_env")
    provider.start_container()

    with patch.object(
        _LocalAuthProxy, "stop", side_effect=[RuntimeError("proxy did not stop"), None]
    ):
        with pytest.raises(RuntimeError, match="did not stop"):
            provider.stop_container()
        # The old proxy thread may still hold the loopback port: a restart
        # must not overwrite the retained handle.
        with pytest.raises(RuntimeError, match="stop_container"):
            provider.start_container()
        provider.stop_container()

    assert provider._proxy is None
    base = provider.start_container()
    assert base.startswith("http://127.0.0.1:")
    provider.stop_container()


def test_second_start_is_rejected(fake_pool):
    provider = HFSandboxProvider(image="hf.co/spaces/openenv/echo_env")
    provider.start_container()
    try:
        with pytest.raises(RuntimeError, match="already has an active service"):
            provider.start_container()
    finally:
        provider.stop_container()
