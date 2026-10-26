# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face-backed provider for OpenEnv environment servers."""

from __future__ import annotations

import asyncio
import hashlib
import socket
import threading
import time
from contextlib import suppress
from typing import Any

import requests
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .providers import ContainerProvider


_DEFAULT_PORT = 8000
_SERVER_COMMAND = (
    'export SBX_PROXY_DIR="${SBX_PROXY_DIR:-$HOME/.sbx/proxy}"; '
    'mkdir -p "$SBX_PROXY_DIR"; '
    'nohup server >"$HOME/openenv-server.log" 2>&1 &'
)
_MAX_WS_MESSAGE_SIZE = 100 * 1024 * 1024
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_POOLS: dict[tuple[str, str], Any] = {}
_POOL_LOCK = threading.Lock()


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _to_ws_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _pool_name(image: str, flavor: str) -> str:
    digest = hashlib.sha1(f"{image}\0{flavor}".encode("utf-8")).hexdigest()[:12]
    return f"openenv-{digest}"


def _get_sandbox_pool_cls() -> Any:
    try:
        from huggingface_hub import SandboxPool
    except ImportError as exc:
        raise RuntimeError(
            "HFSandboxProvider requires a huggingface_hub version with "
            "SandboxPool.create and Sandbox.proxy_url_for support."
        ) from exc
    return SandboxPool


def _get_pool(image: str, flavor: str) -> Any:
    key = (image, flavor)
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            SandboxPool = _get_sandbox_pool_cls()
            pool = SandboxPool(
                image=image,
                flavor=flavor,
                name=_pool_name(image, flavor),
            )
            _POOLS[key] = pool
        return pool


class _LocalAuthProxy:
    def __init__(self, *, target_url: str, headers: dict[str, str]):
        self.target_url = target_url.rstrip("/")
        self.headers = headers
        self.port = _find_available_port()
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> str:
        app = FastAPI()

        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def proxy_http(path: str, request: Request) -> Response:
            query = request.url.query
            target = f"{self.target_url}/{path}"
            if query:
                target = f"{target}?{query}"
            headers = dict(self.headers)
            content_type = request.headers.get("content-type")
            if content_type is not None:
                headers["Content-Type"] = content_type
            body = await request.body()
            try:
                upstream = await asyncio.to_thread(
                    requests.request,
                    request.method,
                    target,
                    data=body,
                    headers=headers,
                    timeout=60.0,
                    allow_redirects=True,
                )
            except requests.RequestException:
                return Response(
                    content=b"upstream HF job unreachable",
                    status_code=502,
                )
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in _HOP_BY_HOP_HEADERS
            }
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
            )

        @app.websocket("/{path:path}")
        async def proxy_websocket(path: str, websocket: WebSocket) -> None:
            query = websocket.url.query
            target = f"{_to_ws_url(self.target_url)}/{path}"
            if query:
                target = f"{target}?{query}"
            headers = {**self.headers, "accept": "*/*"}
            upstream = await self._connect_upstream_websocket(target, headers)
            await websocket.accept()
            try:
                to_upstream = asyncio.create_task(
                    self._client_to_upstream(websocket, upstream)
                )
                to_client = asyncio.create_task(
                    self._upstream_to_client(websocket, upstream)
                )
                done, pending = await asyncio.wait(
                    {to_upstream, to_client},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    with suppress(ConnectionClosed, WebSocketDisconnect):
                        task.result()
            finally:
                with suppress(Exception):
                    await upstream.close()

        import uvicorn

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("HF sandbox auth proxy failed to start")
            time.sleep(0.05)
        return self.base_url

    async def _connect_upstream_websocket(
        self, target: str, headers: dict[str, str]
    ) -> Any:
        last_error: Exception | None = None
        for _ in range(5):
            try:
                return await ws_connect(
                    target,
                    additional_headers=headers,
                    max_size=_MAX_WS_MESSAGE_SIZE,
                    compression=None,
                )
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)
        assert last_error is not None
        raise last_error

    async def _client_to_upstream(self, websocket: WebSocket, upstream: Any) -> None:
        # EnvClient sends JSON text frames; binary frames are only relayed downstream.
        async for message in websocket.iter_text():
            await upstream.send(message)

    async def _upstream_to_client(self, websocket: WebSocket, upstream: Any) -> None:
        async for message in upstream:
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(message)

    def stop(self) -> None:
        if self._server is None or self._thread is None:
            return
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None


class HFSandboxProvider(ContainerProvider):
    """Run an OpenEnv server on Hugging Face infrastructure.

    The server runs in a `huggingface_hub` sandbox pool, which bills its hosts as
    Hugging Face Jobs. `start_container()` returns a local proxy URL that adds the
    Hugging Face auth headers, so any `EnvClient` can connect to it unchanged.

    Args:
        image (`str`):
            Image to run, for example `hf.co/spaces/openenv/coding_env`. It must
            provide a `server` command that serves OpenEnv on port 8000.
        env_vars (`dict[str, str]`, *optional*):
            Environment variables for the server.
        flavor (`str`, *optional*, defaults to `"cpu-basic"`):
            Hugging Face hardware flavor of the sandbox hosts.
    """

    def __init__(
        self,
        *,
        image: str,
        env_vars: dict[str, str] | None = None,
        flavor: str = "cpu-basic",
    ):
        self.image = image
        self.env_vars = env_vars
        self.flavor = flavor
        self._sandbox: Any = None
        self._proxy: _LocalAuthProxy | None = None

    def start_container(
        self,
        image: str | None = None,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        if self._sandbox is not None:
            raise RuntimeError("HFSandboxProvider already has an active service")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(
                f"HFSandboxProvider does not support start_container kwargs: {unsupported}"
            )

        if port not in (None, _DEFAULT_PORT):
            raise ValueError(
                f"HFSandboxProvider only supports port {_DEFAULT_PORT} (got {port})."
            )

        effective_image = self.image if image is None else image
        effective_env = self.env_vars if env_vars is None else env_vars
        pool = _get_pool(effective_image, self.flavor)
        self._sandbox = pool.create(env=effective_env)
        try:
            if not hasattr(self._sandbox, "proxy_url_for") or not hasattr(
                self._sandbox, "proxy_headers"
            ):
                raise RuntimeError(
                    "HFSandboxProvider requires a huggingface_hub version with "
                    "Sandbox.proxy_url_for and Sandbox.proxy_headers support."
                )
            self._sandbox.run(
                _SERVER_COMMAND,
                shell=True,
            )
            self._proxy = _LocalAuthProxy(
                target_url=self._sandbox.proxy_url_for(_DEFAULT_PORT, "/"),
                headers=self._sandbox.proxy_headers,
            )
            return self._proxy.start()
        except Exception:
            self.stop_container()
            raise

    def stop_container(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        if self._sandbox is not None:
            sandbox = self._sandbox
            self._sandbox = None
            with suppress(Exception):
                sandbox.kill()

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        health_url = f"{base_url}/health"
        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(
            f"HF sandbox job at {base_url} did not become ready within {timeout_s}s"
        )

    def close(self) -> None:
        self.stop_container()


__all__ = ["HFSandboxProvider"]
