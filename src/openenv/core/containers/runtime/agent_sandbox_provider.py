# SPDX-License-Identifier: BSD-3-Clause

"""
Agent Sandbox provider for running OpenEnv environments in kubernetes-sigs
agent-sandbox sandboxes.

Requires the ``k8s-agent-sandbox`` SDK: ``pip install openenv[agent-sandbox]``.
The SDK requires Python >= 3.11.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from contextlib import suppress
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .providers import ContainerProvider

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
# Bound the sandbox lifetime (RFC 002 S6). Same default as the Novita and
# Tensorlake providers: the controller deletes the claim on expiry even if
# this process dies without calling stop_container().
_DEFAULT_SHUTDOWN_AFTER_SECONDS = 3600
_RESERVED_HEADER_PREFIX = "x-sandbox-"
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


class _LocalAuthProxy:
    """Loopback proxy that adds the sandbox-router headers to every request.

    The router routes each request to one sandbox by X-Sandbox-* headers, but
    ContainerProvider.start_container() only returns a URL string. The proxy
    gives EnvClient a plain loopback base_url and injects the headers into
    HTTP requests and WebSocket handshakes alike.

    Copied from hf_sandbox_provider._LocalAuthProxy; keep the copies aligned.
    """

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
                    content=b"upstream agent sandbox router unreachable",
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
                raise RuntimeError("Agent sandbox routing proxy failed to start")
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


class AgentSandboxProvider(ContainerProvider):
    """
    Container provider that runs environments in agent-sandbox sandboxes.

    The sandbox is claimed from a pre-created `SandboxWarmPool` whose
    `SandboxTemplate` runs an OpenEnv environment image; the image's CMD
    already serves OpenEnv on port 8000. Requests reach the sandbox through a
    Kubernetes Gateway or sandbox-router URL that routes each request by
    `X-Sandbox-*` headers. `start_container()` returns a loopback proxy URL
    that injects those headers, so any `EnvClient` connects unchanged.

    The ingress must be https. The SDK's Gateway discovery reports `http://`
    even for TLS gateways, so gateway URLs are rewritten to `https://` unless
    `allow_insecure=True`.

    Examples:

    ```python
    provider = AgentSandboxProvider(
        warmpool="openenv-echo-pool",
        gateway_name="external-http-gateway",
    )
    base_url = provider.start_container()
    provider.wait_for_ready(base_url)
    provider.stop_container()
    ```
    """

    def __init__(
        self,
        *,
        warmpool: str,
        namespace: str = "default",
        gateway_name: Optional[str] = None,
        gateway_namespace: str = "default",
        router_url: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        shutdown_after_seconds: int = _DEFAULT_SHUTDOWN_AFTER_SECONDS,
        sandbox_ready_timeout: int = 180,
        allow_insecure: bool = False,
        router_headers: Optional[Dict[str, str]] = None,
        api_client: Any = None,
    ):
        """
        Args:
            warmpool (`str`):
                Name of the `SandboxWarmPool` to claim sandboxes from. Its
                template must run an OpenEnv environment image serving port
                8000.
            namespace (`str`, *optional*, defaults to `"default"`):
                Namespace where the `SandboxClaim` is created.
            gateway_name (`str`, *optional*):
                Name of the Gateway API resource whose address routes to the
                sandbox-router. Exactly one of `gateway_name` or `router_url`
                is required.
            gateway_namespace (`str`, *optional*, defaults to `"default"`):
                Namespace of the Gateway resource.
            router_url (`str`, *optional*):
                Explicit sandbox-router URL, for custom domains or the
                in-cluster router Service DNS. Exactly one of `gateway_name`
                or `router_url` is required.
            env_vars (`dict`, *optional*):
                Environment variables for the sandbox, injected via
                `SandboxClaim.spec.env`. Setting them forces a cold start
                instead of adopting a pre-warmed pod. Merged with the
                `env_vars` given to `start_container()`, which take
                precedence.
            shutdown_after_seconds (`int`, *optional*, defaults to `3600`):
                Sandbox TTL in seconds. The controller deletes the claim on
                expiry even if this process dies (RFC 002 S6).
            sandbox_ready_timeout (`int`, *optional*, defaults to `180`):
                Seconds to wait for the claim to bind and become Ready.
            allow_insecure (`bool`, *optional*, defaults to `False`):
                Accept a plaintext `http://` ingress. Off by default so
                EnvClient traffic is encrypted (RFC 002 S1); enable only for
                local clusters such as Kind.
            router_headers (`dict`, *optional*):
                Extra headers the proxy sends on every request, for routers
                behind an authenticating gateway (e.g. an `Authorization`
                bearer). `X-Sandbox-*` names are reserved and rejected.
            api_client (`Any`, *optional*):
                Pre-configured Kubernetes `ApiClient` forwarded to the SDK to
                target a specific cluster or context.
        """
        if (gateway_name is None) == (router_url is None):
            raise ValueError(
                "AgentSandboxProvider requires exactly one of gateway_name or "
                "router_url."
            )
        for name in router_headers or {}:
            if name.lower().startswith(_RESERVED_HEADER_PREFIX):
                raise ValueError(
                    f"router_headers may not set {name!r}: X-Sandbox-* header "
                    "names are reserved for sandbox routing."
                )
        self._warmpool = warmpool
        self._namespace = namespace
        self._gateway_name = gateway_name
        self._gateway_namespace = gateway_namespace
        self._router_url = router_url
        self._env_vars = env_vars
        self._shutdown_after_seconds = shutdown_after_seconds
        self._sandbox_ready_timeout = sandbox_ready_timeout
        self._allow_insecure = allow_insecure
        self._router_headers = dict(router_headers or {})
        self._api_client = api_client
        self._client: Any = None
        self._sandbox: Any = None
        self._proxy: Optional[_LocalAuthProxy] = None

    def start_container(
        self,
        image: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Claim a sandbox, wait until it is Ready, and expose its port 8000.

        Args:
            image (`str`, *optional*):
                Must be `None`. The sandbox image is defined by the warm
                pool's template.
            port (`int`, *optional*):
                Must be `None` or `8000`.
            env_vars (`dict`, *optional*):
                Environment variables merged over the constructor `env_vars`.
            **kwargs:
                `wait_timeout` is accepted and ignored (AutoEnv forwards it;
                readiness is bounded by `sandbox_ready_timeout`).

        Returns:
            `str`: Loopback URL of the header-injecting proxy, e.g.
            `http://127.0.0.1:54321`.

        Raises:
            RuntimeError: If a sandbox from an earlier call is still active.
        """
        if self._sandbox is not None:
            raise RuntimeError(
                "An agent sandbox is already active. Call stop_container() first."
            )
        if image is not None:
            raise ValueError(
                "AgentSandboxProvider does not take an image: the sandbox image "
                "comes from the SandboxWarmPool template. Pass warmpool= to the "
                "constructor."
            )
        if port is not None and port != _DEFAULT_PORT:
            raise ValueError(
                f"AgentSandboxProvider only supports port {_DEFAULT_PORT} (got {port})."
            )
        # AutoEnv always passes env_vars={}, so merge instead of replace.
        effective_env_vars = {**(self._env_vars or {}), **(env_vars or {})}
        # AutoEnv always forwards wait_timeout; it does not apply here.
        kwargs.pop("wait_timeout", None)
        if kwargs:
            raise ValueError(
                f"Unsupported AgentSandboxProvider options: {', '.join(sorted(kwargs))}"
            )

        try:
            from k8s_agent_sandbox import SandboxClient
            from k8s_agent_sandbox.models import (
                SandboxDirectConnectionConfig,
                SandboxGatewayConnectionConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "AgentSandboxProvider requires the k8s-agent-sandbox SDK "
                "(Python >= 3.11). Install it with: pip install "
                "openenv[agent-sandbox]"
            ) from exc

        if self._gateway_name is not None:
            connection_config = SandboxGatewayConnectionConfig(
                gateway_name=self._gateway_name,
                gateway_namespace=self._gateway_namespace,
                server_port=_DEFAULT_PORT,
            )
        else:
            connection_config = SandboxDirectConnectionConfig(
                api_url=self._router_url,
                server_port=_DEFAULT_PORT,
            )
        # cleanup=True: an atexit hook deletes tracked sandboxes if the process
        # ends without stop_container(); shutdown_after_seconds covers SIGKILL.
        self._client = SandboxClient(
            connection_config=connection_config,
            cleanup=True,
            api_client=self._api_client,
        )
        self._sandbox = self._client.create_sandbox(
            warmpool=self._warmpool,
            namespace=self._namespace,
            sandbox_ready_timeout=self._sandbox_ready_timeout,
            shutdown_after_seconds=self._shutdown_after_seconds,
            env=effective_env_vars or None,
        )
        sandbox_id = self._sandbox.sandbox_id
        try:
            ingress = self._secure_ingress_url(self._sandbox.connector.connect())
            self._proxy = _LocalAuthProxy(
                target_url=ingress,
                headers={
                    **self._router_headers,
                    "X-Sandbox-ID": self._sandbox.sandbox_id,
                    "X-Sandbox-Namespace": self._sandbox.namespace,
                    "X-Sandbox-Port": str(_DEFAULT_PORT),
                },
            )
            return self._proxy.start()
        except Exception:
            # Do not let a cleanup failure hide the original error.
            try:
                self.stop_container()
            except Exception:
                logger.warning(
                    "Could not terminate agent sandbox %s after a failed start.",
                    sandbox_id,
                    exc_info=True,
                )
            raise

    def _secure_ingress_url(self, url: str) -> str:
        """Enforce https/wss transport on the ingress leg (RFC 002 S1).

        The URL is omitted from the error because an ingress URL is a bearer
        capability that must not leak into logs.
        """
        if self._allow_insecure or url.lower().startswith("https://"):
            return url
        if self._gateway_name is not None and url.lower().startswith("http://"):
            # Gateway discovery in the SDK hardcodes http:// even when the
            # Gateway serves TLS, so rewrite the scheme.
            return "https://" + url[len("http://") :]
        raise RuntimeError(
            "The agent-sandbox router URL is not HTTPS. OpenEnv requires an "
            "https/wss ingress so EnvClient traffic is encrypted. Pass "
            "allow_insecure=True to accept a plaintext http:// ingress."
        )

    def stop_container(self) -> None:
        """
        Terminate the sandbox claim and stop the local proxy.

        `Sandbox.terminate()` is idempotent and treats an already-deleted
        claim as success. If termination fails, the handle is kept so the
        next call retries the same claim.
        """
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        if self._sandbox is None:
            return
        sandbox, self._sandbox = self._sandbox, None
        try:
            sandbox.terminate()
        except Exception:
            # Keep the handle so a retry can terminate the same claim.
            self._sandbox = sandbox
            raise

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the `/health` endpoint until the server is ready.

        Args:
            base_url (`str`):
                URL returned by `start_container()`.
            timeout_s (`float`, *optional*, defaults to `120.0`):
                Maximum time to wait in seconds.

        Raises:
            TimeoutError: If the server does not become ready in time.
            RuntimeError: If the sandbox handle was closed or deactivated.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._sandbox is not None and not self._sandbox.is_active():
                raise RuntimeError(
                    "The agent sandbox handle is no longer active; the "
                    "environment server did not become ready."
                )
            try:
                if requests.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"Agent sandbox did not become ready within {timeout_s}s")

    def close(self) -> None:
        """Terminate the active sandbox. Also called on context-manager exit."""
        self.stop_container()


__all__ = ["AgentSandboxProvider"]
