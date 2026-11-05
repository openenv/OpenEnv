# SPDX-License-Identifier: BSD-3-Clause

"""
Tensorlake container provider for running OpenEnv environments in Tensorlake sandboxes.

Requires the ``tensorlake`` SDK: ``pip install openenv[tensorlake]``
"""

from __future__ import annotations

import logging
import os
import shlex
import time
import warnings
from typing import Any, Dict, Literal, Optional

from ._server_config import parse_openenv_app_field
from .providers import ContainerProvider

logger = logging.getLogger(__name__)

_PORT = 8000
_OPENENV_YAML = "/app/env/openenv.yaml"
# Idle timeout. Ongoing proxy traffic keeps the sandbox running, so this is
# not a wall-clock lifetime. Same default as NovitaProvider.
_DEFAULT_TIMEOUT_SECS = 3600
# How long to wait for a new sandbox to leave the "pending" state.
_STARTUP_TIMEOUT_S = 300.0
# Request timeout for readiness probes. The SDK default is 300 seconds, which
# would let one probe outlive the whole `wait_for_ready` budget.
_PROBE_TIMEOUT_S = 5.0
_MAX_LOG_CHARS = 2000
# Link-local range, including the cloud metadata endpoint 169.254.169.254
# (RFC 002 S3). The sandbox DNS resolver is not in this range.
_METADATA_CIDR = "169.254.0.0/16"
# Replaces the image CMD. Otherwise the CMD of an OpenEnv image starts a second
# server at boot that races ours for port 8000 and does not get our env_vars.
# `sh -c` ignores extra arguments, so this also works if the CMD is appended.
_KEEPALIVE_ENTRYPOINT = ["sh", "-c", "exec sleep infinity"]


def _require_secure_url(url: str) -> str:
    """Enforce https/wss transport (RFC 002 security invariant S1).

    The URL is omitted from the error because an anonymous ingress URL is a
    bearer capability that must not leak into logs.
    """
    if not url.lower().startswith("https://"):
        raise RuntimeError(
            "Tensorlake returned a non-HTTPS sandbox URL. OpenEnv requires an "
            "https/wss base_url so EnvClient traffic is encrypted."
        )
    return url


class TensorlakeProvider(ContainerProvider):
    """
    Container provider that runs environments in Tensorlake sandboxes.

    The sandbox boots from a registered Tensorlake sandbox image. Register one
    from a registry image with `tl sbx image import <ref>`, or from a
    Dockerfile with `tl sbx image create <path>`. The provider replaces the
    image `CMD` with a keep-alive command, starts the server on port 8000, and
    exposes that port on a public HTTPS URL. Anyone with the URL can reach the
    server, so you must pass `public=True`.

    Egress to the link-local range `169.254.0.0/16`, which includes the cloud
    metadata endpoint, is always blocked.

    Examples:

    ```python
    provider = TensorlakeProvider(image="echo-env", public=True)
    base_url = provider.start_container()
    provider.wait_for_ready(base_url)
    provider.stop_container()
    ```
    """

    def __init__(
        self,
        *,
        public: Literal[True] | None = None,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        cpus: Optional[float] = None,
        memory_mb: Optional[int] = None,
        timeout_secs: int = _DEFAULT_TIMEOUT_SECS,
        cmd: Optional[str] = None,
        allow_out: Optional[list[str]] = None,
        deny_out: Optional[list[str]] = None,
        surface_server_logs: bool = False,
    ):
        """
        Args:
            public (`bool`, *optional*):
                Must be `True`. Confirms that the sandbox URL is public:
                anyone with the URL can reach the server while the sandbox
                runs.
            image (`str`, *optional*):
                Registered Tensorlake sandbox image name to use when
                `start_container()` is called without an image.
            env_vars (`dict`, *optional*):
                Environment variables for the server process. Merged with the
                `env_vars` given to `start_container()`, which take precedence.
            api_key (`str`, *optional*):
                Tensorlake API key. Falls back to the `TENSORLAKE_API_KEY`
                environment variable.
            cpus (`float`, *optional*):
                CPUs for the sandbox. If `None`, Tensorlake chooses.
            memory_mb (`int`, *optional*):
                Memory for the sandbox in MB. If `None`, Tensorlake chooses.
            timeout_secs (`int`, *optional*, defaults to `3600`):
                Idle timeout in seconds. Ongoing proxy traffic keeps the
                sandbox running; this is not a wall-clock lifetime.
            cmd (`str`, *optional*):
                Shell command that starts the server on port 8000. If `None`,
                the command is built from the `app` field of
                `/app/env/openenv.yaml` inside the sandbox.
            allow_out (`list[str]`, *optional*):
                Egress allowlist of IPs, CIDR ranges, or hostnames. If set,
                all other egress is blocked. An empty list blocks all egress,
                DNS included. If `None`, egress is open except for `deny_out`.
            deny_out (`list[str]`, *optional*):
                Additional egress destinations to block. `169.254.0.0/16` is
                always blocked.
            surface_server_logs (`bool`, *optional*, defaults to `False`):
                If `True`, include the last 50 lines of server output in the
                error when the server exits. Values of `env_vars`, the API
                key and the sandbox URL are replaced with `***`
                (best-effort). If `False`, output is withheld so that secrets
                do not leak into logs.
        """
        # EnvClient cannot send Tensorlake credentials, so the exposed port
        # accepts unauthenticated traffic. The caller must opt in (RFC 002 S2).
        if public is None:
            raise ValueError(
                "TensorlakeProvider requires explicit public=True. The sandbox URL "
                "is public: anyone with the URL can reach the server while the "
                "sandbox runs."
            )
        if not public:
            raise ValueError(
                "Private Tensorlake sandbox ports are not supported yet because "
                "EnvClient cannot send Tensorlake credentials. Use public=True."
            )
        self._image = image
        self._env_vars = env_vars
        self._api_key = api_key or os.environ.get("TENSORLAKE_API_KEY")
        self._cpus = cpus
        self._memory_mb = memory_mb
        self._timeout_secs = timeout_secs
        self._cmd = cmd
        self._allow_out = allow_out
        self._deny_out = [_METADATA_CIDR, *(deny_out or [])]
        self._surface_server_logs = surface_server_logs
        self._secret_values: list[str] = []
        self._client: Any = None
        self._sandbox: Any = None
        self._sandbox_id: Optional[str] = None
        self._pid: Optional[int] = None
        self._probe_config: Dict[str, Any] = {}

    def start_container(
        self,
        image: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Tensorlake sandbox, start the server, and expose port 8000.

        Args:
            image (`str`, *optional*):
                Registered Tensorlake sandbox image name. May be omitted when
                given to the constructor.
            port (`int`, *optional*):
                Must be `None` or `8000`.
            env_vars (`dict`, *optional*):
                Environment variables for the server process. Merged over the
                constructor `env_vars`.
            **kwargs:
                `cmd` (`str`) overrides the server command.

        Returns:
            `str`: Public HTTPS URL of port 8000 in the sandbox.

        Raises:
            RuntimeError: If a sandbox from an earlier call is still active,
                or if the sandbox or the server fails to start. SDK error
                details are withheld.
            TimeoutError: If the sandbox stays pending for too long.
        """
        if self._sandbox_id is not None:
            raise RuntimeError(
                "A Tensorlake sandbox is already active. Call stop_container() first."
            )
        if port is not None and port != _PORT:
            raise ValueError(
                f"TensorlakeProvider only supports port {_PORT} (got {port})."
            )
        effective_image = image or self._image
        if effective_image is None:
            raise ValueError(
                "TensorlakeProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        # AutoEnv always passes env_vars={}, so merge instead of replace.
        effective_env_vars = {**(self._env_vars or {}), **(env_vars or {})}
        cmd = kwargs.pop("cmd", None) or self._cmd
        # AutoEnv always forwards wait_timeout; it does not apply here.
        kwargs.pop("wait_timeout", None)
        if kwargs:
            raise ValueError(
                f"Unsupported TensorlakeProvider options: {', '.join(sorted(kwargs))}"
            )

        try:
            from tensorlake.sandbox import (
                sandbox_url_from_ingress_endpoint,
                SandboxClient,
            )
        except ImportError as exc:
            raise ImportError(
                "TensorlakeProvider requires the tensorlake SDK. "
                "Install it with: pip install openenv[tensorlake]"
            ) from exc

        self._secret_values = [
            v for v in (self._api_key, *effective_env_vars.values()) if v
        ]
        try:
            # SandboxClient is deprecated, but it is the only public API that
            # allocates a sandbox and returns its ID before connecting, and
            # that deletes a sandbox by ID. Sandbox.create() can raise after
            # allocation and then nothing holds the ID (RFC 002 invariant 5).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._client = SandboxClient(api_key=self._api_key)
            info = self._client.create(
                image=effective_image,
                cpus=self._cpus,
                memory_mb=self._memory_mb,
                timeout_secs=self._timeout_secs,
                entrypoint=_KEEPALIVE_ENTRYPOINT,
                # Tensorlake treats an empty allow_out as "no allowlist".
                allow_internet_access=self._allow_out != [],
                allow_out=self._allow_out,
                deny_out=self._deny_out,
            )
            # Record the ID first so that any failure below still deletes it.
            self._sandbox_id = info.sandbox_id
            deadline = time.monotonic() + _STARTUP_TIMEOUT_S
            while info.status == "pending":
                if time.monotonic() >= deadline:
                    raise TimeoutError("Tensorlake sandbox startup timed out.")
                time.sleep(0.5)
                info = self._client.get(self._sandbox_id)
            if info.status != "running":
                raise RuntimeError("Tensorlake sandbox failed to start.")
            self._sandbox = self._client.connect(self._sandbox_id)

            if cmd is None:
                cmd = self._discover_server_cmd()
            process = self._sandbox.start_process(
                "sh", ["-c", cmd], env=effective_env_vars or None
            )
            self._pid = process.pid

            info = self._sandbox.update(
                allow_unauthenticated_access=True, exposed_ports=[_PORT]
            )
            if not info.ingress_endpoint or not info.sandbox_url:
                raise RuntimeError("Tensorlake did not return the sandbox endpoints.")
            # Enough to build a second SDK handle with a short request timeout
            # for readiness probes. See wait_for_ready().
            self._probe_config = {
                "identifier": self._sandbox_id,
                "proxy_url": info.sandbox_url,
                "routing_hint": info.routing_hint,
                "api_key": self._api_key,
            }
            base_url = _require_secure_url(
                sandbox_url_from_ingress_endpoint(
                    info.ingress_endpoint, info.sandbox_id, _PORT
                )
            )
        except Exception as exc:
            # Do not let a cleanup failure hide the original error.
            try:
                self.stop_container()
            except Exception:
                logger.warning(
                    "Could not terminate the Tensorlake sandbox after a failed "
                    "start. Call close() again to retry."
                )
            # Errors raised above carry no SDK details, so they may propagate.
            # SDK errors can quote the sandbox URL, the API key or server
            # output, so they are replaced by a constant message (RFC 002 S2/S4).
            if isinstance(exc, (ValueError, RuntimeError, TimeoutError)):
                raise
            raise RuntimeError(
                "Tensorlake sandbox startup failed. SDK details are withheld."
            ) from None
        # The anonymous URL is a bearer capability: redact it from surfaced
        # server output too.
        self._secret_values += [base_url, base_url.replace("https://", "wss://", 1)]
        return base_url

    def _discover_server_cmd(self) -> str:
        """Build the server command from `openenv.yaml` inside the sandbox."""
        try:
            content = self._sandbox.read_file(_OPENENV_YAML).value.decode()
        except Exception:
            # Not chained: the SDK error may quote the sandbox URL.
            raise ValueError(
                f"Could not read {_OPENENV_YAML} in the sandbox. "
                "Pass cmd= to TensorlakeProvider or start_container()."
            ) from None
        app = parse_openenv_app_field(content)
        if app is None:
            raise ValueError(
                f"{_OPENENV_YAML} has no 'app' field. "
                "Pass cmd= to TensorlakeProvider or start_container()."
            )
        return (
            f"cd /app/env && python -m uvicorn {shlex.quote(app)} "
            f"--host 0.0.0.0 --port {_PORT}"
        )

    def _server_exited_message(self, sandbox: Any) -> str:
        """Build the error for a dead server. Output is withheld by default."""
        if not self._surface_server_logs:
            return (
                "Server process exited. Output is withheld to avoid leaking "
                "secrets. Pass surface_server_logs=True to include it."
            )
        from tensorlake.sandbox import SandboxError

        try:
            log = "\n".join(sandbox.get_output(self._pid).lines[-50:])
        except SandboxError:
            return "Server process exited. Could not read its output."
        # Longest first, so a short value cannot split a longer one (for
        # example the sandbox ID inside the URL) before it is scrubbed.
        for value in sorted(self._secret_values, key=len, reverse=True):
            log = log.replace(value, "***")
        if len(log) > _MAX_LOG_CHARS:
            log = "...(truncated)...\n" + log[-_MAX_LOG_CHARS:]
        return f"Server process exited.\nLog (redacted, best-effort):\n{log}"

    def stop_container(self) -> None:
        """
        Delete the Tensorlake sandbox.

        If deletion fails, the sandbox ID is kept and the next call retries.
        A sandbox that is already gone counts as stopped.

        Raises:
            RuntimeError: If deletion fails. SDK error details are withheld.
        """
        if self._sandbox_id is None:
            return
        from tensorlake.sandbox import SandboxNotFoundError

        try:
            self._client.delete(self._sandbox_id)
        except SandboxNotFoundError:
            pass
        except Exception:
            raise RuntimeError(
                "Tensorlake sandbox termination failed. Retry close(). "
                "SDK details are withheld."
            ) from None
        sandbox, self._sandbox = self._sandbox, None
        self._sandbox_id = None
        self._pid = None
        self._probe_config = {}
        self._secret_values = []
        if sandbox is not None:
            sandbox.close()

    def close(self) -> None:
        """Delete the active sandbox. Also called on context-manager exit."""
        self.stop_container()

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
            RuntimeError: If the server process exits.
        """
        import requests

        deadline = time.monotonic() + timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            # Check the process first: a 200 from another process on port 8000
            # must not hide a dead server.
            if self._probe_config:
                from tensorlake.sandbox import Sandbox, SandboxError

                # A fresh handle, so that the probe is bounded by the
                # remaining budget instead of the SDK's 300 second default.
                # Exiting the context closes the handle; the sandbox stays.
                # Building, using and closing the handle can all raise SDK
                # errors that quote the sandbox URL or the API key. Temporary
                # errors are common during a cold start, so retry until the
                # deadline instead of surfacing them (RFC 002 S2/S4).
                exited_message = None
                try:
                    with Sandbox(
                        **self._probe_config,
                        request_timeout=min(_PROBE_TIMEOUT_S, remaining),
                    ) as probe:
                        if probe.get_process(self._pid).status != "running":
                            exited_message = self._server_exited_message(probe)
                except SandboxError:
                    pass
                if exited_message is not None:
                    raise RuntimeError(exited_message)
            # The probe may have spent part of the budget.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = requests.get(
                    f"{base_url}/health", timeout=min(_PROBE_TIMEOUT_S, remaining)
                )
                healthy = response.status_code == 200
            except requests.RequestException:
                healthy = False
            # A late 200 does not count: the caller's budget is spent.
            if healthy and time.monotonic() < deadline:
                return
            time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))

        raise TimeoutError(
            f"Tensorlake sandbox did not become ready within {timeout_s}s"
        )
