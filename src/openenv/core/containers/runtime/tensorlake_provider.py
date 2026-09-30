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
# Bound the sandbox lifetime (RFC 002 S6). Same default as NovitaProvider.
_DEFAULT_TIMEOUT_SECS = 3600
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


def _require_public_opt_in(public: Optional[bool]) -> None:
    """Require explicit public ingress opt-in (RFC 002 invariant S2).

    `EnvClient` cannot send Tensorlake credentials, so the exposed port must
    accept unauthenticated traffic. The caller must accept this explicitly.
    """
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


class TensorlakeProvider(ContainerProvider):
    """
    Container provider that runs environments in Tensorlake sandboxes.

    The sandbox boots from a registered Tensorlake sandbox image. Register one
    from a registry image with `tl sbx image import <ref>`, or from a
    Dockerfile with `tl sbx image create <path>`. The provider replaces the
    image `CMD` with a keep-alive command, starts the server on port 8000, and
    exposes that port on a public HTTPS URL. Anyone
    with the URL can reach the server, so you must pass `public=True`.

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
                Sandbox lifetime in seconds.
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
                error when the server exits. Values of `env_vars` are replaced
                with `***` (best-effort). If `False`, output is withheld so
                that secrets do not leak into logs.
        """
        _require_public_opt_in(public)
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
        self._sandbox: Any = None
        self._sandbox_id: Optional[str] = None
        self._pid: Optional[int] = None

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
            RuntimeError: If a sandbox from an earlier call is still active.
        """
        if self._sandbox is not None or self._sandbox_id is not None:
            raise RuntimeError(
                "A Tensorlake sandbox is already active. Call stop_container() first."
            )
        if port is not None and port != _PORT:
            raise ValueError(
                f"TensorlakeProvider only supports port {_PORT} (got {port})."
            )
        effective_image = image if image is not None else self._image
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
            from tensorlake.sandbox import Sandbox, sandbox_url_from_ingress_endpoint
        except ImportError as exc:
            raise ImportError(
                "TensorlakeProvider requires the tensorlake SDK. "
                "Install it with: pip install openenv[tensorlake]"
            ) from exc

        self._sandbox = Sandbox.create(
            image=effective_image,
            cpus=self._cpus,
            memory_mb=self._memory_mb,
            timeout_secs=self._timeout_secs,
            entrypoint=_KEEPALIVE_ENTRYPOINT,
            # Tensorlake treats an empty allow_out as "no allowlist".
            allow_internet_access=self._allow_out != [],
            allow_out=self._allow_out,
            deny_out=self._deny_out,
            api_key=self._api_key,
        )
        self._secret_values = [v for v in effective_env_vars.values() if v]
        try:
            self._sandbox_id = self._sandbox.sandbox_id
            if cmd is None:
                cmd = self._discover_server_cmd()
            process = self._sandbox.start_process(
                "sh", ["-c", cmd], env=effective_env_vars or None
            )
            self._pid = process.pid

            info = self._sandbox.update(
                allow_unauthenticated_access=True, exposed_ports=[_PORT]
            )
            if not info.ingress_endpoint:
                raise RuntimeError("Tensorlake did not return an ingress endpoint.")
            return _require_secure_url(
                sandbox_url_from_ingress_endpoint(
                    info.ingress_endpoint, info.sandbox_id, _PORT
                )
            )
        except Exception:
            # Do not let a cleanup failure hide the original error.
            try:
                self.stop_container()
            except Exception:
                logger.warning(
                    "Could not terminate Tensorlake sandbox %s after a failed start.",
                    self._sandbox_id,
                    exc_info=True,
                )
            raise

    def _discover_server_cmd(self) -> str:
        """Build the server command from `openenv.yaml` inside the sandbox."""
        try:
            content = self._sandbox.read_file(_OPENENV_YAML).value.decode()
        except Exception as exc:
            raise ValueError(
                f"Could not read {_OPENENV_YAML} in the sandbox. "
                "Pass cmd= to TensorlakeProvider or start_container()."
            ) from exc
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

    def _server_exited_message(self) -> str:
        """Build the error for a dead server. Output is withheld by default."""
        if not self._surface_server_logs:
            return (
                "Server process exited. Output is withheld to avoid leaking "
                "secrets. Pass surface_server_logs=True to include it."
            )
        from tensorlake.sandbox import SandboxError

        try:
            log = "\n".join(self._sandbox.get_output(self._pid).lines[-50:])
        except SandboxError:
            return "Server process exited. Could not read its output."
        for value in self._secret_values:
            log = log.replace(value, "***")
        if len(log) > _MAX_LOG_CHARS:
            log = "...(truncated)...\n" + log[-_MAX_LOG_CHARS:]
        return f"Server process exited.\nLog (redacted, best-effort):\n{log}"

    def stop_container(self) -> None:
        """
        Terminate the Tensorlake sandbox.

        If termination fails, the sandbox ID is kept. The next call deletes
        the sandbox by ID. A sandbox that is already gone counts as stopped.
        """
        if self._sandbox is None and self._sandbox_id is None:
            return
        from tensorlake.sandbox import SandboxClient, SandboxNotFoundError

        try:
            if self._sandbox is not None:
                sandbox, self._sandbox, self._pid = self._sandbox, None, None
                # Sandbox.terminate() is one-shot: it drops its lifecycle client
                # before the delete call, so a retry on the same handle does
                # nothing.
                sandbox.terminate()
            else:
                # Do not use Sandbox.connect(): it raises SandboxNotRoutableError
                # for a sandbox that is not running, for example one that is
                # still terminating. SandboxClient is deprecated, but it is the
                # only public API that deletes a sandbox by ID.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    client = SandboxClient(api_key=self._api_key)
                client.delete(self._sandbox_id)
        except SandboxNotFoundError:
            pass
        self._sandbox_id = None
        self._secret_values = []

    def close(self) -> None:
        """Terminate the active sandbox. Also called on context-manager exit."""
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

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # Check the process first: a 200 from another process on port 8000
            # must not hide a dead server.
            if self._sandbox is not None and self._pid is not None:
                from tensorlake.sandbox import SandboxError

                try:
                    status = self._sandbox.get_process(self._pid).status
                except SandboxError:
                    # Temporary SDK or proxy errors are common during a cold start.
                    status = "running"
                if status != "running":
                    raise RuntimeError(self._server_exited_message())

            try:
                if requests.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                    return
            except requests.RequestException:
                pass

            time.sleep(1.0)

        raise TimeoutError(
            f"Tensorlake sandbox did not become ready within {timeout_s}s"
        )
