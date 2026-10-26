# SPDX-License-Identifier: BSD-3-Clause

"""
Modal container provider for running OpenEnv environments in Modal sandboxes.

Requires the ``modal`` SDK: ``pip install modal>=1.4``

The provider boots an OpenEnv server inside a Modal sandbox, exposes it on an
encrypted tunnel, and returns an ``https://`` URL that ``EnvClient`` connects to
over ``wss://``.

Supports both the stable Sandbox API (``modal.Sandbox.create``) and the beta
Sandbox v2 API (``modal.Sandbox._experimental_create``). Sandbox v2 is opt-in
via ``use_sandbox_v2=True`` because it is an experimental, private SDK feature;
see https://modal.com/docs/guide/sandbox-v2 .
"""

from __future__ import annotations

import logging
import shlex
import time
from typing import Any, Dict, Optional

from ._server_config import parse_dockerfile_cmd, parse_openenv_app_field
from .providers import ContainerProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODAL_PORT = 8000

# Modal's default per-container resource requests. We pass configured cpu/memory
# as the *limit* half of Modal's ``(request, limit)`` tuple so they cap usage
# rather than reserving (and billing for) the full amount. See
# https://modal.com/docs/guide/resources .
_DEFAULT_CPU_REQUEST = 0.125
_DEFAULT_MEMORY_REQUEST = 128


def _require_secure_url(url: str) -> str:
    """Enforce https/wss transport (RFC 002 security invariant S1).

    ``EnvClient`` derives its WebSocket URL from this base URL, so a plaintext
    URL would become a cleartext ``ws://`` connection. The offending URL is
    deliberately omitted from the error because a Modal tunnel URL is a bearer
    capability that must not leak into logs.
    """
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise RuntimeError(
            "Modal sandbox returned a non-HTTPS tunnel URL. OpenEnv requires an "
            "https/wss base_url so EnvClient traffic is encrypted. Refusing to "
            "connect over plaintext."
        )
    return url


class _DefaultModalAdapter:
    """Thin adapter over the ``modal`` SDK.

    The provider talks to this private adapter instead of spreading SDK details
    through its own logic; tests inject a duck-typed fake in its place. Keeping
    the SDK surface here also localizes the feature check for the private
    Sandbox v2 (``_experimental_create``) API.
    """

    def __init__(self, *, app_name: str, use_sandbox_v2: bool):
        import modal

        self._modal = modal
        self._app_name = app_name
        self._use_sandbox_v2 = use_sandbox_v2

        # Sandbox v2 rides on a private SDK entry point that is not present in
        # every modal release. Fail fast at construction with clear guidance
        # rather than at start_container time with an AttributeError.
        if use_sandbox_v2 and not hasattr(modal.Sandbox, "_experimental_create"):
            raise RuntimeError(
                "use_sandbox_v2=True requires modal.Sandbox._experimental_create, "
                "which is not available in the installed modal SDK. Upgrade modal, "
                "or use the stable Sandbox API (use_sandbox_v2=False). Sandbox v2 "
                "is an experimental feature; contact support@modal.com for access."
            )

    def image_from_registry(self, tag: str) -> Any:
        return self._modal.Image.from_registry(tag)

    def image_from_dockerfile(self, dockerfile_path: str, context_dir: str) -> Any:
        return self._modal.Image.from_dockerfile(
            dockerfile_path, context_dir=context_dir
        )

    def create_sandbox(
        self,
        *,
        image: Any,
        encrypted_ports: list[int],
        timeout: int,
        env: Optional[dict[str, str]],
        extra: dict[str, Any],
    ) -> Any:
        app = self._modal.App.lookup(self._app_name, create_if_missing=True)
        kwargs: Dict[str, Any] = {
            "app": app,
            "image": image,
            "encrypted_ports": encrypted_ports,
            "timeout": timeout,
            **extra,
        }
        if env:
            kwargs["env"] = dict(env)

        # The sandbox is started with a keep-alive entrypoint; the server
        # command is launched via exec afterwards (see ModalProvider).
        if self._use_sandbox_v2:
            return self._modal.Sandbox._experimental_create(
                "sleep", "infinity", **kwargs
            )
        return self._modal.Sandbox.create("sleep", "infinity", **kwargs)

    def exec(self, sandbox: Any, command: str, *, timeout: int = 10) -> str:
        """Run *command* through a shell inside *sandbox* and return its stdout."""
        proc = sandbox.exec("bash", "-c", command, timeout=timeout)
        try:
            out = proc.stdout.read()
        except Exception:
            out = ""
        proc.wait()
        return out or ""

    def tunnel_url(self, sandbox: Any, port: int) -> str:
        return sandbox.tunnels()[port].url

    def terminate(self, sandbox: Any) -> None:
        sandbox.terminate()


class ModalProvider(ContainerProvider):
    """
    Container provider that runs environments in Modal sandboxes.

    ``start_container``'s ``image`` is either a registry tag
    (``"echo-env:latest"``) or a ``"dockerfile:<path>"`` reference returned by
    :meth:`image_from_dockerfile`. The server is exposed on an encrypted Modal
    tunnel and the returned ``https://`` URL is what ``EnvClient`` connects to
    over ``wss://``.

    The environment runs untrusted code, so the provider is secure by default:
    it enforces https/wss transport, treats the tunnel URL as a bearer secret
    (never interpolated into errors), and never surfaces raw sandbox output
    unless ``surface_server_logs=True``.

    Only one sandbox is active per provider: calling ``start_container`` again
    before ``stop_container()``/``close()`` raises ``RuntimeError`` rather than
    orphaning the running sandbox. ``close()`` (and context-manager exit) stops
    the active sandbox.

    Example:
        ```python
        with ModalProvider(app_name="openenv", cpu=2.0, memory=4096) as provider:
            image = ModalProvider.image_from_dockerfile(
                "envs/echo_env/server/Dockerfile"
            )
            base_url = provider.start_container(image)
            provider.wait_for_ready(base_url)
        # sandbox terminated on exit
        ```

    Sandbox v2 (beta) is opt-in:
        ```python
        provider = ModalProvider(app_name="openenv", use_sandbox_v2=True)
        ```
    """

    _dockerfile_registry: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        *,
        image: str | None = None,
        env_vars: dict[str, str] | None = None,
        app_name: str = "openenv",
        use_sandbox_v2: bool = False,
        timeout: int = 300,
        cmd: str | None = None,
        cpu: float | None = None,
        memory: int | None = None,
        surface_server_logs: bool = False,
        _adapter: Any = None,
    ):
        """
        Args:
            image (`str`, *optional*):
                Registry image tag or ``"dockerfile:<path>"`` source to use
                when ``start_container()`` is called without an image.
            env_vars (`dict`, *optional*):
                Environment variables to use when ``start_container()`` is
                called without explicit ``env_vars``.
            app_name (`str`, *optional*, defaults to `"openenv"`):
                Modal app name the sandbox is created under. Looked up (and
                created if missing) via ``modal.App.lookup``.
            use_sandbox_v2 (`bool`, *optional*, defaults to `False`):
                When `True`, sandboxes are created via the beta
                ``modal.Sandbox._experimental_create`` API (Sandbox v2). This is
                an experimental, private SDK feature and is therefore off by
                default; a feature check fails fast at construction if the
                installed SDK lacks it. See https://modal.com/docs/guide/sandbox-v2 .
            timeout (`int`, *optional*, defaults to `300`):
                Maximum sandbox lifetime in seconds.
            cmd (`str`, *optional*):
                Shell command to start the server inside the sandbox. When
                omitted, the command is auto-discovered from ``openenv.yaml``
                (falling back to the Dockerfile ``CMD``).
            cpu (`float`, *optional*):
                Hard CPU-core limit for the sandbox. Passed as the limit half of
                Modal's ``(request, limit)`` tuple (request stays at Modal's
                default), so it caps usage rather than reserving cores. When
                `None`, Modal's default applies.
            memory (`int`, *optional*):
                Hard memory limit in MiB for the sandbox (containers exceeding it
                are OOM-killed). Passed as the limit half of Modal's
                ``(request, limit)`` tuple. When `None`, Modal's default applies.
            surface_server_logs (`bool`, *optional*, defaults to `False`):
                When `False` (default), captured sandbox output is withheld from
                raised errors so secrets the workload printed cannot leak into
                orchestrator/CI logs. When `True`, a best-effort redacted,
                length-bounded excerpt is included in startup-crash errors.
        """
        self._image = image
        self._env_vars = env_vars
        self._app_name = app_name
        self._use_sandbox_v2 = use_sandbox_v2
        self._timeout = timeout
        self._cmd = cmd
        self._cpu = cpu
        self._memory = memory
        self._surface_server_logs = surface_server_logs
        self._sandbox: Any = None
        self._base_url: str | None = None
        # Injected env-var values, used to scrub captured server output before
        # it is ever surfaced in an error.
        self._redact_values: set[str] = set()

        if _adapter is None:
            # Import eagerly (inside the adapter) so SDK/configuration errors —
            # including the Sandbox v2 feature check — surface at construction.
            self._adapter: Any = _DefaultModalAdapter(
                app_name=app_name, use_sandbox_v2=use_sandbox_v2
            )
        else:
            self._adapter = _adapter

        if use_sandbox_v2:
            logger.info(
                "Using Modal Sandbox v2 (experimental). This feature must be "
                "explicitly enabled. Contact support@modal.com to get access."
            )

    def _discover_server_cmd(self, port: int = _DEFAULT_MODAL_PORT) -> str:
        """Discover the server command from ``openenv.yaml`` inside the sandbox.

        Finds the file, reads the ``app`` field, and constructs a command
        of the form ``cd <env_root> && python -m uvicorn <app> --host 0.0.0.0 --port <port>``.

        Raises:
            ValueError: If ``openenv.yaml`` is not found or lacks an ``app`` field.
        """
        yaml_path = self._find_openenv_yaml()
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml inside the sandbox. "
                "Pass an explicit cmd= to ModalProvider or start_container()."
            )

        content = self._adapter.exec(self._sandbox, f"cat {shlex.quote(yaml_path)}")
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass an explicit cmd= to ModalProvider or start_container()."
            )

        # The directory containing openenv.yaml is the env root
        env_root = yaml_path.rsplit("/", 1)[0]
        return (
            f"cd {shlex.quote(env_root)} && "
            f"python -m uvicorn {shlex.quote(app)} --host 0.0.0.0 --port {port}"
        )

    def _find_openenv_yaml(self) -> str | None:
        """Locate ``openenv.yaml`` inside the sandbox.

        Tries the modern layout path ``/app/env/openenv.yaml`` first,
        then falls back to a ``find`` command for the old layout.
        """
        # Fast path: modern Dockerfile layout
        out = self._adapter.exec(
            self._sandbox, "test -f /app/env/openenv.yaml && echo found"
        )
        if "found" in (out or ""):
            return "/app/env/openenv.yaml"

        # Fallback: search for it (redirect stderr so error messages
        # like "No such file or directory" don't get mistaken for paths).
        path = self._adapter.exec(
            self._sandbox,
            "find /app -maxdepth 4 -name openenv.yaml -print -quit 2>/dev/null",
        ).strip()
        if path and path.startswith("/"):
            return path

        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> str | None:
        """Extract the ``app`` value from raw openenv.yaml content.

        Uses PyYAML to handle comments, quotes, and nested keys correctly.
        """
        return parse_openenv_app_field(yaml_content)

    @staticmethod
    def _parse_dockerfile_cmd(dockerfile_content: str) -> str | None:
        """Extract the server command from the last ``CMD`` in a Dockerfile.

        Handles exec form (``CMD ["prog", "arg"]``) and shell form
        (``CMD prog arg``).  When a Dockerfile has multiple ``CMD``
        instructions (e.g. multi-stage builds), the last one wins - same
        semantics as Docker itself.  Lines where ``CMD`` appears inside a
        comment are ignored.

        Returns:
            The command as a single string, or ``None`` if no ``CMD`` found.
        """
        return parse_dockerfile_cmd(dockerfile_content)

    @classmethod
    def image_from_dockerfile(
        cls,
        dockerfile_path: str,
        context_dir: str | None = None,
    ) -> str:
        """Validate a Dockerfile and return a ``dockerfile:`` URI for
        :meth:`start_container`.

        Eagerly validates the Dockerfile (existence, COPY sources) and stores
        its content in an internal registry.  The actual ``modal.Image`` is
        created later inside ``start_container``.

        Args:
            dockerfile_path (`str`):
                Path to the Dockerfile on disk.
            context_dir (`str`, *optional*):
                Build context directory.  Defaults to the Dockerfile's
                grandparent directory, matching the ``openenv init``
                convention where Dockerfiles live in
                ``<env>/server/Dockerfile`` and the build context is
                ``<env>/``.  Pass explicitly for non-standard layouts
                (e.g. ``context_dir="."`` for repo-root contexts).

        Returns:
            `str`: A ``"dockerfile:<abs_path>"`` string to pass to
            ``start_container``.

        Raises:
            FileNotFoundError: If *dockerfile_path* does not exist.
            ValueError: If *context_dir* is given but does not exist,
                or if COPY sources in the Dockerfile cannot be found
                under the resolved context directory.
        """
        import pathlib
        import re

        src = pathlib.Path(dockerfile_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        if context_dir is not None:
            ctx = pathlib.Path(context_dir)
            if not ctx.is_dir():
                raise ValueError(f"context_dir does not exist: {context_dir}")
        else:
            # Default: grandparent of the Dockerfile, matching the
            # openenv init layout (<env>/server/Dockerfile -> <env>/).
            ctx = src.parent.parent

        content = src.read_text()

        # Validate that COPY sources exist under the context directory.
        # This catches mismatches early (e.g. a Dockerfile expecting repo
        # root as context when we defaulted to the env directory).
        for line in content.splitlines():
            m = re.match(r"^\s*COPY\s+(?!--from=)(\S+)\s+", line, re.IGNORECASE)
            if not m:
                continue
            copy_src = m.group(1)
            if copy_src.startswith("/"):
                continue
            resolved = ctx / copy_src
            if not resolved.exists() and not any(ctx.glob(copy_src)):
                raise ValueError(
                    f"Dockerfile COPY source '{copy_src}' not found "
                    f"under context_dir '{ctx}'. This Dockerfile may "
                    f"expect a different build context (e.g. the repo "
                    f"root). Pass context_dir explicitly."
                )

        # Parse CMD from the Dockerfile so start_container can use it as a
        # fallback when openenv.yaml is unavailable.
        parsed_cmd = cls._parse_dockerfile_cmd(content)

        cls._dockerfile_registry[str(src)] = {
            "dockerfile_path": str(src),
            "context_dir": str(ctx),
            "server_cmd": parsed_cmd,
        }

        return f"dockerfile:{src}"

    def _build_image(self, image: str) -> Any:
        """Build the ``modal.Image`` for *image* (registry tag or dockerfile:)."""
        if image.startswith("dockerfile:"):
            dockerfile_path = image[len("dockerfile:") :]
            meta = self._dockerfile_registry.get(dockerfile_path)
            if meta is None:
                raise ValueError(
                    f"No registered Dockerfile metadata for {dockerfile_path}. "
                    "Call ModalProvider.image_from_dockerfile() first."
                )
            return self._adapter.image_from_dockerfile(
                meta["dockerfile_path"], meta["context_dir"]
            )

        # Plain registry tag (e.g. "echo-env:latest").
        return self._adapter.image_from_registry(image)

    def start_container(
        self,
        image: str | None = None,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Modal sandbox from a Docker image or Dockerfile.

        The sandbox is started with a keep-alive process and the server
        command is launched via ``exec`` afterwards, mirroring the discovery
        flow used by other cloud providers. The server command is resolved in
        order:

        1. Explicit ``cmd`` passed to the constructor.
        2. ``cmd`` key in ``**kwargs`` (popped before forwarding).
        3. Auto-discovered from ``openenv.yaml`` inside the sandbox.
        4. ``CMD`` parsed from the Dockerfile (when *image* came from
           ``image_from_dockerfile``).

        Args:
            image (`str`, *optional*):
                Registry image tag (e.g. ``"echo-env:latest"``) or
                ``"dockerfile:<path>"`` returned by
                :meth:`image_from_dockerfile`. May be omitted when supplied to
                the constructor.
            port (`int`, *optional*):
                Must be ``None`` or ``8000``. Modal exposes port 8000 via an
                encrypted tunnel; other ports raise ``ValueError``.
            env_vars (`dict`, *optional*):
                Environment variables forwarded to the sandbox.
            **kwargs:
                ``cmd`` (`str`) to override the server command; any remaining
                keyword arguments are forwarded to ``modal.Sandbox.create``.

        Returns:
            `str`: HTTPS tunnel URL for the sandbox (base_url).
        """
        if self._sandbox is not None:
            raise RuntimeError(
                "ModalProvider already has an active sandbox. Call "
                "stop_container() (or close()) before starting another — a "
                "second start would orphan the running sandbox."
            )

        if port is not None and port != _DEFAULT_MODAL_PORT:
            raise ValueError(
                f"ModalProvider only supports port {_DEFAULT_MODAL_PORT} "
                f"(got {port}). The Modal tunnel routes to port "
                f"{_DEFAULT_MODAL_PORT} inside the sandbox."
            )

        effective_image = image if image is not None else self._image
        if effective_image is None:
            raise ValueError(
                "ModalProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        effective_env_vars = self._env_vars if env_vars is None else env_vars

        # Resolve the server command (may be None; discovery happens after
        # sandbox creation when we can inspect the filesystem).
        cmd = kwargs.pop("cmd", None) or self._cmd

        # CMD parsed from Dockerfile (populated for "dockerfile:" images).
        parsed_cmd: str | None = None
        if effective_image.startswith("dockerfile:"):
            meta = self._dockerfile_registry.get(effective_image[len("dockerfile:") :])
            if meta is not None:
                parsed_cmd = meta.get("server_cmd")

        modal_image = self._build_image(effective_image)

        extra: Dict[str, Any] = dict(kwargs)
        if self._cpu is not None:
            extra["cpu"] = (_DEFAULT_CPU_REQUEST, self._cpu)
        if self._memory is not None:
            extra["memory"] = (_DEFAULT_MEMORY_REQUEST, self._memory)

        # Record injected secret values so captured server output can be
        # scrubbed before it is ever surfaced in an error.
        self._redact_values = {
            value for value in (effective_env_vars or {}).values() if value
        }

        # A create failure created nothing, so just drop the recorded secrets
        # and re-raise (the double-start guard above guarantees there is no
        # pre-existing sandbox to delete).
        try:
            self._sandbox = self._adapter.create_sandbox(
                image=modal_image,
                encrypted_ports=[_DEFAULT_MODAL_PORT],
                timeout=self._timeout,
                env=effective_env_vars,
                extra=extra,
            )
        except Exception:
            self._redact_values = set()
            raise

        try:
            # Discover server command from openenv.yaml if not explicitly set.
            if cmd is None:
                try:
                    cmd = self._discover_server_cmd()
                except ValueError:
                    # Fall back to CMD parsed from Dockerfile (if available).
                    if parsed_cmd:
                        cmd = parsed_cmd
                    else:
                        raise

            # Launch the server in the background. Write the PID so we can
            # check whether the process crashed in wait_for_ready().
            escaped_cmd = shlex.quote(cmd)
            self._adapter.exec(
                self._sandbox,
                f"nohup bash -c {escaped_cmd} > /tmp/openenv-server.log 2>&1 &"
                " echo $! > /tmp/openenv-server.pid",
            )

            # Resolve the public tunnel URL for port 8000.
            self._base_url = _require_secure_url(
                self._adapter.tunnel_url(self._sandbox, _DEFAULT_MODAL_PORT)
            )
        except Exception:
            # A cleanup failure here must not mask the original error: swallow
            # any exception from stop_container() so the root cause propagates.
            try:
                self.stop_container()
            except Exception:
                pass
            raise

        return self._base_url

    def stop_container(self) -> None:
        """Terminate the Modal sandbox."""
        if self._sandbox is None:
            # Still drop any injected secret values recorded by a failed start.
            self._redact_values = set()
            return

        try:
            self._adapter.terminate(self._sandbox)
        finally:
            self._sandbox = None
            self._base_url = None
            self._redact_values = set()

    def close(self) -> None:
        """Stop the active sandbox.

        Overrides the base no-op so a caller holding a bare ``ContainerProvider``
        reference can release the sandbox polymorphically (also invoked on
        context-manager exit). ``ModalProvider`` holds no separate SDK client, so
        this is equivalent to ``stop_container()``.
        """
        self.stop_container()

    @property
    def base_url(self) -> str:
        """URL returned by the last ``start_container``."""
        if self._base_url is None:
            raise RuntimeError(
                "ModalProvider has no active base_url. Start the provider "
                "before reading base_url."
            )
        return self._base_url

    def _redact(self, text: str, *, max_chars: int = 2000) -> str:
        """Scrub injected secret values and bound length before surfacing output.

        Replaces any injected env-var value with `***` and keeps only the tail.
        This is best-effort (exact-match only), which is why server output is
        withheld entirely unless ``surface_server_logs=True``.
        """
        redacted = text or ""
        for value in self._redact_values:
            redacted = redacted.replace(value, "***")
        if len(redacted) > max_chars:
            redacted = "...(truncated)...\n" + redacted[-max_chars:]
        return redacted

    def _server_died_message(self) -> str:
        """Build the startup-crash error, secure by default.

        Untrusted code can print secrets then force a crash to exfiltrate them
        through the exception (which lands in orchestrator/CI logs), so sandbox
        output is excluded unless ``surface_server_logs=True``, in which case a
        best-effort redacted, bounded excerpt is included.
        """
        base = (
            "Modal sandbox server process died during startup. Server output is "
            "not surfaced to avoid leaking secrets injected into the sandbox; "
            "retrieve /tmp/openenv-server.log from the sandbox out of band, or "
            "construct the provider with surface_server_logs=True to include a "
            "redacted excerpt."
        )
        if not self._surface_server_logs or self._sandbox is None:
            return base

        log = self._redact(
            self._adapter.exec(self._sandbox, "cat /tmp/openenv-server.log 2>/dev/null")
        )
        return (
            "Modal sandbox server process died during startup. The excerpt below "
            "is the sandbox server output with injected secret values redacted "
            "(best-effort); it may still contain secrets the workload printed "
            f"by other means.\nLog (redacted):\n{log}"
        )

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the /health endpoint until the sandbox is ready.

        Uses a longer default timeout (120s) than local Docker providers
        because Modal sandboxes may have cold-start latency.

        Args:
            base_url (`str`):
                Tunnel URL returned by ``start_container()``.
            timeout_s (`float`, *optional*, defaults to `120.0`):
                Maximum seconds to wait.

        Raises:
            TimeoutError: If the sandbox doesn't become ready in time.
            RuntimeError: If the server process died (detected via PID check).
        """
        import requests

        health_url = f"{base_url}/health"

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            # Early exit: if the server process died, raise immediately
            # instead of waiting for the full health-check timeout.
            if self._sandbox is not None:
                out = self._adapter.exec(
                    self._sandbox,
                    "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null"
                    " && echo RUNNING || echo DEAD",
                )
                if "DEAD" in (out or ""):
                    raise RuntimeError(self._server_died_message())

            time.sleep(1.0)

        # The tunnel URL is a bearer capability, so it is deliberately omitted
        # from the timeout error.
        raise TimeoutError(f"Modal sandbox did not become ready within {timeout_s}s.")


__all__ = ["ModalProvider"]
