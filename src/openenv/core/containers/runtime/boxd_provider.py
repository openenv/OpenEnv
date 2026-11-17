# SPDX-License-Identifier: BSD-3-Clause

"""
boxd container provider for running OpenEnv environments in boxd machines.

Requires the ``boxd`` SDK: ``pip install boxd>=0.3.1``

The provider creates a boxd machine (a microVM with Docker installed), runs the
environment image in it with ``docker run``, and returns the machine's
``https://`` URL that ``EnvClient`` connects to over ``wss://``.
"""

from __future__ import annotations

import io
import logging
import pathlib
import re
import shlex
import tarfile
import time
import uuid
from typing import Any, Dict

from .providers import ContainerProvider

logger = logging.getLogger(__name__)

_PORT = 8000
_CONTAINER = "openenv-server"
_BUILD_DIR = "/home/boxd/openenv-build"
_BUILT_IMAGE = "openenv-env:latest"
_ENV_FILE = "/home/boxd/openenv-server.env"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SKIP_IN_CONTEXT = {".git", ".venv", "__pycache__"}
# wait_for_ready skips the container state check with less time than this left.
_MIN_PROBE_S = 1.0
# Time limit for reading the container log after the container exited.
_LOGS_TIMEOUT_S = 10.0


def _require_secure_url(url: Any) -> str:
    """Refuse a base URL that is not https, so ``EnvClient`` never falls back
    to plaintext ``ws://``. The URL itself stays out of the error."""
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise RuntimeError(
            "The boxd machine returned a non-HTTPS URL. OpenEnv requires an "
            "https/wss base_url, so the provider refuses to connect."
        )
    return url


class BoxdProvider(ContainerProvider):
    """
    Container provider that runs environments in [boxd](https://boxd.sh)
    machines.

    Each ``start_container`` creates one isolated machine, runs the environment
    image in it with Docker, and returns the machine's ``https://`` URL.
    ``image`` is either a registry reference (``"ghcr.io/org/echo-env:latest"``)
    or a ``"dockerfile:<path>"`` reference returned by
    [`~openenv.core.containers.runtime.boxd_provider.BoxdProvider.image_from_dockerfile`],
    which is built inside the machine.

    The URL is public: anyone who has it can drive the environment, with no
    login in front of it. The provider therefore requires ``public=True``.

    The environment runs untrusted code, so the machine is always created
    isolated: it joins no private network and holds no boxd credentials.
    Outbound internet is open unless ``egress_allow`` restricts it.
    Server output is kept out of raised errors unless
    ``surface_server_logs=True``.

    Only one machine is active per provider: calling ``start_container`` again
    before ``stop_container()``/``close()`` raises ``RuntimeError``.
    ``stop_container()`` deletes the machine.

    Idle suspend and hibernation are turned off, so a quiet session is not
    dropped. As a backstop, boxd deletes the machine ``delete_after`` seconds
    after it was created (one hour by default), even if ``stop_container()``
    never runs, for example when the process is killed.

    Examples:

    ```python
    with BoxdProvider(public=True, region="fra") as provider:
        base_url = provider.start_container("ghcr.io/huggingface/openenv-echo-env:latest")
        provider.wait_for_ready(base_url)
    # machine deleted on exit
    ```
    """

    _dockerfile_registry: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        *,
        public: bool = False,
        image: str | None = None,
        env_vars: dict[str, str] | None = None,
        api_key: str | None = None,
        org: str | None = None,
        region: str | None = None,
        vcpu: int | None = None,
        memory: str | None = None,
        name_prefix: str = "openenv",
        delete_after: int | None = 3600,
        egress_allow: list[str] | None = None,
        start_timeout: float = 900.0,
        surface_server_logs: bool = False,
        _client: Any = None,
    ):
        """
        Args:
            public (`bool`, *optional*, defaults to `False`):
                Must be `True`. Confirms that the machine's URL is public:
                anyone who has it can drive the environment while the machine
                runs.
            image (`str`, *optional*):
                Registry image or ``"dockerfile:<path>"`` source to use when
                ``start_container()`` is called without an image.
            env_vars (`dict`, *optional*):
                Environment variables to use when ``start_container()`` is
                called without explicit ``env_vars``.
            api_key (`str`, *optional*):
                boxd API key. Falls back to the ``BOXD_API_KEY`` env var.
            org (`str`, *optional*):
                boxd organization to create the machine in. Defaults to the
                key's own organization.
            region (`str`, *optional*):
                boxd region, ``"fra"`` or ``"nyc"``. Falls back to
                ``BOXD_REGION``, then to the account default.
            vcpu (`int`, *optional*):
                vCPUs for the machine (``1``, ``2``, ``4`` or ``8``). When
                `None`, the organization's default size applies.
            memory (`str`, *optional*):
                Memory for the machine, matching ``vcpu`` (``"4G"``, ``"8G"``,
                ``"16G"`` or ``"32G"``).
            name_prefix (`str`, *optional*, defaults to `"openenv"`):
                Prefix for the machine name. A random suffix is appended.
            delete_after (`int` or `None`, *optional*, defaults to `3600`):
                Seconds after creation when boxd deletes the machine, whatever
                it is doing. Set it above the longest session you run. `None`
                turns the timer off, so the machine lives until
                ``stop_container()``.
            egress_allow (`list[str]`, *optional*):
                Hostnames (``"api.example.com"``, ``"*.example.com"``) and
                public IPv4 addresses or CIDRs the machine may reach. Applied
                after the image is pulled or built and before the container
                starts, so the environment never runs with open egress.
                ``["none"]`` blocks all outbound traffic. When `None`,
                outbound internet is open.
            start_timeout (`float`, *optional*, defaults to `900.0`):
                Seconds allowed for ``docker build`` (for a Dockerfile image)
                or ``docker pull`` (for a registry image).
            surface_server_logs (`bool`, *optional*, defaults to `False`):
                When `True`, a failed ``docker build`` includes the tail of
                its output, and startup-crash errors include the tail of the
                container's output with injected env-var values and the
                machine's hostname redacted.
        """
        if not public:
            raise ValueError(
                "BoxdProvider requires public=True: the machine's URL is "
                "reachable from the internet with no login in front of it."
            )
        if delete_after is not None and (
            isinstance(delete_after, bool)
            or not isinstance(delete_after, int)
            or delete_after <= 0
        ):
            raise ValueError(
                f"delete_after must be a positive number of seconds or None "
                f"(got {delete_after!r})."
            )
        if _client is None:
            from boxd import Boxd

            _client = Boxd(api_key=api_key, region=region)
        self._client = _client
        self._image = image
        self._env_vars = env_vars
        self._org = org
        self._vcpu = vcpu
        self._memory = memory
        self._name_prefix = name_prefix
        self._delete_after = delete_after
        self._egress_allow = egress_allow
        self._start_timeout = start_timeout
        self._surface_server_logs = surface_server_logs
        self._machine_id: str | None = None
        self._base_url: str | None = None
        self._redact_values: set[str] = set()

    @classmethod
    def image_from_dockerfile(
        cls,
        dockerfile_path: str,
        context_dir: str | None = None,
    ) -> str:
        """Validate a Dockerfile and return a ``dockerfile:`` URI for
        [`~openenv.core.containers.runtime.boxd_provider.BoxdProvider.start_container`].

        The build context is uploaded to the machine and built there with
        ``docker build`` when the provider starts.

        Args:
            dockerfile_path (`str`):
                Path to the Dockerfile on disk.
            context_dir (`str`, *optional*):
                Build context directory. Defaults to the Dockerfile's
                grandparent directory, matching the ``openenv init`` layout
                where Dockerfiles live in ``<env>/server/Dockerfile`` and the
                build context is ``<env>/``.

        Returns:
            `str`: A ``"dockerfile:<abs_path>"`` string to pass to
            ``start_container``.

        Raises:
            FileNotFoundError: If *dockerfile_path* does not exist.
            ValueError: If *context_dir* is given but does not exist, or if
                COPY sources in the Dockerfile cannot be found under it.
        """
        src = pathlib.Path(dockerfile_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        if context_dir is not None:
            ctx = pathlib.Path(context_dir).resolve()
            if not ctx.is_dir():
                raise ValueError(f"context_dir does not exist: {context_dir}")
        else:
            ctx = src.parent.parent

        for line in src.read_text().splitlines():
            m = re.match(r"^\s*COPY\s+(?!--from=)(\S+)\s+", line, re.IGNORECASE)
            if not m or m.group(1).startswith("/"):
                continue
            copy_src = m.group(1)
            if not (ctx / copy_src).exists() and not any(ctx.glob(copy_src)):
                raise ValueError(
                    f"Dockerfile COPY source '{copy_src}' not found under "
                    f"context_dir '{ctx}'. Pass context_dir explicitly."
                )

        cls._dockerfile_registry[str(src)] = {
            "dockerfile_path": str(src),
            "context_dir": str(ctx),
        }
        return f"dockerfile:{src}"

    def _exec(self, command: str, *, timeout: float = 30.0) -> Any:
        return self._client.machines.exec(self._machine_id, command, timeout=timeout)

    def _build_image(self, meta: Dict[str, str]) -> str:
        """Upload the registered build context and build it in the machine."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(
                meta["context_dir"],
                arcname=".",
                filter=lambda info: (
                    None
                    if _SKIP_IN_CONTEXT & set(pathlib.PurePath(info.name).parts)
                    else info
                ),
            )
        files = self._client.machines.files
        files.upload(self._machine_id, f"{_BUILD_DIR}.tar.gz", buf.getvalue())
        files.upload(
            self._machine_id,
            f"{_BUILD_DIR}.Dockerfile",
            pathlib.Path(meta["dockerfile_path"]).read_bytes(),
        )

        result = self._exec(
            f"mkdir -p {_BUILD_DIR} && tar -xzf {_BUILD_DIR}.tar.gz -C {_BUILD_DIR}"
            f" && docker build --progress=plain -t {_BUILT_IMAGE}"
            f" -f {_BUILD_DIR}.Dockerfile {_BUILD_DIR}",
            timeout=self._start_timeout,
        )
        if result.exit_code != 0:
            message = (
                f"docker build failed in the boxd machine (exit {result.exit_code})."
            )
            if self._surface_server_logs:
                message += f"\n{(result.stdout + result.stderr)[-2000:]}"
            else:
                message += (
                    " Its output is not surfaced; construct the provider with "
                    "surface_server_logs=True to include it."
                )
            raise RuntimeError(message)
        return _BUILT_IMAGE

    def start_container(
        self,
        image: str | None = None,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a boxd machine and run the environment image in it.

        Args:
            image (`str`, *optional*):
                Registry image (e.g. ``"ghcr.io/org/echo-env:latest"``) or
                ``"dockerfile:<path>"`` returned by
                [`~openenv.core.containers.runtime.boxd_provider.BoxdProvider.image_from_dockerfile`].
                May be omitted when supplied to the constructor.
            port (`int`, *optional*):
                Must be `None` or `8000`. The provider pins the machine's
                HTTPS URL to port 8000; other ports raise ``ValueError``.
            env_vars (`dict`, *optional*):
                Environment variables passed to the container. Names must be
                shell identifiers and values must be single lines.
            **kwargs:
                Not supported; any keyword raises ``TypeError``.

        Returns:
            `str`: The machine's HTTPS URL (base_url).
        """
        if self._machine_id is not None:
            raise RuntimeError(
                "BoxdProvider already has an active machine. Call "
                "stop_container() (or close()) before starting another."
            )
        if port is not None and port != _PORT:
            raise ValueError(
                f"BoxdProvider only supports port {_PORT} (got {port}). The "
                f"machine's HTTPS URL is pinned to port {_PORT}."
            )
        if kwargs:
            raise TypeError(f"Unsupported start_container options: {sorted(kwargs)}")

        effective_image = image if image is not None else self._image
        if effective_image is None:
            raise ValueError(
                "BoxdProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        build_meta = None
        if effective_image.startswith("dockerfile:"):
            dockerfile_path = effective_image[len("dockerfile:") :]
            build_meta = self._dockerfile_registry.get(dockerfile_path)
            if build_meta is None:
                raise ValueError(
                    f"No registered Dockerfile metadata for {dockerfile_path}. "
                    "Call BoxdProvider.image_from_dockerfile() first."
                )
        effective_env_vars = dict(
            (self._env_vars if env_vars is None else env_vars) or {}
        )
        for name, value in effective_env_vars.items():
            if not _ENV_NAME.match(name) or "\n" in value:
                raise ValueError(
                    f"Invalid env var {name!r}: names must be shell identifiers "
                    "and values must be single lines."
                )
        self._redact_values = {v for v in effective_env_vars.values() if v}

        name = f"{self._name_prefix}-{uuid.uuid4().hex[:12]}"
        # Until create answers, the name is the only handle on the machine.
        self._machine_id = name
        try:
            # Idle suspend is off, so a quiet session is not dropped;
            # delete_after bounds the machine's lifetime instead.
            machine = self._client.machines.create(
                name,
                org=self._org,
                isolated=True,
                labels={"managed-by": "openenv"},
                vcpu=self._vcpu,
                memory=self._memory,
                auto_suspend_timeout=0,
                delete_after=self._delete_after,
            )
            self._machine_id = machine.id
            ready = self._client.machines.wait_until_ready(machine.id)
            url = _require_secure_url(ready.access.url)
            # The host is the secret part of the URL: keep it out of debug logs.
            self._redact_values.add(url.removeprefix("https://"))
            self._client.machines.set_auto_hibernate_timeout(machine.id, 0)
            # Pin the URL to the server's port, whatever else listens later.
            self._client.machines.proxies.set_port(machine.id, _PORT)
            # EnvClient is not a browser, so the browser check on the
            # machine's URL is turned off.
            self._client.machines.set_bot_protection(machine.id, mode="off")

            if build_meta is not None:
                effective_image = self._build_image(build_meta)
            else:
                pull = self._exec(
                    f"docker pull -q {shlex.quote(effective_image)}",
                    timeout=self._start_timeout,
                )
                if pull.exit_code != 0:
                    raise RuntimeError(
                        f"docker pull failed in the boxd machine (exit {pull.exit_code}):"
                        f"\n{pull.stderr[-2000:]}"
                    )
            if self._egress_allow is not None:
                self._client.machines.set_egress_allow(machine.id, self._egress_allow)

            # Env values go in a file read by `docker run --env-file`, so they
            # never appear on a command line. The file is removed after the run.
            env_flag = ""
            if effective_env_vars:
                self._client.machines.files.upload(
                    machine.id,
                    _ENV_FILE,
                    "".join(f"{k}={v}\n" for k, v in effective_env_vars.items()),
                )
                env_flag = f" --env-file {_ENV_FILE}"
            result = self._exec(
                f"docker run -d --name {_CONTAINER}{env_flag} -p {_PORT}:{_PORT}"
                f" {shlex.quote(effective_image)}; rc=$?; rm -f {_ENV_FILE}; exit $rc"
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"docker run failed in the boxd machine (exit {result.exit_code}):"
                    f"\n{result.stderr[-2000:]}"
                )
            self._base_url = url
        except BaseException:
            try:
                self.stop_container()
            except Exception:
                logger.warning(
                    "Could not delete the boxd machine after a failed start; "
                    "retry stop_container() or close()."
                )
            raise

        return self._base_url

    def stop_container(self) -> None:
        """Delete the boxd machine.

        The machine's handle is kept when the delete fails, so calling this
        again retries it. A machine that is already gone counts as deleted.
        """
        if self._machine_id is None:
            self._redact_values = set()
            return
        try:
            self._client.machines.delete(self._machine_id)
        except Exception as exc:
            from boxd.errors import NotFoundError

            if not isinstance(exc, NotFoundError):
                raise
        self._machine_id = None
        self._base_url = None
        self._redact_values = set()

    def close(self) -> None:
        """Delete the active machine and close the SDK client."""
        try:
            self.stop_container()
        finally:
            self._client.close()

    @property
    def machine_id(self) -> str:
        """Id of the boxd machine started by the last ``start_container``.

        After a failed start whose cleanup also failed, this is the handle
        ``stop_container()`` retries with (the machine's name when the create
        call itself never answered).
        """
        if self._machine_id is None:
            raise RuntimeError("BoxdProvider has no active machine.")
        return self._machine_id

    def _server_died_message(self) -> str:
        base = "The environment container in the boxd machine exited during startup."
        if not self._surface_server_logs:
            return (
                f"{base} Its output is not surfaced to avoid leaking injected "
                "secrets; construct the provider with surface_server_logs=True "
                "to include a redacted excerpt."
            )
        try:
            log = self._exec(
                f"docker logs --tail 100 {_CONTAINER} 2>&1", timeout=_LOGS_TIMEOUT_S
            ).stdout
        except Exception:
            return f"{base} Its log could not be read."
        for value in sorted(self._redact_values, key=len, reverse=True):
            log = log.replace(value, "***")
        return f"{base}\nLog (redacted):\n{log[-2000:]}"

    def _container_state(self, *, timeout: float) -> str | None:
        """State of the environment container, or ``None`` when the check
        timed out or the connection failed (the caller keeps polling)."""
        try:
            result = self._exec(
                f"docker inspect -f '{{{{.State.Status}}}}' {_CONTAINER}",
                timeout=timeout,
            )
        except Exception as exc:
            from boxd.errors import (
                AuthenticationError,
                BoxdError,
                NotFoundError,
                PermissionDeniedError,
            )

            definite = (AuthenticationError, NotFoundError, PermissionDeniedError)
            if isinstance(exc, BoxdError) and not isinstance(exc, definite):
                return None
            raise
        return result.stdout.strip()

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the /health endpoint until the environment server answers.

        Args:
            base_url (`str`):
                URL returned by ``start_container()``.
            timeout_s (`float`, *optional*, defaults to `120.0`):
                Maximum seconds to wait.

        Raises:
            TimeoutError: If the server doesn't become ready in time.
            RuntimeError: If the container exited.
        """
        import requests

        health_url = f"{base_url}/health"
        deadline = time.monotonic() + timeout_s
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                response = requests.get(health_url, timeout=min(5.0, remaining))
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            remaining = deadline - time.monotonic()
            if self._machine_id is not None and remaining >= _MIN_PROBE_S:
                state = self._container_state(timeout=min(30.0, remaining))
                if state in ("exited", "dead"):
                    raise RuntimeError(self._server_died_message())

            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

        raise TimeoutError(f"The boxd machine did not become ready within {timeout_s}s")


__all__ = ["BoxdProvider"]
