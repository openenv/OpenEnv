# SPDX-License-Identifier: BSD-3-Clause

"""
Daytona container provider for running OpenEnv environments in Daytona cloud sandboxes.

Requires the ``daytona`` SDK: ``pip install daytona>=0.10``
"""

from __future__ import annotations

import os
import shlex
import time
from typing import Any, Callable, Dict, Optional

from ._server_config import parse_dockerfile_cmd, parse_openenv_app_field
from .providers import ContainerProvider


class DaytonaProvider(ContainerProvider):
    """
    Container provider that runs environments in Daytona cloud sandboxes.

    Example:
        >>> provider = DaytonaProvider(api_key="your-key")
        >>> image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
        >>> base_url = provider.start_container(image)
        >>> provider.wait_for_ready(base_url)
        >>> provider.stop_container()
    """

    _dockerfile_registry: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        *,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        public: bool = False,
        resources: Optional[Any] = None,
        auto_stop_interval: int = 15,
        target: Optional[str] = None,
        on_snapshot_create_logs: Optional[Callable[[str], None]] = None,
        cmd: Optional[str] = None,
        create_timeout: float = 300,
    ):
        """
        Args:
            image: Docker image, snapshot reference, or ``"dockerfile:<path>"``
                source to use when ``start_container()`` is called without an
                image.
            env_vars: Environment variables to use when ``start_container()`` is
                called without explicit ``env_vars``.
            api_key: Daytona API key. Falls back to ``DAYTONA_API_KEY`` env var.
            public: If True, the sandbox preview is publicly accessible.
            resources: Optional ``daytona.Resources`` instance for CPU/memory.
            auto_stop_interval: Minutes of inactivity before auto-stop (0 disables).
            target: Daytona target region (e.g. "us").
            on_snapshot_create_logs: Callback for snapshot build log lines.
            cmd: Shell command to start the server inside the sandbox.
            create_timeout: Seconds to wait for sandbox creation (default 300).
                Heavy images (e.g. with Playwright/Chromium) may need more.
        """
        from daytona import Daytona, DaytonaConfig

        config_kwargs: Dict[str, Any] = {}
        resolved_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if resolved_key:
            config_kwargs["api_key"] = resolved_key
        if target:
            config_kwargs["target"] = target

        self._daytona = Daytona(DaytonaConfig(**config_kwargs))
        self._image = image
        self._env_vars = env_vars
        self._public = public
        self._resources = resources
        self._auto_stop_interval = auto_stop_interval
        self._on_snapshot_create_logs = on_snapshot_create_logs
        self._cmd = cmd
        self._create_timeout = create_timeout
        self._sandbox: Any = None
        self._preview_url: Optional[str] = None

    def _discover_server_cmd(self, sandbox: Any, port: int = 8000) -> str:
        """Discover the server command from ``openenv.yaml`` inside *sandbox*.

        Finds the file, reads the ``app`` field, and constructs a command
        of the form ``cd <env_root> && python -m uvicorn <app> --host 0.0.0.0 --port <port>``.

        Raises:
            ValueError: If ``openenv.yaml`` is not found or lacks an ``app`` field.
        """
        yaml_path = self._find_openenv_yaml(sandbox)
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml inside the sandbox. "
                "Pass an explicit cmd= to DaytonaProvider or start_container()."
            )

        cat_resp = sandbox.process.exec(f"cat {shlex.quote(yaml_path)}", timeout=10)
        content = cat_resp.result if hasattr(cat_resp, "result") else str(cat_resp)
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass an explicit cmd= to DaytonaProvider or start_container()."
            )

        # The directory containing openenv.yaml is the env root
        env_root = yaml_path.rsplit("/", 1)[0]
        return (
            f"cd {shlex.quote(env_root)} && "
            f"python -m uvicorn {shlex.quote(app)} --host 0.0.0.0 --port {port}"
        )

    def _find_openenv_yaml(self, sandbox: Any) -> Optional[str]:
        """Locate ``openenv.yaml`` inside the sandbox.

        Tries the modern layout path ``/app/env/openenv.yaml`` first,
        then falls back to a ``find`` command for the old layout.
        """
        # Fast path: modern Dockerfile layout
        resp = sandbox.process.exec(
            "test -f /app/env/openenv.yaml && echo found", timeout=10
        )
        out = resp.result if hasattr(resp, "result") else str(resp)
        if "found" in (out or ""):
            return "/app/env/openenv.yaml"

        # Fallback: search for it (redirect stderr so error messages
        # like "No such file or directory" don't get mistaken for paths).
        resp = sandbox.process.exec(
            "find /app -maxdepth 4 -name openenv.yaml -print -quit 2>/dev/null",
            timeout=10,
        )
        path = (resp.result if hasattr(resp, "result") else str(resp) or "").strip()
        if path and path.startswith("/"):
            return path

        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> Optional[str]:
        """Extract the ``app`` value from raw openenv.yaml content.

        Uses PyYAML to handle comments, quotes, and nested keys correctly.
        """
        return parse_openenv_app_field(yaml_content)

    @staticmethod
    def _parse_dockerfile_cmd(dockerfile_content: str) -> Optional[str]:
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

    @staticmethod
    def strip_buildkit_syntax(dockerfile_content: str) -> str:
        """Remove BuildKit ``--mount=...`` flags from ``RUN`` instructions.

        Handles single-line flags, multi-line continuations, and multiple
        ``--mount`` flags spread across continuation lines. Only leading
        ``--mount`` flags are removed (before the actual command starts).

        Daytona's ``Image.from_dockerfile`` does not support BuildKit
        ``--mount`` syntax.  This helper strips the flags so that standard
        Dockerfiles (like the ones generated by ``openenv build``) can
        be used directly.
        """
        import re

        def strip_leading_mounts(text: str) -> str:
            remaining = text
            while True:
                match = re.match(r"\s*--mount=\S+\s*", remaining)
                if not match:
                    return remaining
                remaining = remaining[match.end() :]

        lines = dockerfile_content.split("\n")
        result: list[str] = []
        in_run = False
        in_mount_prefix = False

        for line in lines:
            line_out = line
            run_start = False
            if re.match(r"\s*RUN(\s+|$)", line, flags=re.IGNORECASE):
                in_run = True
                in_mount_prefix = True
                run_start = True

            if in_run and in_mount_prefix:
                original_ends_with_slash = line_out.rstrip().endswith("\\")
                if run_start:
                    match = re.match(r"(\s*RUN\s+)(.*)$", line_out, flags=re.IGNORECASE)
                    if match:
                        run_prefix, remainder = match.group(1), match.group(2)
                    else:
                        run_prefix, remainder = line_out, ""
                    new_remainder = strip_leading_mounts(remainder)
                    line_out = run_prefix + new_remainder
                    content_for_check = new_remainder
                else:
                    new_remainder = strip_leading_mounts(line_out)
                    line_out = new_remainder
                    content_for_check = new_remainder

                if original_ends_with_slash and not line_out.rstrip().endswith("\\"):
                    line_out = line_out.rstrip() + " \\"

                if content_for_check.strip() not in ("", "\\"):
                    in_mount_prefix = False

            if in_run and not line_out.rstrip().endswith("\\"):
                in_run = False
                in_mount_prefix = False

            result.append(line_out)

        return "\n".join(result)

    @classmethod
    def image_from_dockerfile(
        cls,
        dockerfile_path: str,
        context_dir: str | None = None,
    ) -> str:
        """Validate a Dockerfile and return a ``dockerfile:`` URI for
        :meth:`start_container`.

        Eagerly validates the Dockerfile (existence, COPY sources,
        BuildKit stripping) and stores the processed content in an
        internal registry.  The actual ``daytona.Image`` is created
        later inside ``start_container``.

        Args:
            dockerfile_path: Path to the Dockerfile on disk.
            context_dir: Build context directory.  Defaults to the
                Dockerfile's grandparent directory, matching the
                ``openenv init`` convention where Dockerfiles live in
                ``<env>/server/Dockerfile`` and the build context is
                ``<env>/``.  Pass explicitly for non-standard layouts
                (e.g. ``context_dir="."`` for repo-root contexts).

        Returns:
            A ``"dockerfile:<abs_path>"`` string to pass to
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
        stripped = cls.strip_buildkit_syntax(content)

        # Validate that COPY sources exist under the context directory.
        # This catches mismatches early (e.g. a Dockerfile expecting repo
        # root as context when we defaulted to the env directory).
        for line in stripped.splitlines():
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

        # Parse CMD from the original Dockerfile so start_container can
        # use it as a fallback when openenv.yaml is unavailable.
        parsed_cmd = cls._parse_dockerfile_cmd(content)

        cls._dockerfile_registry[str(src)] = {
            "stripped_content": stripped,
            "context_dir": str(ctx),
            "server_cmd": parsed_cmd,
        }

        return f"dockerfile:{src}"

    def start_container(
        self,
        image: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Daytona sandbox from a Docker image or snapshot.

        Daytona does not execute the image's CMD (known bug — ENTRYPOINT
        runs, CMD does not).  The server command is resolved in order:

        1. Explicit ``cmd`` passed to the constructor.
        2. ``cmd`` key in ``**kwargs`` (popped before forwarding).
        3. Auto-discovered from ``openenv.yaml`` inside the sandbox.
        4. ``CMD`` parsed from the Dockerfile (when *image* came from
           ``image_from_dockerfile``).

        Args:
            image: Docker image name (e.g. ``"echo-env:latest"``),
                   ``"snapshot:<name>"`` to create from a pre-built snapshot,
                   or ``"dockerfile:<path>"`` returned by
                   :meth:`image_from_dockerfile`. May be omitted when supplied
                   to the constructor.
            port: Must be ``None`` or ``8000``. Daytona exposes port 8000
                  via its preview proxy; other ports raise ``ValueError``.
            env_vars: Environment variables forwarded to the sandbox.
            **kwargs: ``cmd`` (str) to override the server command;
                remaining kwargs passed through to ``Daytona.create()``.

        Returns:
            HTTPS preview URL for the sandbox (base_url).
        """
        if port is not None and port != 8000:
            raise ValueError(
                f"DaytonaProvider only supports port 8000 (got {port}). "
                "The Daytona preview proxy routes to port 8000 inside the sandbox."
            )

        effective_image = image if image is not None else self._image
        if effective_image is None:
            raise ValueError(
                "DaytonaProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        effective_env_vars = self._env_vars if env_vars is None else env_vars

        # Resolve the server command (may be None; discovery happens after
        # sandbox creation when we can inspect the filesystem).
        cmd = kwargs.pop("cmd", None) or self._cmd

        # CMD parsed from Dockerfile (populated for "dockerfile:" images).
        parsed_cmd: Optional[str] = None

        # Build creation params
        create_kwargs: Dict[str, Any] = {}
        if effective_env_vars:
            create_kwargs["env_vars"] = effective_env_vars
        if self._public:
            create_kwargs["public"] = True
        if self._auto_stop_interval != 15:
            create_kwargs["auto_stop_interval"] = self._auto_stop_interval

        if effective_image.startswith("snapshot:"):
            from daytona import CreateSandboxFromSnapshotParams

            snapshot_name = effective_image[len("snapshot:") :]
            params = CreateSandboxFromSnapshotParams(
                snapshot=snapshot_name, **create_kwargs
            )
        elif effective_image.startswith("dockerfile:"):
            from daytona import CreateSandboxFromImageParams, Image

            dockerfile_path = effective_image[len("dockerfile:") :]
            meta = self._dockerfile_registry.get(dockerfile_path)
            if meta is None:
                raise ValueError(
                    f"No registered Dockerfile metadata for {dockerfile_path}. "
                    "Call DaytonaProvider.image_from_dockerfile() first."
                )

            parsed_cmd = meta.get("server_cmd")

            # Build the daytona Image from the pre-stripped content.
            import pathlib
            import uuid

            ctx = pathlib.Path(meta["context_dir"])
            tmp_name = f".daytona-{uuid.uuid4().hex[:8]}.dockerfile"
            tmp_path = ctx / tmp_name
            try:
                tmp_path.write_text(meta["stripped_content"])
                daytona_image = Image.from_dockerfile(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)

            img_kwargs: Dict[str, Any] = {
                "image": daytona_image,
                **create_kwargs,
            }
            if self._resources is not None:
                img_kwargs["resources"] = self._resources
            params = CreateSandboxFromImageParams(**img_kwargs)
        else:
            from daytona import CreateSandboxFromImageParams

            img_kwargs = {"image": effective_image, **create_kwargs}
            if self._resources is not None:
                img_kwargs["resources"] = self._resources
            params = CreateSandboxFromImageParams(**img_kwargs)

        # Create sandbox
        extra: Dict[str, Any] = dict(kwargs)
        if self._on_snapshot_create_logs is not None:
            extra["on_snapshot_create_logs"] = self._on_snapshot_create_logs

        self._sandbox = self._daytona.create(
            params, timeout=self._create_timeout, **extra
        )

        try:
            # Discover server command from openenv.yaml if not explicitly set.
            if cmd is None:
                try:
                    cmd = self._discover_server_cmd(self._sandbox)
                except ValueError:
                    # Fall back to CMD parsed from Dockerfile (if available).
                    if parsed_cmd:
                        cmd = parsed_cmd
                    else:
                        raise

            # Wrap in bash -c so compound commands (cd ... && uvicorn ...)
            # are handled correctly by nohup.  Write PID so we can check
            # if the process crashed later in wait_for_ready().
            escaped_cmd = shlex.quote(cmd)
            self._sandbox.process.exec(
                f"nohup bash -c {escaped_cmd} > /tmp/openenv-server.log 2>&1 &"
                " echo $! > /tmp/openenv-server.pid",
                timeout=10,
            )

            # Get a signed preview URL for port 8000.  The token is
            # embedded in the URL itself so no extra headers are needed.
            signed = self._sandbox.create_signed_preview_url(
                8000, expires_in_seconds=86400
            )
            self._preview_url = signed.url
        except Exception:
            self.stop_container()
            raise

        return self._preview_url

    def refresh_preview_url(self) -> str:
        """Get a fresh signed preview URL (valid for 24h).

        Daytona signed URLs expire after at most 24 hours.  Call this to
        get a new one for long-running sessions.  The returned URL points
        to the same sandbox — clients will need to reconnect using it.
        """
        if self._sandbox is None:
            raise RuntimeError("No active sandbox to refresh URL for.")
        signed = self._sandbox.create_signed_preview_url(8000, expires_in_seconds=86400)
        self._preview_url = signed.url
        return self._preview_url

    def stop_container(self) -> None:
        """Delete the Daytona sandbox."""
        if self._sandbox is None:
            return

        try:
            self._daytona.delete(self._sandbox)
        finally:
            self._sandbox = None
            self._preview_url = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the /health endpoint until the sandbox is ready.

        Uses a longer default timeout (120s) than Docker providers because
        Daytona sandboxes may have cold-start latency.

        Args:
            base_url: Preview URL returned by ``start_container()``.
            timeout_s: Maximum seconds to wait.

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
                resp = self._sandbox.process.exec(
                    "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null"
                    " && echo RUNNING || echo DEAD",
                    timeout=10,
                )
                out = resp.result if hasattr(resp, "result") else str(resp)
                if "DEAD" in (out or ""):
                    log_resp = self._sandbox.process.exec(
                        "cat /tmp/openenv-server.log 2>/dev/null", timeout=10
                    )
                    log = (
                        log_resp.result
                        if hasattr(log_resp, "result")
                        else str(log_resp)
                    )
                    raise RuntimeError(f"Server process died.\nLog:\n{log}")

            time.sleep(1.0)

        raise TimeoutError(
            f"Daytona sandbox at {base_url} did not become ready within {timeout_s}s"
        )
