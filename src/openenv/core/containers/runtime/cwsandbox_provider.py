"""CoreWeave Sandbox provider for running OpenEnv environments."""

from __future__ import annotations

import logging
import os
import shlex
import time
from typing import Any
from urllib.parse import urlsplit

from .providers import ContainerProvider

logger = logging.getLogger(__name__)


class CWSandboxProvider(ContainerProvider):
    """Container provider that runs OpenEnv servers in CoreWeave Sandbox.

    The cwsandbox SDK uses a keepalive command by default, so this provider starts
    the OpenEnv server explicitly after the sandbox reaches RUNNING.

    Supply ``image`` here for provider-owned client startup, or to
    ``start_container`` to override it. Ingress must be explicitly configured;
    ``ingress_mode="public"`` opts into public access. No private mode is assumed.
    Startup requires an explicit HTTPS service address from the backend, before
    any server command runs. The SDK v0.23.3 documented bare-address path is not
    supported: neither ``url_scheme`` nor an exposed-port name configures TLS.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        base_url: str | None = None,
        ingress_mode: str | None = None,
        egress_mode: str | None = "internet",
        resources: Any | None = None,
        request_timeout_seconds: float = 300.0,
        max_lifetime_seconds: float | None = None,
        max_timeout_seconds: int | None = None,
        tags: list[str] | None = None,
        profile_ids: list[str] | None = None,
        profile_names: list[str] | None = None,
        runner_ids: list[str] | None = None,
        cmd: str | None = None,
        url_scheme: str = "https",
        delete_on_stop: bool = True,
        ready_settle_seconds: float = 2.0,
        surface_server_logs: bool = False,
        sdk: Any | None = None,
    ) -> None:
        if url_scheme.rstrip(":/") != "https":
            raise ValueError("CWSandboxProvider requires HTTPS transport.")
        self._image = image
        self._sdk = sdk if sdk is not None else _import_cwsandbox()
        self._base_url = base_url
        self._ingress_mode = ingress_mode
        self._egress_mode = egress_mode
        self._resources = resources
        self._request_timeout_seconds = request_timeout_seconds
        self._max_lifetime_seconds = max_lifetime_seconds
        self._max_timeout_seconds = max_timeout_seconds
        self._tags = tags
        self._profile_ids = profile_ids
        self._profile_names = profile_names
        self._runner_ids = runner_ids
        self._cmd = cmd
        self._delete_on_stop = delete_on_stop
        self._ready_settle_seconds = ready_settle_seconds
        self._surface_server_logs = surface_server_logs

        self._sandbox: Any = None
        self._base_env_url: str | None = None
        self._redact_values: set[str] = set()

    @classmethod
    def preflight(cls, *, sdk: Any | None = None) -> None:
        """Validate that CoreWeave Sandbox auth works before launch."""
        resolved_sdk = sdk if sdk is not None else _import_cwsandbox()
        if sdk is None and not os.environ.get("CWSANDBOX_API_KEY"):
            raise SystemExit(
                "CoreWeave Sandbox requires CWSANDBOX_API_KEY to be set. "
                "Please set this environment variable and try again."
            )
        try:
            resolved_sdk.Sandbox.list().result()
        except Exception as exc:
            auth_error = getattr(resolved_sdk, "CWSandboxAuthenticationError", None)
            if isinstance(auth_error, type) and isinstance(exc, auth_error):
                raise SystemExit(
                    f"CoreWeave Sandbox auth check failed: {exc}. "
                    "Verify your CWSANDBOX_API_KEY and try again."
                ) from exc
            raise

    @property
    def base_url(self) -> str:
        """Base URL returned to OpenEnv clients after startup."""
        if self._base_env_url is None:
            raise RuntimeError(
                "CWSandboxProvider has no active base_url. Start the provider "
                "before reading base_url."
            )
        return self._base_env_url

    @property
    def sandbox_id(self) -> str | None:
        """Sandbox ID for debugging after startup."""
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "sandbox_id", None)

    @property
    def service_address(self) -> str | None:
        """External service address reported by cwsandbox after startup."""
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "service_address", None)

    def start_container(
        self,
        image: str | None = None,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Start an OpenEnv image in a sandbox and return its base URL."""
        if self._sandbox is not None:
            raise RuntimeError(
                "CWSandboxProvider already has an active sandbox. Call "
                "stop_container() (or close()) before starting another; a "
                "second start would orphan the running sandbox."
            )

        image = image if image is not None else self._image
        if not image:
            raise ValueError(
                "CWSandboxProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        if not self._ingress_mode:
            raise ValueError(
                "CWSandboxProvider requires an explicit ingress_mode. Available "
                "modes depend on the runner; ingress_mode='public' opts into "
                "public access. No private ingress mode is assumed."
            )

        port = port or 8000
        if port != 8000:
            raise ValueError(
                f"CWSandboxProvider only supports port 8000 (got {port}). "
                "OpenEnv images and generated server commands listen on port 8000."
            )

        cmd = kwargs.pop("cmd", None) or self._cmd
        unknown = set(kwargs)
        if unknown:
            raise ValueError(f"Unsupported kwargs for CWSandboxProvider: {unknown}")

        network = self._sdk.NetworkOptions(
            ingress_mode=self._ingress_mode,
            exposed_ports=(port,),
            egress_mode=self._egress_mode,
        )

        run_kwargs: dict[str, Any] = {
            "container_image": image,
            "network": network,
            "request_timeout_seconds": self._request_timeout_seconds,
        }
        optional_kwargs = {
            "base_url": self._base_url,
            "environment_variables": env_vars,
            "resources": self._resources,
            "max_lifetime_seconds": self._max_lifetime_seconds,
            "max_timeout_seconds": self._max_timeout_seconds,
            "tags": self._tags,
            "profile_ids": self._profile_ids,
            "profile_names": self._profile_names,
            "runner_ids": self._runner_ids,
        }
        run_kwargs.update(
            {key: value for key, value in optional_kwargs.items() if value is not None}
        )

        self._redact_values = {value for value in (env_vars or {}).values() if value}
        try:
            sandbox = self._sdk.Sandbox.run(**run_kwargs)
        except Exception:
            self._redact_values = set()
            raise

        self._sandbox = sandbox
        try:
            sandbox.wait(timeout=self._request_timeout_seconds)
            self._base_env_url = self._base_url_from_sandbox(sandbox)
            server_cmd = cmd or self._discover_server_cmd(sandbox, port=port)
            self._start_server(sandbox, server_cmd)
            return self._base_env_url
        except Exception:
            try:
                self.stop_container()
            except Exception as cleanup_exc:
                logger.warning(
                    "Failed to clean up CoreWeave sandbox after startup failure",
                    exc_info=cleanup_exc,
                )
            raise

    def stop_container(self) -> None:
        """Stop and optionally delete the sandbox."""
        sandbox = self._sandbox
        if sandbox is None:
            self._redact_values = set()
            return

        sandbox_id = getattr(sandbox, "sandbox_id", None)
        stop_error: BaseException | None = None
        deleted = False
        try:
            self._result_with_retries(lambda: sandbox.stop(missing_ok=True))
        except Exception as exc:
            if not self._is_sandbox_not_running_error(exc):
                stop_error = exc
        finally:
            if self._delete_on_stop and sandbox_id:
                self._result_with_retries(
                    lambda: self._sdk.Sandbox.delete(
                        sandbox_id,
                        base_url=self._base_url,
                        timeout_seconds=self._request_timeout_seconds,
                        missing_ok=True,
                    )
                )
                deleted = True

        # A completed delete releases ownership even if stopping failed.
        if stop_error is None or deleted:
            self._sandbox = None
            self._base_env_url = None
            self._redact_values = set()

        if stop_error is not None:
            raise stop_error

    def close(self) -> None:
        """Stop the active sandbox, if any."""
        self.stop_container()

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """Wait for the OpenEnv server to respond to /health and /ws."""
        import requests

        deadline = time.time() + timeout_s
        health_url = f"{base_url.rstrip('/')}/health"
        last_error: BaseException | None = None

        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    if self._ready_settle_seconds:
                        time.sleep(self._ready_settle_seconds)
                    return
            except requests.RequestException as exc:
                last_error = exc

            if self._server_process_dead():
                raise RuntimeError(self._server_died_message())

            time.sleep(1.0)

        detail = f" Last error: {last_error}" if last_error else ""
        raise TimeoutError(
            f"OpenEnv server at {base_url} did not become ready within {timeout_s}s."
            f"{detail}"
        )

    def _discover_server_cmd(self, sandbox: Any, port: int) -> str:
        yaml_path = self._find_openenv_yaml(sandbox)
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml in the sandbox. Pass cmd= to "
                "CWSandboxProvider or start_container()."
            )

        content = self._exec_stdout(
            sandbox,
            f"cat {shlex.quote(yaml_path)}",
            timeout_seconds=10,
        )
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass cmd= to CWSandboxProvider or start_container()."
            )

        env_root = yaml_path.rsplit("/", 1)[0]
        return (
            f"cd {shlex.quote(env_root)} && "
            f"python -m uvicorn {shlex.quote(app)} --host 0.0.0.0 --port {port}"
        )

    def _find_openenv_yaml(self, sandbox: Any) -> str | None:
        candidate = "/app/env/openenv.yaml"
        out = self._exec_stdout(
            sandbox,
            f"test -f {shlex.quote(candidate)} && echo found",
            timeout_seconds=10,
        )
        if "found" in out:
            return candidate

        path = self._exec_stdout(
            sandbox,
            "find /app -maxdepth 5 -name openenv.yaml -print -quit 2>/dev/null",
            timeout_seconds=10,
        ).strip()
        if path.startswith("/"):
            return path
        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> str | None:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "CWSandboxProvider command discovery requires PyYAML. "
                "Install openenv with its normal dependencies."
            ) from exc

        try:
            data = yaml.safe_load(yaml_content) or {}
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        value = data.get("app")
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None

    def _start_server(self, sandbox: Any, cmd: str) -> None:
        escaped = shlex.quote(cmd)
        self._exec_stdout(
            sandbox,
            f"nohup bash -c {escaped} > /tmp/openenv-server.log 2>&1 & "
            "echo $! > /tmp/openenv-server.pid",
            timeout_seconds=10,
        )

    def _base_url_from_sandbox(self, sandbox: Any) -> str:
        address = getattr(sandbox, "service_address", None)
        if not address:
            raise RuntimeError(
                "Sandbox did not return service_address; cannot construct OpenEnv URL."
            )
        parsed = urlsplit(address)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise RuntimeError(
                "Sandbox must return an explicit HTTPS service_address. The "
                "SDK's bare addresses and exposed-port names do not establish "
                "TLS support; url_scheme cannot enable TLS. Configure a runner "
                "with verified HTTPS ingress before using this provider."
            )
        return address.rstrip("/")

    def _server_process_dead(self) -> bool:
        sandbox = self._sandbox
        if sandbox is None:
            return False
        out = self._exec_stdout(
            sandbox,
            "test -f /tmp/openenv-server.pid && "
            "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null "
            "&& echo RUNNING || echo DEAD",
            timeout_seconds=10,
            suppress_errors=True,
        )
        return "DEAD" in out

    def _server_log(self) -> str:
        sandbox = self._sandbox
        if sandbox is None:
            return ""
        return self._exec_stdout(
            sandbox,
            "cat /tmp/openenv-server.log 2>/dev/null",
            timeout_seconds=10,
            suppress_errors=True,
        )

    def _redact(self, text: str, *, max_chars: int = 2000) -> str:
        """Scrub injected env-var values before surfacing sandbox output."""
        redacted = text or ""
        for value in self._redact_values:
            redacted = redacted.replace(value, "***")
        if len(redacted) > max_chars:
            redacted = "...(truncated)...\n" + redacted[-max_chars:]
        return redacted

    def _server_died_message(self) -> str:
        """Build the startup-crash error, secure by default.

        The sandbox can run untrusted environment code, so raw server output is
        withheld unless explicitly requested. When surfaced, injected env-var
        values are redacted on a best-effort basis.
        """
        base = (
            "CoreWeave sandbox server process died during startup. Server output "
            "is not surfaced to avoid leaking secrets injected into the sandbox; "
            "retrieve /tmp/openenv-server.log from the sandbox out of band, or "
            "construct the provider with surface_server_logs=True to include a "
            "redacted excerpt."
        )
        if not self._surface_server_logs:
            return base

        log = self._redact(self._server_log())
        return (
            "CoreWeave sandbox server process died during startup. The excerpt "
            "below is the sandbox server output with injected secret values "
            "redacted (best-effort); it may still contain secrets the workload "
            f"printed by other means.\nLog (redacted):\n{log}"
        )

    def _is_sandbox_not_running_error(self, exc: BaseException) -> bool:
        sandbox_not_running = getattr(self._sdk, "SandboxNotRunningError", None)
        if isinstance(sandbox_not_running, type) and isinstance(
            exc, sandbox_not_running
        ):
            return True
        exc_type = type(exc)
        return (
            exc_type.__name__ == "SandboxNotRunningError"
            and exc_type.__module__.split(".", maxsplit=1)[0] in {"cwsandbox", "wandb"}
        )

    @staticmethod
    def _exec_stdout(
        sandbox: Any,
        script: str,
        *,
        timeout_seconds: int,
        suppress_errors: bool = False,
    ) -> str:
        try:
            process = sandbox.exec(
                ["bash", "-lc", script],
                timeout_seconds=timeout_seconds,
            )
            result = process.result()
            return getattr(result, "stdout", "") or ""
        except Exception:
            if suppress_errors:
                return ""
            raise

    @staticmethod
    def _result_with_retries(operation: Any, attempts: int = 3) -> Any:
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                return operation().result()
            except Exception as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(1.0)
        assert last_error is not None
        raise last_error


def _import_cwsandbox() -> Any:
    try:
        return __import__("cwsandbox")
    except ImportError as exc:
        raise RuntimeError(
            "CWSandboxProvider requires cwsandbox. Install it with "
            "`pip install cwsandbox`."
        ) from exc
