# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for NovitaSandboxProvider.

Provider-behavior tests inject a duck-typed fake adapter (``_adapter=``), so
they run without ``pip install novita-sandbox``. A separate test installs a
minimal fake ``novita_sandbox`` module to pin the real ``_DefaultNovitaAdapter``
SDK shape (method names and kwargs), so SDK churn shows up as a test failure.
"""

from __future__ import annotations

import shlex
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.novita_provider import (
    _DefaultNovitaAdapter,
    NovitaSandboxProvider,
)
from openenv.core.containers.runtime.providers import ContainerProvider


# ---------------------------------------------------------------------------
# Fake adapter (duck-typed to _DefaultNovitaAdapter)
# ---------------------------------------------------------------------------
_SANDBOX_HOST = "8000-sbx123.us-phx-1.sandbox.novita.ai"


class _FakeAdapter:
    def __init__(self):
        self.created: list[dict] = []
        self.exec_commands: list[str] = []
        self.exec_timeouts: list[float] = []
        self.exec_background: list[bool] = []
        self.exec_users: list[str | None] = []
        self.killed = 0
        self.host_value = _SANDBOX_HOST
        self.fail_create = False
        self.fail_kill = False
        self.fail_build = False
        self.dead_process = False
        self.pid_ready = True
        self.log = "server crashed"
        self.has_yaml = True
        self.template_builds: list[dict] = []
        self.template_id = "tpl-abc123"

    def create_sandbox(
        self,
        *,
        image,
        template,
        env_vars,
        timeout,
        metadata,
        secure,
    ):
        if self.fail_create:
            raise RuntimeError("create failed")
        sandbox = object()
        self.created.append(
            {
                "image": image,
                "template": template,
                "env_vars": env_vars,
                "timeout": timeout,
                "metadata": metadata,
                "secure": secure,
                "sandbox": sandbox,
            }
        )
        return sandbox

    def build_template(
        self,
        *,
        dockerfile_content,
        context_dir,
        name,
        cpu_count,
        memory_mb,
        on_build_logs,
    ):
        if self.fail_build:
            raise RuntimeError("build failed")
        self.template_builds.append(
            {
                "content": dockerfile_content,
                "context_dir": context_dir,
                "name": name,
                "cpu_count": cpu_count,
                "memory_mb": memory_mb,
                "on_build_logs": on_build_logs,
            }
        )
        return self.template_id

    def exec(self, sandbox, command, *, timeout=10, background=False, user=None):
        self.exec_commands.append(command)
        self.exec_timeouts.append(timeout)
        self.exec_background.append(background)
        self.exec_users.append(user)
        if "test -f /app/env/openenv.yaml" in command:
            return "found" if self.has_yaml else ""
        if command.startswith("cat /app/env/openenv.yaml"):
            return "spec_version: 1\nname: test\napp: server.app:app\nport: 8000\n"
        if "find /app" in command:
            return ""
        if "kill -0" in command:
            if not self.pid_ready:
                return "STARTING"
            return "DEAD" if self.dead_process else "RUNNING"
        if "cat /tmp/openenv-server.log" in command:
            return self.log
        return ""

    def host(self, sandbox, port):
        return self.host_value

    def kill(self, sandbox):
        self.killed += 1
        if self.fail_kill:
            raise RuntimeError("kill failed")


@pytest.fixture()
def adapter():
    return _FakeAdapter()


@pytest.fixture()
def provider(adapter):
    return NovitaSandboxProvider(_adapter=adapter)


@pytest.fixture(autouse=True)
def _fast_provider_sleep():
    """Avoid real sleeps in NovitaSandboxProvider (wait_for_ready)."""
    with patch("openenv.core.containers.runtime.novita_provider.time.sleep"):
        yield


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_is_container_provider(self, provider):
        assert isinstance(provider, ContainerProvider)

    def test_adapter_not_built_when_injected(self):
        """Injecting an adapter must not touch the SDK."""
        NovitaSandboxProvider(_adapter=_FakeAdapter())


# ---------------------------------------------------------------------------
# Tests: start_container — image resolution
# ---------------------------------------------------------------------------
class TestStartContainer:
    def test_registry_image(self, provider, adapter):
        url = provider.start_container("ghcr.io/org/echo-env:latest")
        assert url == f"https://{_SANDBOX_HOST}"
        assert adapter.created[0]["image"] == "ghcr.io/org/echo-env:latest"

    def test_constructor_image_used_when_start_omits_image(self, adapter):
        """A provider-owned image lets the outer client call start_container()."""
        provider = NovitaSandboxProvider(
            image="ghcr.io/org/env:latest",
            env_vars={"DEBUG": "1"},
            _adapter=adapter,
        )
        provider.start_container()
        created = adapter.created[0]
        assert created["image"] == "ghcr.io/org/env:latest"
        assert created["env_vars"] == {"DEBUG": "1"}

    def test_start_image_overrides_constructor_image(self, provider, adapter):
        provider = NovitaSandboxProvider(image="constructor:latest", _adapter=adapter)
        provider.start_container("explicit:latest")
        assert adapter.created[0]["image"] == "explicit:latest"

    def test_requires_image_from_constructor_or_start(self, provider):
        with pytest.raises(ValueError, match="requires an image"):
            provider.start_container()

    def test_base_url_has_https_scheme(self, provider):
        """get_host returns a bare host; the provider adds the scheme."""
        url = provider.start_container("img:latest")
        assert url.startswith("https://")


# ---------------------------------------------------------------------------
# Tests: port validation
# ---------------------------------------------------------------------------
class TestPortValidation:
    def test_port_none_accepted(self, provider):
        assert provider.start_container("img:latest", port=None)

    def test_port_8000_accepted(self, provider):
        assert provider.start_container("img:latest", port=8000)

    def test_other_port_raises(self, provider):
        with pytest.raises(ValueError, match="only supports port 8000"):
            provider.start_container("img:latest", port=3000)

    def test_port_not_validated_before_sandbox_guard(self, adapter):
        """An active-sandbox error wins over a port error (no orphan risk)."""
        provider = NovitaSandboxProvider(_adapter=adapter)
        provider.start_container("img:latest")
        with pytest.raises(RuntimeError, match="already has an active sandbox"):
            provider.start_container("img:latest", port=3000)


# ---------------------------------------------------------------------------
# Tests: create kwargs forwarding
# ---------------------------------------------------------------------------
class TestCreateKwargs:
    def test_timeout_and_metadata_forwarded(self, adapter):
        provider = NovitaSandboxProvider(
            _adapter=adapter,
            timeout=120,
            metadata={"idle_timeout": "900"},
        )
        provider.start_container("img:latest")
        created = adapter.created[0]
        assert created["timeout"] == 120
        assert created["metadata"] == {"idle_timeout": "900"}

    def test_default_timeout_is_one_hour(self, provider, adapter):
        provider.start_container("img:latest")
        assert adapter.created[0]["timeout"] == 3600

    def test_secure_omitted_when_none(self, provider, adapter):
        provider.start_container("img:latest")
        assert adapter.created[0]["secure"] is None

    def test_env_vars_none_when_unset(self, provider, adapter):
        provider.start_container("img:latest")
        assert adapter.created[0]["env_vars"] is None

    def test_unknown_start_option_raises(self, provider):
        with pytest.raises(ValueError, match="Unsupported NovitaSandboxProvider"):
            provider.start_container("img:latest", bogus=1)

    def test_wait_timeout_accepted_and_ignored(self, provider, adapter):
        """AutoEnv.from_env() always forwards wait_timeout; rejecting it would
        make the whole AutoEnv path unusable. No provider ever sees its value
        (`_bootstrap_container` calls wait_for_ready without a timeout), so it
        is dropped rather than rejected."""
        url = provider.start_container(
            "img:latest", wait_timeout=30.0, env_vars={"A": "1"}
        )
        assert url.startswith("https://")
        # The received kwargs are exactly the ones create_sandbox understands.
        assert set(adapter.created[0]) == {
            "image",
            "template",
            "env_vars",
            "timeout",
            "metadata",
            "secure",
            "sandbox",
        }

    def test_typo_still_rejected_alongside_wait_timeout(self, provider):
        """Dropping wait_timeout must not weaken the typo guard."""
        with pytest.raises(ValueError, match="Unsupported NovitaSandboxProvider"):
            provider.start_container("img:latest", wait_timeout=30.0, env_varz={})


# ---------------------------------------------------------------------------
# Tests: server command resolution
# ---------------------------------------------------------------------------
class TestServerCmd:
    def test_auto_discovered_cmd(self, provider, adapter):
        provider.start_container("img:latest")
        commands = adapter.exec_commands
        assert any(
            "cd /app/env && python -m uvicorn server.app:app" in c for c in commands
        )

    def test_explicit_cmd_used(self, adapter):
        provider = NovitaSandboxProvider(_adapter=adapter, cmd="python -m myserver")
        provider.start_container("img:latest")
        assert any("python -m myserver" in c for c in adapter.exec_commands)

    def test_kwargs_cmd_overrides_constructor(self, adapter):
        provider = NovitaSandboxProvider(_adapter=adapter, cmd="default-cmd")
        provider.start_container("img:latest", cmd="override-cmd")
        assert any("override-cmd" in c for c in adapter.exec_commands)

    def test_no_yaml_raises(self, adapter):
        adapter.has_yaml = False
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="Could not find openenv.yaml"):
            provider.start_container("img:latest")

    def test_working_directory_prepended(self, adapter):
        provider = NovitaSandboxProvider(
            _adapter=adapter, cmd="serve.sh", working_directory="/app/env"
        )
        provider.start_container("img:latest")
        assert any("cd /app/env && serve.sh" in c for c in adapter.exec_commands)


# ---------------------------------------------------------------------------
# Tests: launch (SDK background command + PID capture)
# ---------------------------------------------------------------------------
class TestLaunch:
    def _launch_command(self, adapter):
        return next(c for c in adapter.exec_commands if "echo $$" in c)

    def test_background_launch_with_pid_capture(self, provider, adapter):
        provider.start_container("img:latest")
        launch = self._launch_command(adapter)
        assert "echo $$ > /tmp/openenv-server.pid" in launch
        assert "/tmp/openenv-server.log" in launch
        assert adapter.exec_timeouts[-1] == 0
        assert adapter.exec_background[-1] is True
        assert adapter.exec_users[-1] == "root"

    def test_liveness_probe_uses_server_user(self, provider, adapter):
        adapter.dead_process = True
        provider.start_container("img:latest")
        url = provider.base_url

        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="server process died"):
                provider.wait_for_ready(url)
        assert adapter.exec_users[-1] == "root"

    def test_cmd_is_shlex_quoted(self, adapter):
        provider = NovitaSandboxProvider(
            _adapter=adapter, cmd="sh -c 'echo hi; sleep 1'"
        )
        provider.start_container("img:latest")
        launch = self._launch_command(adapter)
        # Quoted, so the semicolon cannot split into a second command.
        parsed = shlex.split(launch)
        assert parsed[:2] == ["sh", "-c"]
        nested = shlex.split(parsed[2])
        assert nested[:2] == ["echo", "$$"]
        assert "exec" in nested and "sh" in nested
        assert "sh -c 'echo hi; sleep 1'" in nested


# ---------------------------------------------------------------------------
# Tests: single-sandbox guard and failure cleanup
# ---------------------------------------------------------------------------
class TestLifecycleGuards:
    def test_second_start_raises(self, provider):
        provider.start_container("img:latest")
        with pytest.raises(RuntimeError, match="already has an active sandbox"):
            provider.start_container("img:latest")

    def test_create_failure_does_not_kill(self, adapter):
        """A create failure created nothing, so nothing is killed."""
        adapter.fail_create = True
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(RuntimeError, match="create failed"):
            provider.start_container("img:latest")
        assert adapter.killed == 0

    def test_start_failure_after_create_cleans_up(self, adapter):
        """A failure after the sandbox exists kills it."""
        adapter.has_yaml = False  # discovery fails after create
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="Could not find openenv.yaml"):
            provider.start_container("img:latest")
        assert adapter.killed == 1
        assert provider._sandbox is None

    def test_cleanup_failure_does_not_mask_root_cause(self, adapter):
        """A failing kill must not replace the original error."""
        adapter.has_yaml = False
        adapter.fail_kill = True
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="Could not find openenv.yaml"):
            provider.start_container("img:latest")


# ---------------------------------------------------------------------------
# Tests: stop_container / close
# ---------------------------------------------------------------------------
class TestStopContainer:
    def test_kill_called(self, provider, adapter):
        provider.start_container("img:latest")
        provider.stop_container()
        assert adapter.killed == 1

    def test_stop_clears_state(self, provider):
        provider.start_container("img:latest")
        provider.stop_container()
        assert provider._sandbox is None
        assert provider._base_url is None

    def test_stop_noop_when_no_sandbox(self, provider, adapter):
        provider.stop_container()
        assert adapter.killed == 0

    def test_close_stops_sandbox(self, provider, adapter):
        provider.start_container("img:latest")
        provider.close()
        assert adapter.killed == 1

    def test_context_manager_kills_sandbox(self, adapter):
        with NovitaSandboxProvider(image="img:latest", _adapter=adapter) as p:
            p.start_container()
        assert adapter.killed == 1


# ---------------------------------------------------------------------------
# Tests: base_url property
# ---------------------------------------------------------------------------
class TestBaseUrl:
    def test_returns_start_url(self, provider):
        url = provider.start_container("img:latest")
        assert provider.base_url == url

    def test_raises_without_sandbox(self, provider):
        with pytest.raises(RuntimeError, match="no active base_url"):
            provider.base_url


# ---------------------------------------------------------------------------
# Tests: secure-URL enforcement (RFC 002 S1)
# ---------------------------------------------------------------------------
class TestSecureUrl:
    def test_plaintext_host_rejected(self, adapter):
        adapter.host_value = "8000-sbx.novita.ai:80"  # host with no scheme is fine...
        provider = NovitaSandboxProvider(_adapter=adapter)
        # ...the provider always prepends https, so this still passes.
        assert provider.start_container("img:latest").startswith("https://")

    def test_loopback_host_rejected(self, adapter):
        """NOVITA_DEBUG makes get_host return localhost; refuse to connect."""
        adapter.host_value = "localhost:8000"
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(RuntimeError, match="loopback host"):
            provider.start_container("img:latest")

    @pytest.mark.parametrize(
        "host", ["https://example.test:8000", "http://example.test:8000"]
    )
    def test_scheme_bearing_host_rejected(self, adapter, host):
        adapter.host_value = host
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(RuntimeError, match="invalid host"):
            provider.start_container("img:latest")

    def test_loopback_failure_cleans_up(self, adapter):
        adapter.host_value = "127.0.0.1:8000"
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(RuntimeError, match="loopback host"):
            provider.start_container("img:latest")
        assert adapter.killed == 1


# ---------------------------------------------------------------------------
# Tests: wait_for_ready
# ---------------------------------------------------------------------------
class TestWaitForReady:
    def test_health_polling(self, provider):
        url = provider.start_container("img:latest")
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            provider.wait_for_ready(url)
            assert f"{url}/health" == mock_get.call_args.args[0]

    def test_timeout_raises(self, provider):
        url = provider.start_container("img:latest")

        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("nope")):
            with pytest.raises(TimeoutError, match="did not become ready"):
                provider.wait_for_ready(url, timeout_s=0.1)

    def test_dead_process_raises_without_surfacing_log(self, provider, adapter):
        """Crash is detected early, and the log is withheld by default (S4)."""
        adapter.dead_process = True
        url = provider.start_container("img:latest")

        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="server process died") as exc:
                provider.wait_for_ready(url)
        assert "server crashed" not in str(exc.value)

    def test_missing_pid_file_is_still_starting(self, provider, adapter):
        """The SDK background launch may write the PID after the first probe."""
        adapter.pid_ready = False
        url = provider.start_container("img:latest")

        import requests

        responses = [requests.ConnectionError("refused"), MagicMock(status_code=200)]
        with patch("requests.get", side_effect=responses):
            provider.wait_for_ready(url)

    def test_dead_process_log_surfaced_when_opted_in(self, adapter):
        adapter.dead_process = True
        provider = NovitaSandboxProvider(_adapter=adapter, surface_server_logs=True)
        url = provider.start_container("img:latest")

        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError) as exc:
                provider.wait_for_ready(url)
        assert "server crashed" in str(exc.value)

    def test_secret_redacted_from_surfaced_log(self, adapter):
        adapter.dead_process = True
        provider = NovitaSandboxProvider(
            _adapter=adapter,
            env_vars={"API_KEY": "super-secret-value"},
            surface_server_logs=True,
        )
        adapter.log = "using key super-secret-value"
        url = provider.start_container("img:latest")

        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError) as exc:
                provider.wait_for_ready(url)
        assert "super-secret-value" not in str(exc.value)
        assert "***" in str(exc.value)


# ---------------------------------------------------------------------------
# Tests: image_from_dockerfile -> template build
# ---------------------------------------------------------------------------
class TestImageFromDockerfile:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        NovitaSandboxProvider._dockerfile_registry.clear()
        yield
        NovitaSandboxProvider._dockerfile_registry.clear()

    @staticmethod
    def _write_dockerfile(tmp_path, body):
        """Write <tmp>/server/Dockerfile so context_dir defaults to <tmp>."""
        server = tmp_path / "server"
        server.mkdir(exist_ok=True)
        dockerfile = server / "Dockerfile"
        dockerfile.write_text(body)
        return dockerfile

    def test_returns_template_ref(self, tmp_path):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\nRUN echo hi\n")
        result = NovitaSandboxProvider.image_from_dockerfile(str(df))
        assert result == f"template:{df.resolve()}"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NovitaSandboxProvider.image_from_dockerfile(str(tmp_path / "nope"))

    def test_context_dir_not_found(self, tmp_path):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        with pytest.raises(ValueError, match="context_dir"):
            NovitaSandboxProvider.image_from_dockerfile(
                str(df), context_dir="/no/such/dir"
            )

    def test_does_not_build_eagerly(self, tmp_path):
        """No adapter is involved, so this works without credentials."""
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        NovitaSandboxProvider.image_from_dockerfile(str(df))
        assert str(df.resolve()) in NovitaSandboxProvider._dockerfile_registry

    def test_copy_source_missing_raises(self, tmp_path):
        df = self._write_dockerfile(
            tmp_path, "FROM python:3.11\nCOPY nonexistent_dir /app\n"
        )
        with pytest.raises(ValueError, match="COPY source.*not found"):
            NovitaSandboxProvider.image_from_dockerfile(str(df))

    def test_start_container_builds_and_uses_template(self, tmp_path, adapter):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\nRUN echo hi\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()

        assert len(adapter.template_builds) == 1
        assert adapter.created[0]["template"] == adapter.template_id
        assert adapter.created[0]["image"] is None

    def test_registration_after_construction_also_works(self, tmp_path, adapter):
        """Constructor order must not matter.

        The registry is class-level, so a provider built BEFORE
        `image_from_dockerfile()` still sees the entry. An instance-level copy
        taken in `__init__` (the previous behavior) made this order fail with a
        misleading "call image_from_dockerfile() first" error.
        """
        provider = NovitaSandboxProvider(_adapter=adapter)
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider.start_container(image)

        assert len(adapter.template_builds) == 1
        assert adapter.created[0]["template"] == adapter.template_id

    def test_start_container_accepts_template_ref_from_kwarg(self, tmp_path, adapter):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(_adapter=adapter)
        provider.start_container(image)
        assert len(adapter.template_builds) == 1

    def test_unregistered_template_ref_raises(self, adapter):
        provider = NovitaSandboxProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="No registered Dockerfile metadata"):
            provider.start_container("template:/no/such/path")

    def test_build_failure_leaves_no_sandbox(self, tmp_path, adapter):
        """The build runs before create, so a failure can't orphan a sandbox."""
        adapter.fail_build = True
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        with pytest.raises(RuntimeError, match="build failed"):
            provider.start_container()
        assert adapter.created == []
        assert adapter.killed == 0

    def test_default_template_name_derived_from_env_dir(self, tmp_path, adapter):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()
        # <tmp>/server/Dockerfile -> grandparent is <tmp>, so the name is
        # "openenv-<tmp-slug>-<content-hash>".
        name = adapter.template_builds[0]["name"]
        assert name.startswith("openenv-")
        assert name.endswith(
            "-" + __import__("hashlib").sha256(df.read_bytes()).hexdigest()[:8]
        )

    def test_template_name_hashes_content(self, tmp_path, adapter):
        """Two different Dockerfiles must not collide on one template name."""
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        first = NovitaSandboxProvider._default_template_name(str(df))
        df.write_text("FROM python:3.12\n")
        second = NovitaSandboxProvider._default_template_name(str(df))
        assert first != second

    def test_explicit_template_name_wins(self, tmp_path, adapter):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(
            image=image, template_name="my-env", _adapter=adapter
        )
        provider.start_container()
        assert adapter.template_builds[0]["name"] == "my-env"

    def test_cpu_and_memory_forwarded_to_build(self, tmp_path, adapter):
        df = self._write_dockerfile(tmp_path, "FROM python:3.11\n")
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(
            image=image, cpu_count=4, memory_mb=2048, _adapter=adapter
        )
        provider.start_container()
        build = adapter.template_builds[0]
        assert build["cpu_count"] == 4
        assert build["memory_mb"] == 2048

    def test_multistage_dockerfile_is_flattened(self, tmp_path, adapter):
        """A two-stage OpenEnv Dockerfile builds as one flattened stage."""
        df = self._write_dockerfile(
            tmp_path,
            "ARG BASE_IMAGE=python:3.11\n"
            "FROM ${BASE_IMAGE} AS builder\n"
            "WORKDIR /app/env\n"
            "RUN uv sync\n"
            "FROM ${BASE_IMAGE}\n"
            "WORKDIR /app\n"
            "COPY --from=builder /app/env /app/env\n"
            "COPY --from=builder /app/env/.venv /app/.venv\n"
            'CMD ["sh","-c","uvicorn server.app:app"]\n',
        )
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()

        content = adapter.template_builds[0]["content"]
        assert content.count("FROM ") == 1
        assert "${BASE_IMAGE}" not in content
        assert "python:3.11" in content
        # The venv copy becomes an in-place cp; the same-path copy is dropped.
        assert "cp -a /app/env/.venv /app/.venv" in content
        assert "COPY --from=" not in content

    def test_incompatible_multistage_raises_with_registry_hint(self, tmp_path):
        """Stages with different base images cannot be flattened."""
        df = self._write_dockerfile(
            tmp_path,
            "FROM python:3.11 AS builder\n"
            "RUN echo hi\n"
            "FROM alpine:3.19\n"
            "COPY --from=builder /x /x\n",
        )
        with pytest.raises(ValueError, match="openenv build"):
            NovitaSandboxProvider.image_from_dockerfile(str(df))

    def test_buildkit_mount_stripped(self, tmp_path, adapter):
        df = self._write_dockerfile(
            tmp_path,
            "FROM python:3.11\nRUN --mount=type=cache,target=/root/.cache/uv uv sync\n",
        )
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()
        assert "--mount=" not in adapter.template_builds[0]["content"]

    def test_platform_flag_stripped_from_from(self, tmp_path, adapter):
        df = self._write_dockerfile(
            tmp_path, "FROM --platform=linux/amd64 python:3.10-slim\nRUN echo hi\n"
        )
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()
        content = adapter.template_builds[0]["content"]
        assert "--platform" not in content
        assert "FROM python:3.10-slim" in content

    def test_context_dir_used_as_file_context_path(self, tmp_path, adapter):
        server = tmp_path / "server"
        server.mkdir()
        df = server / "Dockerfile"
        df.write_text("FROM python:3.11\nCOPY . /app/env\n")
        image = NovitaSandboxProvider.image_from_dockerfile(
            str(df), context_dir=str(tmp_path)
        )
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()
        assert adapter.template_builds[0]["context_dir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Tests: on_build_logs default
# ---------------------------------------------------------------------------
class TestBuildLogsDefault:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        NovitaSandboxProvider._dockerfile_registry.clear()
        yield
        NovitaSandboxProvider._dockerfile_registry.clear()

    @staticmethod
    def _write_dockerfile(tmp_path):
        server = tmp_path / "server"
        server.mkdir(exist_ok=True)
        dockerfile = server / "Dockerfile"
        dockerfile.write_text("FROM python:3.11\nRUN echo hi\n")
        return dockerfile

    def test_default_is_none(self):
        """Build logs are off by default so nothing unredacted is printed.

        The entries are not passed through `_redact` (RFC 002 S4), so surfacing
        them is the caller's explicit choice rather than the default.
        """
        import inspect

        sig = inspect.signature(NovitaSandboxProvider.__init__)
        assert sig.parameters["on_build_logs"].default is None

    def test_default_prints_nothing(self, tmp_path, adapter, capsys):
        df = self._write_dockerfile(tmp_path)
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        provider = NovitaSandboxProvider(image=image, _adapter=adapter)
        provider.start_container()
        assert capsys.readouterr().out == ""

    def test_custom_callback_receives_entries(self, tmp_path, adapter):
        df = self._write_dockerfile(tmp_path)
        image = NovitaSandboxProvider.image_from_dockerfile(str(df))
        collected = []
        provider = NovitaSandboxProvider(
            image=image, on_build_logs=collected.append, _adapter=adapter
        )
        provider.start_container()
        # The adapter always forwards whatever it was given (including None).
        # `==` not `is`: `list.append` hands back a fresh bound-method object
        # on each attribute access, so an identity check never holds.
        forwarded = adapter.template_builds[0]["on_build_logs"]
        assert forwarded == collected.append
        forwarded("entry")
        assert collected == ["entry"]


# ---------------------------------------------------------------------------
# Tests: Dockerfile rewriting helpers
# ---------------------------------------------------------------------------
class TestDockerfileRewriting:
    def test_strip_mount_flags_single(self):
        from openenv.core.containers.runtime.novita_provider import _strip_mount_flags

        out = _strip_mount_flags("RUN --mount=type=cache,target=/x uv sync\n")
        assert out.strip() == "RUN uv sync"

    def test_strip_mount_flags_multiline(self):
        from openenv.core.containers.runtime.novita_provider import _strip_mount_flags

        content = (
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n    uv sync --frozen\n"
        )
        out = _strip_mount_flags(content)
        assert "--mount=" not in out
        assert "uv sync --frozen" in out

    @pytest.mark.parametrize(
        "mount",
        [
            "type=secret,id=token",
            "type=ssh",
            "type=bind,source=.,target=/src",
            "target=/src",
        ],
    )
    def test_strip_mount_flags_rejects_unsupported_types(self, mount):
        from openenv.core.containers.runtime.novita_provider import _strip_mount_flags

        with pytest.raises(ValueError, match="only supports explicit"):
            _strip_mount_flags(f"RUN --mount={mount} echo hi\n")

    def test_strip_mount_flags_rejects_unsupported_mount_on_later_line(self):
        from openenv.core.containers.runtime.novita_provider import _strip_mount_flags

        content = (
            "RUN --mount=type=cache,target=/root/.cache/uv \\\n"
            "    --mount=type=secret,id=token \\\n"
            "    uv sync\n"
        )
        with pytest.raises(ValueError, match="only supports explicit"):
            _strip_mount_flags(content)

    def test_strip_mount_flags_preserves_non_run_lines(self):
        from openenv.core.containers.runtime.novita_provider import _strip_mount_flags

        content = "FROM python:3.11\nCOPY . /app\nRUN echo hi\n"
        assert _strip_mount_flags(content) == content

    def test_resolve_arg_in_from(self):
        from openenv.core.containers.runtime.novita_provider import (
            _resolve_from_references,
        )

        out = _resolve_from_references(
            "ARG BASE_IMAGE=python:3.11\nFROM ${BASE_IMAGE}\nRUN echo hi\n"
        )
        assert "FROM python:3.11" in out
        assert "${BASE_IMAGE}" not in out.split("\n")[1]

    def test_resolve_prefix_colliding_unbraced_args(self):
        """Shorter ARG names must not corrupt longer sibling references.

        Declaring ``ARG BASE`` before ``ARG BASE_IMAGE`` used to rewrite
        ``FROM $BASE_IMAGE`` / ``FROM ${BASE_IMAGE}`` into
        ``FROM python:3.12_IMAGE`` because ``$BASE`` is a prefix of
        ``$BASE_IMAGE``.
        """
        from openenv.core.containers.runtime.novita_provider import (
            _resolve_from_references,
        )

        dockerfile = (
            "ARG BASE=python:3.12\n"
            "ARG BASE_IMAGE=python:3.12-slim\n"
            "FROM $BASE_IMAGE\n"
            "RUN echo hi\n"
        )
        out = _resolve_from_references(dockerfile)
        assert "FROM python:3.12-slim" in out
        assert "python:3.12_IMAGE" not in out

        braced = (
            "ARG BASE=python:3.12\n"
            "ARG BASE_IMAGE=python:3.12-slim\n"
            "FROM ${BASE_IMAGE}\n"
        )
        out_braced = _resolve_from_references(braced)
        assert "FROM python:3.12-slim" in out_braced
        assert "python:3.12_IMAGE" not in out_braced

    def test_arg_after_first_from_not_in_scope(self):
        from openenv.core.containers.runtime.novita_provider import (
            _resolve_from_references,
        )

        # A later ARG must not rewrite an earlier FROM.
        out = _resolve_from_references("FROM python:3.11\nARG X=alpine\nRUN echo hi\n")
        assert "FROM python:3.11" in out

    def test_flatten_drops_same_path_copy(self):
        from openenv.core.containers.runtime.novita_provider import _flatten_multistage

        out = _flatten_multistage(
            "FROM python:3.11 AS builder\nRUN uv sync\nFROM python:3.11\n"
            "COPY --from=builder /app/env /app/env\n"
        )
        assert out.count("FROM ") == 1
        assert "COPY --from=" not in out

    def test_flatten_rejects_unknown_stage(self):
        from openenv.core.containers.runtime.novita_provider import _flatten_multistage

        with pytest.raises(ValueError, match="does not reference the builder"):
            _flatten_multistage(
                "FROM python:3.11 AS builder\nRUN x\nFROM python:3.11\n"
                "COPY --from=other /a /b\n"
            )

    @pytest.mark.parametrize(
        "copy_line",
        [
            "COPY --chown=app:app --from=builder /a /b",
            "COPY --from=builder --chown=app:app /a /b",
            "COPY --from=builder /a /b /c",
        ],
    )
    def test_flatten_rejects_unflattened_copy_from(self, copy_line):
        from openenv.core.containers.runtime.novita_provider import _flatten_multistage

        with pytest.raises(ValueError, match="COPY --from"):
            _flatten_multistage(
                "FROM python:3.11 AS builder\nRUN echo hi\nFROM python:3.11\n"
                f"{copy_line}\n"
            )

    def test_single_stage_passthrough(self):
        from openenv.core.containers.runtime.novita_provider import _flatten_multistage

        content = "FROM python:3.11\nRUN echo hi\n"
        assert _flatten_multistage(content) == content

    def test_flatten_quotes_copy_paths_for_shell(self):
        from openenv.core.containers.runtime.novita_provider import _flatten_multistage

        out = _flatten_multistage(
            "FROM python:3.11 AS builder\nRUN echo hi\nFROM python:3.11\n"
            "COPY --from=builder /app/source;id /out/$(id)\n"
        )

        assert (
            "RUN mkdir -p $(dirname '/out/$(id)') && "
            "cp -a '/app/source;id' '/out/$(id)'"
        ) in out


# ---------------------------------------------------------------------------
# Tests: _DefaultNovitaAdapter against a fake novita_sandbox SDK
# ---------------------------------------------------------------------------
def _install_fake_novita():
    """Install a minimal fake ``novita_sandbox`` package into sys.modules.

    Pins the real SDK surface the adapter depends on: ``Novita.sandbox.create``
    with the kwargs the provider forwards, and the ``wait_for_timeout`` readiness
    helper. If the SDK renames any of these, this test fails.
    """
    mod = types.ModuleType("novita_sandbox")

    calls: dict = {"create_kwargs": None}

    class _SandboxNS:
        def create(self, **kwargs):
            calls["create_kwargs"] = kwargs
            sandbox = MagicMock()
            sandbox.get_host = MagicMock(return_value=_SANDBOX_HOST)
            sandbox.commands.run = MagicMock(
                return_value=types.SimpleNamespace(
                    stdout="ok", stderr="", exit_code=0, error=None
                )
            )
            sandbox.kill = MagicMock(return_value=True)
            return sandbox

    class _FakeBuilder:
        def __init__(self):
            self.start_cmd = None
            self.ready_cmd = None
            self.user = None

        def set_user(self, user):
            self.user = user
            return self

        def set_start_cmd(self, start_cmd, ready_cmd):
            self.start_cmd = start_cmd
            self.ready_cmd = ready_cmd
            return self

    class _FakeTemplateClass:
        """Pins the Template surface `_DefaultNovitaAdapter.build_template` uses.

        Mirrors the real shape: `Template(...)` is constructed with the build
        context, `.from_dockerfile(...)` returns a builder, and
        `novita.template.build(builder, name, **kwargs)` returns the build info.
        """

        def __init__(self, file_context_path=None, file_ignore_patterns=None):
            calls["template_context"] = file_context_path
            self.builder = _FakeBuilder()

        def from_dockerfile(self, content):
            calls["dockerfile_content"] = content
            return self.builder

        @staticmethod
        def build(builder, name, **kwargs):
            calls["build_name"] = name
            calls["build_builder"] = builder
            calls["build_kwargs"] = kwargs
            return types.SimpleNamespace(template_id="tpl-built-1")

    class _TemplateNS:
        def build(self, builder, name, **kwargs):
            calls["template_namespace_build"] = {
                "api_key": calls["api_key"],
                "domain": calls["domain"],
            }
            return _FakeTemplateClass.build(builder, name, **kwargs)

    class _FakeNovita:
        def __init__(self, api_key=None, domain=None, **kwargs):
            calls["api_key"] = api_key
            calls["domain"] = domain
            self.sandbox = _SandboxNS()
            self.template = _TemplateNS()

    class _ReadyCmd:
        def __init__(self, cmd):
            self._cmd = cmd

        def get_cmd(self):
            return self._cmd

    class _CommandExitException(Exception):
        def __init__(self, stderr="", stdout="", exit_code=1, error=None):
            self.stdout, self.stderr, self.exit_code, self.error = (
                stdout,
                stderr,
                exit_code,
                error,
            )

    mod.Novita = _FakeNovita
    mod.Template = _FakeTemplateClass
    mod.CommandExitException = _CommandExitException
    mod.wait_for_timeout = lambda ms: _ReadyCmd(f"sleep {ms}")
    mod.wait_for_url = lambda url, status_code=200: _ReadyCmd(f"curl {url}")

    sys.modules["novita_sandbox"] = mod
    return mod, calls


class TestDefaultAdapter:
    def test_uses_env_var_fallbacks_when_unset(self):
        """None for api_key/domain lets the SDK read its own env vars."""
        _, calls = _install_fake_novita()
        try:
            _DefaultNovitaAdapter(api_key=None, domain=None)
            assert calls["api_key"] is None
            assert calls["domain"] is None
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_explicit_credentials_passed_through(self):
        _, calls = _install_fake_novita()
        try:
            _DefaultNovitaAdapter(api_key="k-123", domain="us-phx-1.sandbox.novita.ai")
            assert calls["api_key"] == "k-123"
            assert calls["domain"] == "us-phx-1.sandbox.novita.ai"
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_create_sandbox_kwargs_match_sdk_signature(self):
        """The kwargs the provider forwards must match Sandbox.create's params."""
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            adapter.create_sandbox(
                image="img:latest",
                template=None,
                env_vars={"A": "1"},
                timeout=3600,
                metadata={"idle_timeout": "900"},
                secure=None,
            )
            kwargs = calls["create_kwargs"]
            assert kwargs["image"] == "img:latest"
            assert kwargs["envs"] == {"A": "1"}
            assert kwargs["timeout"] == 3600
            assert kwargs["metadata"] == {"idle_timeout": "900"}
            # secure=None must be omitted, not forwarded as a null.
            assert "secure" not in kwargs
            # A registry image and a template are mutually exclusive sources.
            assert "template" not in kwargs
            # Network posture is left to the SDK default, matching DaytonaProvider
            # -- OpenEnv does not set an egress policy for either provider.
            assert "allow_internet_access" not in kwargs
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_create_sandbox_with_template_id(self):
        """A built template id is passed as `template`, with no image."""
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            adapter.create_sandbox(
                image=None,
                template="tpl-built-1",
                env_vars=None,
                timeout=3600,
                metadata=None,
                secure=None,
            )
            kwargs = calls["create_kwargs"]
            assert kwargs["template"] == "tpl-built-1"
            assert "image" not in kwargs
            # No image means no image-resolution build block either.
            assert "build" not in kwargs
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_start_cmd_pins_keepalive_not_image_cmd(self):
        """The build block must override the image's own CMD.

        from_image defaults to inherit_config=True, which would otherwise adopt
        the image's ENTRYPOINT/CMD (the uvicorn server) as the template start
        command and bind port 8000 before the provider launches its own copy.
        """
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            adapter.create_sandbox(
                image="img:latest",
                template=None,
                env_vars=None,
                timeout=3600,
                metadata=None,
                secure=None,
            )
            build = calls["create_kwargs"]["build"]
            assert build["cmd"] == "sleep infinity"
            # Sandbox.create fingerprints this mapping with json.dumps before
            # converting it into a template builder.
            import json

            json.dumps(build)
            assert build["ready_cmd"] == "sleep 5000"
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_build_template_uses_dockerfile_and_pins_keepalive(self):
        """build_template drives Template.from_dockerfile -> build -> id."""
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(
                api_key="k-123", domain="us-phx-1.sandbox.novita.ai"
            )
            template_id = adapter.build_template(
                dockerfile_content="FROM python:3.12\nRUN echo hi\n",
                context_dir="/ctx",
                name="openenv-echo-abc",
                cpu_count=4,
                memory_mb=2048,
                on_build_logs=None,
            )
            assert template_id == "tpl-built-1"
            assert calls["template_context"] == "/ctx"
            assert calls["dockerfile_content"] == "FROM python:3.12\nRUN echo hi\n"
            assert calls["build_name"] == "openenv-echo-abc"
            assert calls["build_kwargs"]["cpu_count"] == 4
            assert calls["build_kwargs"]["memory_mb"] == 2048
            assert "on_build_logs" not in calls["build_kwargs"]
            assert calls["template_namespace_build"] == {
                "api_key": "k-123",
                "domain": "us-phx-1.sandbox.novita.ai",
            }
            # The template's own start command is the keepalive, not the
            # Dockerfile's CMD -- otherwise port 8000 is taken before launch.
            builder = calls["build_builder"]
            assert builder.start_cmd == "sleep infinity"
            assert builder.ready_cmd.get_cmd()
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_build_template_forwards_on_build_logs(self):
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            log_fn = MagicMock()
            adapter.build_template(
                dockerfile_content="FROM python:3.12\n",
                context_dir="/ctx",
                name="n",
                cpu_count=2,
                memory_mb=1024,
                on_build_logs=log_fn,
            )
            assert calls["build_kwargs"]["on_build_logs"] is log_fn
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_build_template_sets_user_root(self):
        """The template must run as root, not the parser's default "user".

        Novita's Dockerfile parser rewrites USER to "user" when the Dockerfile
        declares none. The OpenEnv image installs under /app, which root creates
        and "user" cannot write, so a non-root template breaks every in-process
        command the environment runs (observed as "Permission denied" on
        /app/resources and the task's git work tree).
        """
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            adapter.build_template(
                dockerfile_content="FROM python:3.12\nWORKDIR /app\n",
                context_dir="/ctx",
                name="n",
                cpu_count=2,
                memory_mb=1024,
                on_build_logs=None,
            )
            assert calls["build_builder"].user == "root"
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_build_template_preserves_explicit_dockerfile_user(self):
        _, calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            adapter.build_template(
                dockerfile_content=("FROM python:3.12\nUSER app\nWORKDIR /app\n"),
                context_dir="/ctx",
                name="n",
                cpu_count=2,
                memory_mb=1024,
                on_build_logs=None,
            )
            assert calls["build_builder"].user is None
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_exec_swallows_nonzero_exit(self):
        """Probes exit non-zero by design; stdout must still come back."""
        mod, _ = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            sandbox = MagicMock()
            sandbox.commands.run = MagicMock(
                side_effect=mod.CommandExitException(
                    stdout="", stderr="nope", exit_code=1, error=None
                )
            )
            assert adapter.exec(sandbox, "false") == ""
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_exec_returns_stdout(self):
        _, _calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            sandbox = MagicMock()
            sandbox.commands.run = MagicMock(
                return_value=types.SimpleNamespace(stdout="found\n")
            )
            assert adapter.exec(sandbox, "echo found") == "found\n"
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_exec_forwards_background(self):
        _, _calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            sandbox = MagicMock()
            sandbox.commands.run = MagicMock(return_value=object())
            adapter.exec(sandbox, "python -m server", background=True, user="root")
            sandbox.commands.run.assert_called_once_with(
                "python -m server", timeout=10, background=True, user="root"
            )
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_missing_sdk_raises_install_hint(self):
        sys.modules["novita_sandbox"] = None  # forces ImportError
        try:
            with pytest.raises(RuntimeError, match=r"openenv\[novita\]"):
                _DefaultNovitaAdapter(api_key=None, domain=None)
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_host_uses_get_host(self):
        _, _calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            sandbox = MagicMock()
            sandbox.get_host = MagicMock(return_value=_SANDBOX_HOST)
            assert adapter.host(sandbox, 8000) == _SANDBOX_HOST
            sandbox.get_host.assert_called_once_with(8000)
        finally:
            sys.modules.pop("novita_sandbox", None)

    def test_kill_uses_sandbox_kill(self):
        _, _calls = _install_fake_novita()
        try:
            adapter = _DefaultNovitaAdapter(api_key="k", domain=None)
            sandbox = MagicMock()
            adapter.kill(sandbox)
            sandbox.kill.assert_called_once_with()
        finally:
            sys.modules.pop("novita_sandbox", None)
