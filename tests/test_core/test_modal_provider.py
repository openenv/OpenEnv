# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ModalProvider.

Provider-behavior tests inject a duck-typed fake adapter (``_adapter=``), so
they run without ``pip install modal``. A separate test installs a minimal fake
``modal`` module to pin the real ``_DefaultModalAdapter`` SDK shape (including
the Sandbox v2 feature check).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.modal_provider import (
    _DefaultModalAdapter,
    ModalProvider,
)
from openenv.core.containers.runtime.providers import ContainerProvider


# ---------------------------------------------------------------------------
# Fake adapter (duck-typed to _DefaultModalAdapter)
# ---------------------------------------------------------------------------
class _FakeImage:
    def __init__(self, kind: str, ref: str):
        self.kind = kind
        self.ref = ref


class _FakeAdapter:
    def __init__(self):
        self.created: list[dict] = []
        self.exec_commands: list[str] = []
        self.terminated = 0
        self.tunnel = "https://sb-abc-8000.modal.host"
        self.fail_create = False
        self.fail_terminate = False
        self.fail_tunnel = False
        self.dead_process = False
        self.log = "server crashed"
        self.has_yaml = True

    def image_from_registry(self, tag):
        return _FakeImage("registry", tag)

    def image_from_dockerfile(self, dockerfile_path, context_dir):
        return _FakeImage("dockerfile", dockerfile_path)

    def create_sandbox(self, *, image, encrypted_ports, timeout, env, extra):
        if self.fail_create:
            raise RuntimeError("create failed")
        sandbox = object()
        self.created.append(
            {
                "image": image,
                "encrypted_ports": encrypted_ports,
                "timeout": timeout,
                "env": env,
                "extra": extra,
                "sandbox": sandbox,
            }
        )
        return sandbox

    def exec(self, sandbox, command, *, timeout=10):
        self.exec_commands.append(command)
        if "test -f /app/env/openenv.yaml" in command:
            return "found" if self.has_yaml else ""
        if command.startswith("cat /app/env/openenv.yaml"):
            return "spec_version: 1\nname: test\napp: server.app:app\nport: 8000\n"
        if "find /app" in command:
            return ""
        if "kill -0" in command:
            return "DEAD" if self.dead_process else "RUNNING"
        if "cat /tmp/openenv-server.log" in command:
            return self.log
        return ""

    def tunnel_url(self, sandbox, port):
        if self.fail_tunnel:
            raise RuntimeError("tunnel failed")
        return self.tunnel

    def terminate(self, sandbox):
        self.terminated += 1
        if self.fail_terminate:
            raise RuntimeError("terminate failed")


@pytest.fixture()
def adapter():
    return _FakeAdapter()


@pytest.fixture()
def provider(adapter):
    return ModalProvider(app_name="test-app", _adapter=adapter)


@pytest.fixture(autouse=True)
def _fast_provider_sleep():
    """Avoid real sleeps in ModalProvider (wait_for_ready)."""
    with patch("openenv.core.containers.runtime.modal_provider.time.sleep"):
        yield


@pytest.fixture(autouse=True)
def _clean_dockerfile_registry():
    """Clear the Dockerfile registry between tests."""
    ModalProvider._dockerfile_registry.clear()
    yield
    ModalProvider._dockerfile_registry.clear()


def _write_dockerfile(tmp_path, body):
    server = tmp_path / "server"
    server.mkdir(exist_ok=True)
    df = server / "Dockerfile"
    df.write_text(body)
    return df


# ---------------------------------------------------------------------------
# Tests: start_container
# ---------------------------------------------------------------------------
class TestStartContainer:
    def test_returns_tunnel_url(self, provider):
        assert provider.start_container("echo-env:latest") == (
            "https://sb-abc-8000.modal.host"
        )

    def test_rejects_non_8000_port(self, provider):
        with pytest.raises(ValueError, match="only supports port 8000"):
            provider.start_container("echo-env:latest", port=9000)

    def test_port_8000_allowed(self, provider):
        assert provider.start_container("echo-env:latest", port=8000).startswith(
            "https://"
        )

    def test_registry_image_used(self, provider, adapter):
        provider.start_container("echo-env:latest")
        image = adapter.created[0]["image"]
        assert image.kind == "registry"
        assert image.ref == "echo-env:latest"
        assert adapter.created[0]["encrypted_ports"] == [8000]

    def test_env_vars_forwarded(self, provider, adapter):
        provider.start_container("echo-env:latest", env_vars={"FOO": "bar"})
        assert adapter.created[0]["env"] == {"FOO": "bar"}

    def test_constructor_image_used_when_start_omits_image(self, adapter):
        provider = ModalProvider(
            _adapter=adapter,
            image="constructor-env:latest",
            env_vars={"FOO": "bar"},
        )
        provider.start_container()
        image = adapter.created[0]["image"]
        assert image.kind == "registry"
        assert image.ref == "constructor-env:latest"
        assert adapter.created[0]["env"] == {"FOO": "bar"}

    def test_start_image_overrides_constructor_image(self, adapter):
        provider = ModalProvider(_adapter=adapter, image="constructor-env:latest")
        provider.start_container("explicit-env:latest")
        image = adapter.created[0]["image"]
        assert image.ref == "explicit-env:latest"

    def test_requires_image_from_constructor_or_start(self, adapter):
        provider = ModalProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="requires an image"):
            provider.start_container()

    def test_server_started_in_background(self, provider, adapter):
        provider.start_container("echo-env:latest")
        assert any(
            "nohup" in c and "uvicorn server.app:app" in c
            for c in adapter.exec_commands
        )

    def test_rejects_non_https_tunnel(self, adapter):
        adapter.tunnel = "http://insecure.modal.host"
        provider = ModalProvider(_adapter=adapter)
        with pytest.raises(RuntimeError, match="non-HTTPS"):
            provider.start_container("echo-env:latest")
        # Failed start cleans up the sandbox it created.
        assert adapter.terminated == 1


# ---------------------------------------------------------------------------
# Tests: cpu / memory resources
# ---------------------------------------------------------------------------
class TestResources:
    def test_cpu_memory_forwarded_as_limits(self, adapter):
        provider = ModalProvider(_adapter=adapter, cpu=2.0, memory=4096)
        provider.start_container("echo-env:latest")
        extra = adapter.created[0]["extra"]
        # Passed as the limit half of Modal's (request, limit) tuple.
        assert extra["cpu"] == (0.125, 2.0)
        assert extra["memory"] == (128, 4096)

    def test_no_resources_means_no_extra_keys(self, provider, adapter):
        provider.start_container("echo-env:latest")
        extra = adapter.created[0]["extra"]
        assert "cpu" not in extra
        assert "memory" not in extra

    def test_extra_kwargs_forwarded(self, provider, adapter):
        provider.start_container("echo-env:latest", gpu="A100")
        assert adapter.created[0]["extra"]["gpu"] == "A100"


# ---------------------------------------------------------------------------
# Tests: server command resolution
# ---------------------------------------------------------------------------
class TestServerCmd:
    def test_explicit_cmd_constructor(self, adapter):
        provider = ModalProvider(_adapter=adapter, cmd="python -m myserver")
        provider.start_container("echo-env:latest")
        assert any("python -m myserver" in c for c in adapter.exec_commands)

    def test_kwarg_cmd_overrides(self, provider, adapter):
        provider.start_container("echo-env:latest", cmd="custom-cmd")
        assert any("custom-cmd" in c for c in adapter.exec_commands)

    def test_discovers_from_yaml(self, provider, adapter):
        provider.start_container("echo-env:latest")
        assert any(
            "cd /app/env && python -m uvicorn server.app:app" in c
            for c in adapter.exec_commands
        )

    def test_no_yaml_raises(self, adapter):
        adapter.has_yaml = False
        provider = ModalProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="Could not find openenv.yaml"):
            provider.start_container("no-yaml:latest")
        # The created sandbox is cleaned up after the discovery failure.
        assert adapter.terminated == 1

    def test_falls_back_to_dockerfile_cmd(self, adapter, tmp_path):
        adapter.has_yaml = False
        df = _write_dockerfile(
            tmp_path,
            'FROM python:3.11\nCOPY . /app\nCMD ["uvicorn", "fallback:app"]\n',
        )
        image = ModalProvider.image_from_dockerfile(str(df))
        provider = ModalProvider(_adapter=adapter)
        provider.start_container(image)
        assert any("uvicorn fallback:app" in c for c in adapter.exec_commands)


# ---------------------------------------------------------------------------
# Tests: image_from_dockerfile
# ---------------------------------------------------------------------------
class TestImageFromDockerfile:
    def test_returns_dockerfile_uri(self, tmp_path):
        df = _write_dockerfile(tmp_path, "FROM python:3.11\nCMD uvicorn app:app\n")
        result = ModalProvider.image_from_dockerfile(str(df))
        assert result.startswith("dockerfile:")
        assert str(df.resolve()) in result

    def test_missing_dockerfile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModalProvider.image_from_dockerfile(str(tmp_path / "nope"))

    def test_bad_context_dir_raises(self, tmp_path):
        df = _write_dockerfile(tmp_path, "FROM python:3.11\n")
        with pytest.raises(ValueError, match="context_dir does not exist"):
            ModalProvider.image_from_dockerfile(str(df), context_dir="/no/such/dir")

    def test_parses_cmd_into_registry(self, tmp_path):
        df = _write_dockerfile(
            tmp_path, 'FROM python:3.11\nCMD ["uvicorn", "app:app"]\n'
        )
        ModalProvider.image_from_dockerfile(str(df))
        key = str(df.resolve())
        assert (
            ModalProvider._dockerfile_registry[key]["server_cmd"] == "uvicorn app:app"
        )

    def test_dockerfile_image_built_from_dockerfile(self, provider, adapter, tmp_path):
        df = _write_dockerfile(
            tmp_path, "FROM python:3.11\nCOPY . /app\nCMD uvicorn app:app\n"
        )
        image = ModalProvider.image_from_dockerfile(str(df))
        provider.start_container(image)
        assert adapter.created[0]["image"].kind == "dockerfile"

    def test_unregistered_dockerfile_raises(self, provider):
        with pytest.raises(ValueError, match="No registered Dockerfile metadata"):
            provider.start_container("dockerfile:/tmp/never-registered")


# ---------------------------------------------------------------------------
# Tests: _parse_app_field
# ---------------------------------------------------------------------------
class TestParseAppField:
    def test_simple(self):
        assert ModalProvider._parse_app_field("app: server.app:app") == "server.app:app"

    def test_missing(self):
        assert ModalProvider._parse_app_field("name: test\nport: 8000") is None

    def test_invalid_yaml(self):
        assert ModalProvider._parse_app_field("::: not yaml :::") is None


# ---------------------------------------------------------------------------
# Tests: lifecycle (double-start, stop, close, context manager)
# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_double_start_raises_and_preserves_existing_sandbox(
        self, provider, adapter
    ):
        provider.start_container("echo-env:latest")
        first = provider._sandbox

        with pytest.raises(RuntimeError, match="already has an active sandbox"):
            provider.start_container("other:latest")

        assert adapter.terminated == 0  # existing sandbox untouched
        assert len(adapter.created) == 1  # guard fired before a second create
        assert provider._sandbox is first

    def test_start_after_stop_is_allowed(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.stop_container()
        provider.start_container("other:latest")
        assert len(adapter.created) == 2

    def test_stop_terminates_sandbox(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.stop_container()
        assert adapter.terminated == 1
        assert provider._sandbox is None

    def test_stop_without_start_is_noop(self, provider, adapter):
        provider.stop_container()
        assert adapter.terminated == 0

    def test_close_terminates_sandbox(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.close()
        assert adapter.terminated == 1
        assert provider._sandbox is None

    def test_context_manager_terminates_on_exit(self, adapter):
        with ModalProvider(_adapter=adapter) as p:
            p.start_container("echo-env:latest")
        assert adapter.terminated == 1

    def test_context_manager_terminates_on_exception(self, adapter):
        with pytest.raises(RuntimeError, match="boom"):
            with ModalProvider(_adapter=adapter) as p:
                p.start_container("echo-env:latest")
                raise RuntimeError("boom")
        assert adapter.terminated == 1

    def test_is_container_provider(self, provider):
        assert isinstance(provider, ContainerProvider)


# ---------------------------------------------------------------------------
# Tests: cleanup does not mask the original error
# ---------------------------------------------------------------------------
class TestCleanup:
    def test_cleanup_failure_does_not_mask_original_error(self, adapter):
        adapter.fail_tunnel = True  # original failure (after sandbox exists)
        adapter.fail_terminate = True  # secondary failure during cleanup
        provider = ModalProvider(_adapter=adapter)

        with pytest.raises(RuntimeError, match="tunnel failed"):
            provider.start_container("echo-env:latest")

        assert adapter.terminated == 1  # cleanup was still attempted

    def test_create_failure_clears_secrets_without_terminating(self, adapter):
        adapter.fail_create = True
        provider = ModalProvider(_adapter=adapter)

        with pytest.raises(RuntimeError, match="create failed"):
            provider.start_container("echo-env:latest", env_vars={"TOKEN": "secret"})

        assert provider._redact_values == set()
        assert adapter.terminated == 0  # created nothing -> deleted nothing


# ---------------------------------------------------------------------------
# Tests: wait_for_ready (health, secrets, URL non-leak)
# ---------------------------------------------------------------------------
class TestWaitForReady:
    def test_ready_when_health_200(self, provider):
        provider.start_container("echo-env:latest")
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            provider.wait_for_ready("https://x.modal.host", timeout_s=5)

    def test_dead_process_raises_without_log_by_default(self, provider, adapter):
        import requests

        adapter.dead_process = True
        adapter.log = "TOKEN=topsecret traceback"
        provider.start_container("echo-env:latest", env_vars={"TOKEN": "topsecret"})
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError, match="not surfaced") as exc_info:
                provider.wait_for_ready("https://x.modal.host", timeout_s=5)
        assert "topsecret" not in str(exc_info.value)
        assert "traceback" not in str(exc_info.value)

    def test_dead_process_surfaces_redacted_log_when_opted_in(self, adapter):
        import requests

        adapter.dead_process = True
        adapter.log = "Traceback: TOKEN=supersecret123 failed"
        provider = ModalProvider(_adapter=adapter, surface_server_logs=True)
        provider.start_container(
            "echo-env:latest", env_vars={"TOKEN": "supersecret123"}
        )

        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(RuntimeError) as exc_info:
                provider.wait_for_ready("https://x.modal.host", timeout_s=5)

        msg = str(exc_info.value)
        assert "supersecret123" not in msg
        assert "***" in msg

    def test_timeout_does_not_leak_url(self, provider):
        import requests

        provider.start_container("echo-env:latest")
        # Keep the process "RUNNING" so the death check doesn't short-circuit.
        provider._sandbox = None  # skip the in-loop death check entirely
        url = "https://secret-bearer.modal.host/tok123"
        with (
            patch("requests.get", side_effect=requests.ConnectionError("refused")),
            patch(
                "openenv.core.containers.runtime.modal_provider.time.time"
            ) as mock_time,
        ):
            mock_time.side_effect = [0, 1, 100]
            with pytest.raises(TimeoutError) as exc_info:
                provider.wait_for_ready(url, timeout_s=5)
        assert url not in str(exc_info.value)
        assert "secret-bearer.modal.host" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: _DefaultModalAdapter SDK shape + Sandbox v2 feature check
# ---------------------------------------------------------------------------
def _install_fake_modal(monkeypatch, *, has_experimental: bool = True):
    """Install a minimal fake ``modal`` module pinned to the SDK shape used by
    ``_DefaultModalAdapter`` and return it for introspection."""
    modal_mod = types.ModuleType("modal")
    calls: dict = {"create": None, "create_v2": None}

    class _FakeProc:
        def __init__(self, output=""):
            self.stdout = MagicMock()
            self.stdout.read = MagicMock(return_value=output)

        def wait(self):
            return 0

    class _FakeTunnel:
        url = "https://sb-xyz-8000.modal.host"

    class _FakeSandboxInstance:
        def __init__(self):
            self.exec_calls = []
            self.terminated = False

        def exec(self, *args, **kwargs):
            self.exec_calls.append(args)
            return _FakeProc("ok")

        def tunnels(self, *args, **kwargs):
            return {8000: _FakeTunnel()}

        def terminate(self, *args, **kwargs):
            self.terminated = True

    class _FakeSandbox:
        @staticmethod
        def create(*args, **kwargs):
            calls["create"] = (args, kwargs)
            return _FakeSandboxInstance()

    if has_experimental:

        def _experimental_create(*args, **kwargs):
            calls["create_v2"] = (args, kwargs)
            return _FakeSandboxInstance()

        _FakeSandbox._experimental_create = staticmethod(_experimental_create)

    class _FakeApp:
        @staticmethod
        def lookup(name, *, create_if_missing=False, **kwargs):
            calls["app"] = name
            return object()

    class _FakeImage:
        @staticmethod
        def from_dockerfile(path, **kwargs):
            return ("dockerfile", str(path), kwargs)

        @staticmethod
        def from_registry(tag, *args, **kwargs):
            return ("registry", tag)

    modal_mod.Sandbox = _FakeSandbox
    modal_mod.App = _FakeApp
    modal_mod.Image = _FakeImage
    modal_mod._calls = calls
    monkeypatch.setitem(sys.modules, "modal", modal_mod)
    return modal_mod


class TestDefaultAdapter:
    def test_matches_sdk_shape_v1(self, monkeypatch):
        modal_mod = _install_fake_modal(monkeypatch)
        adapter = _DefaultModalAdapter(app_name="openenv", use_sandbox_v2=False)

        image = adapter.image_from_registry("echo-env:latest")
        assert image == ("registry", "echo-env:latest")

        sandbox = adapter.create_sandbox(
            image=image,
            encrypted_ports=[8000],
            timeout=300,
            env={"A": "B"},
            extra={"cpu": 2.0},
        )
        args, kwargs = modal_mod._calls["create"]
        assert args[:2] == ("sleep", "infinity")
        assert kwargs["encrypted_ports"] == [8000]
        assert kwargs["timeout"] == 300
        assert kwargs["env"] == {"A": "B"}
        assert kwargs["cpu"] == 2.0

        assert adapter.exec(sandbox, "echo hi") == "ok"
        assert adapter.tunnel_url(sandbox, 8000) == "https://sb-xyz-8000.modal.host"
        adapter.terminate(sandbox)
        assert sandbox.terminated is True

    def test_v2_uses_experimental_create(self, monkeypatch):
        modal_mod = _install_fake_modal(monkeypatch)
        adapter = _DefaultModalAdapter(app_name="openenv", use_sandbox_v2=True)
        adapter.create_sandbox(
            image=("registry", "x"),
            encrypted_ports=[8000],
            timeout=300,
            env=None,
            extra={},
        )
        assert modal_mod._calls["create_v2"] is not None
        assert modal_mod._calls["create"] is None

    def test_v2_feature_check_raises_when_missing(self, monkeypatch):
        _install_fake_modal(monkeypatch, has_experimental=False)
        with pytest.raises(RuntimeError, match="_experimental_create"):
            _DefaultModalAdapter(app_name="openenv", use_sandbox_v2=True)

    def test_v1_works_without_experimental_create(self, monkeypatch):
        _install_fake_modal(monkeypatch, has_experimental=False)
        # No feature check for the stable API.
        _DefaultModalAdapter(app_name="openenv", use_sandbox_v2=False)


class TestSandboxV2Flag:
    def test_default_is_off(self, adapter):
        assert ModalProvider(_adapter=adapter)._use_sandbox_v2 is False

    def test_v2_emits_log(self, adapter, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            ModalProvider(_adapter=adapter, use_sandbox_v2=True)
        assert any("Sandbox v2" in r.message for r in caplog.records)

    def test_off_emits_no_log(self, adapter, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            ModalProvider(_adapter=adapter)
        assert not any("Sandbox v2" in r.message for r in caplog.records)
