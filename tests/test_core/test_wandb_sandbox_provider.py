"""Unit tests for WandbSandboxProvider."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import pytest  # type: ignore[import-not-found]
from openenv.core.containers.runtime.cwsandbox_provider import (  # type: ignore[import-not-found]
    CWSandboxProvider,
)
from openenv.core.containers.runtime.wandb_sandbox_provider import (  # type: ignore[import-not-found]
    WandbSandboxProvider,
)


class _AuthError(Exception):
    pass


class _FakeRef:
    def __init__(self, value=None):
        self.value = value

    def result(self):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class _FakeProcess:
    def __init__(self, stdout: str = ""):
        self._result = SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    def result(self):
        return self._result


class _FakeNetworkOptions:
    def __init__(self, **kwargs):
        self.ingress_mode = kwargs.get("ingress_mode")
        self.exposed_ports = kwargs.get("exposed_ports")
        self.egress_mode = kwargs.get("egress_mode")


class _FakeSandbox:
    def __init__(self):
        self.sandbox_id = "sandbox-123"
        self.runner_id = "runner-123"
        self.started_at = "2026-06-22T00:00:00Z"
        self.service_address = "https://wandb-sandbox.example.test:8000"
        self.exec_calls: list[list[str]] = []
        self.wait_calls: list[float | None] = []
        self.stopped = False

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return self

    def exec(self, command, timeout_seconds=None):
        self.exec_calls.append(command)
        script = command[-1]
        if "test -f /app/env/openenv.yaml" in script:
            return _FakeProcess("found\n")
        if script.startswith("cat /app/env/openenv.yaml"):
            return _FakeProcess("spec_version: 1\napp: echo_env.server.app:app\n")
        if "kill -0" in script:
            return _FakeProcess("RUNNING\n")
        if script.startswith("cat /tmp/openenv-server.log"):
            return _FakeProcess("")
        return _FakeProcess("")

    def stop(self, missing_ok=False):
        self.stopped = True
        self.stop_missing_ok = missing_ok
        return _FakeRef()


class _FakeSandboxClass:
    def __init__(self):
        self.last_run_kwargs = None
        self.deleted: list[dict] = []
        self.instance = _FakeSandbox()
        self.list_result = _FakeRef([])
        self.run_count = 0

    def run(self, **kwargs):
        self.run_count += 1
        self.last_run_kwargs = kwargs
        return self.instance

    def delete(self, sandbox_id, **kwargs):
        self.deleted.append({"sandbox_id": sandbox_id, **kwargs})
        return _FakeRef()

    def list(self, **kwargs):
        self.last_list_kwargs = kwargs
        return self.list_result


@pytest.fixture()
def fake_wandb_sdk():
    sandbox_cls = _FakeSandboxClass()
    metadata_calls = []

    def set_integration_metadata(name):
        metadata_calls.append(name)

    return SimpleNamespace(
        CWSandboxAuthenticationError=_AuthError,
        NetworkOptions=_FakeNetworkOptions,
        Sandbox=sandbox_cls,
        _sandbox_cls=sandbox_cls,
        _metadata_calls=metadata_calls,
        set_integration_metadata=set_integration_metadata,
    )


def test_start_container_uses_wandb_sandbox_sdk(fake_wandb_sdk):
    provider = WandbSandboxProvider(
        sdk=fake_wandb_sdk,
        ingress_mode="public",
        egress_mode="internet",
        request_timeout_seconds=123,
        max_timeout_seconds=300,
        tags=["openenv-wandb-smoke"],
    )

    url = provider.start_container(
        "registry.hf.space/openenv-echo-env:latest",
        env_vars={"WB_CHECK_VAR": "from-openenv"},
    )

    kwargs = fake_wandb_sdk._sandbox_cls.last_run_kwargs
    assert url == "https://wandb-sandbox.example.test:8000"
    assert fake_wandb_sdk._metadata_calls == ["openenv"]
    assert kwargs["container_image"] == "registry.hf.space/openenv-echo-env:latest"
    assert kwargs["environment_variables"] == {"WB_CHECK_VAR": "from-openenv"}
    assert kwargs["max_timeout_seconds"] == 300
    assert kwargs["tags"] == ["openenv-wandb-smoke"]
    assert kwargs["network"].ingress_mode == "public"
    assert kwargs["network"].exposed_ports == (8000,)
    assert kwargs["network"].egress_mode == "internet"


def test_wandb_provider_inherits_cwsandbox_provider(fake_wandb_sdk):
    provider = WandbSandboxProvider(ingress_mode="public", sdk=fake_wandb_sdk)

    assert isinstance(provider, CWSandboxProvider)


def test_preflight_lists_sandboxes(fake_wandb_sdk):
    WandbSandboxProvider.preflight(sdk=fake_wandb_sdk)

    assert fake_wandb_sdk._metadata_calls == ["openenv"]
    assert fake_wandb_sdk._sandbox_cls.last_list_kwargs == {}


def test_preflight_rejects_invalid_wandb_auth(fake_wandb_sdk):
    fake_wandb_sdk._sandbox_cls.list_result = _FakeRef(_AuthError("bad token"))

    with pytest.raises(SystemExit, match="wandb login.*WANDB_API_KEY"):
        WandbSandboxProvider.preflight(sdk=fake_wandb_sdk)


def test_stop_container_deletes_wandb_sandbox(fake_wandb_sdk):
    provider = WandbSandboxProvider(ingress_mode="public", sdk=fake_wandb_sdk)
    provider.start_container("echo-env:latest")

    provider.stop_container()

    sandbox = fake_wandb_sdk._sandbox_cls.instance
    assert sandbox.stopped is True
    assert sandbox.stop_missing_ok is True
    assert fake_wandb_sdk._sandbox_cls.deleted == [
        {
            "sandbox_id": "sandbox-123",
            "base_url": None,
            "timeout_seconds": 300.0,
            "missing_ok": True,
        }
    ]


def test_wandb_provider_inherits_active_sandbox_lifecycle_guard(fake_wandb_sdk):
    provider = WandbSandboxProvider(ingress_mode="public", sdk=fake_wandb_sdk)
    provider.start_container("echo-env:latest")

    with pytest.raises(RuntimeError, match="already has an active sandbox"):
        provider.start_container("other-env:latest")

    provider.close()

    assert fake_wandb_sdk._sandbox_cls.run_count == 1
    assert fake_wandb_sdk._sandbox_cls.instance.stopped is True


def test_lifecycle_kwargs_pass_through_to_cwsandbox_provider(fake_wandb_sdk):
    provider = WandbSandboxProvider(
        sdk=fake_wandb_sdk,
        base_url="https://api.example.test",
        ingress_mode="public",
        egress_mode="internet",
        resources={"requests": {"cpu": "2"}},
        request_timeout_seconds=45,
        max_lifetime_seconds=3600,
        max_timeout_seconds=300,
        tags=["openenv", "wandb"],
        profile_ids=["profile-id"],
        profile_names=["profile-name"],
        runner_ids=["runner-id"],
        cmd="python -m server",
        url_scheme="https",
        delete_on_stop=False,
        ready_settle_seconds=0,
    )

    provider.start_container("echo-env:latest")
    kwargs = fake_wandb_sdk._sandbox_cls.last_run_kwargs

    assert kwargs["base_url"] == "https://api.example.test"
    assert kwargs["resources"] == {"requests": {"cpu": "2"}}
    assert kwargs["request_timeout_seconds"] == 45
    assert kwargs["max_lifetime_seconds"] == 3600
    assert kwargs["max_timeout_seconds"] == 300
    assert kwargs["tags"] == ["openenv", "wandb"]
    assert kwargs["profile_ids"] == ["profile-id"]
    assert kwargs["profile_names"] == ["profile-name"]
    assert kwargs["runner_ids"] == ["runner-id"]
    assert provider.base_url == "https://wandb-sandbox.example.test:8000"


def test_missing_wandb_sandbox_dependency_raises_runtime_error():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb.sandbox":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="requires wandb.sandbox"):
            WandbSandboxProvider()
