"""Unit tests for ACASandboxProvider. All tests use a fake ACA adapter."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.aca_provider import (
    _DefaultACASandboxAdapter,
    ACASandboxProvider,
)
from openenv.core.containers.runtime.providers import ContainerProvider


def test_container_provider_base_has_noop_close_and_context_manager():
    """The base ContainerProvider now defines close()/__enter__/__exit__ so a
    caller can release resources polymorphically; the default close() is a no-op
    and __exit__ calls it."""

    class _Minimal(ContainerProvider):
        def start_container(self, image, port=None, env_vars=None, **kwargs):
            return "http://localhost:8000"

        def stop_container(self):
            pass

        def wait_for_ready(self, base_url, timeout_s=30.0):
            pass

    # Default close() is a no-op and must not raise.
    _Minimal().close()

    # __enter__ returns self.
    p = _Minimal()
    with p as ctx:
        assert ctx is p

    # __exit__ invokes close() (override path).
    class _Closes(_Minimal):
        closed = False

        def close(self):
            type(self).closed = True

    with _Closes():
        pass
    assert _Closes.closed is True


class _ExecResult:
    def __init__(self, stdout: str = ""):
        self.stdout = stdout


class _FakeSandbox:
    pass


class _FakeAdapter:
    def __init__(self):
        self.sandbox = _FakeSandbox()
        self.created = []
        self.ports = []
        self.exec_commands = []
        self.deleted = []
        self.closed = False
        self.close_count = 0
        self.port_url = "https://aca-sandbox-port.example"
        self.fail_add_port = False
        self.fail_delete = False
        self.dead_process = False
        self.log = "server crashed"

    def create_sandbox(self, **kwargs):
        self.created.append(kwargs)
        return self.sandbox

    def add_port(self, sandbox, *, port, anonymous):
        self.ports.append({"sandbox": sandbox, "port": port, "anonymous": anonymous})
        if self.fail_add_port:
            raise RuntimeError("port failed")
        return self.port_url

    def exec(self, sandbox, command, *, working_directory=None):
        self.exec_commands.append(
            {
                "sandbox": sandbox,
                "command": command,
                "working_directory": working_directory,
            }
        )
        if "kill -0" in command:
            return _ExecResult("DEAD" if self.dead_process else "RUNNING")
        if "cat /tmp/openenv-server.log" in command:
            return _ExecResult(self.log)
        return _ExecResult("")

    def delete_sandbox(self, sandbox):
        self.deleted.append(sandbox)
        if self.fail_delete:
            raise RuntimeError("delete failed")

    def close(self):
        self.closed = True
        self.close_count += 1


@pytest.fixture()
def adapter():
    return _FakeAdapter()


# A sentinel non-None egress policy so tests that are not about egress do not
# trigger the "unrestricted egress" security warning. The fake adapter just
# records whatever object is passed.
_SAFE_EGRESS = object()


def _provider(adapter, **kwargs):
    kwargs.setdefault("egress_policy", _SAFE_EGRESS)
    return ACASandboxProvider(_adapter=adapter, anonymous_port=True, **kwargs)


def test_start_container_creates_disk_sandbox_and_exposes_port(adapter):
    provider = ACASandboxProvider(
        _adapter=adapter,
        anonymous_port=True,
        egress_policy=_SAFE_EGRESS,
        cmd="python -m uvicorn server.app:app --host 0.0.0.0 --port 8000",
        labels={"provider": "aca"},
    )

    url = provider.start_container(
        "disk:openenv-test",
        env_vars={"DEBUG": "1"},
        labels={"task": "smoke"},
    )

    assert url == "https://aca-sandbox-port.example"
    created = adapter.created[0]
    assert created["disk"] == "openenv-test"
    assert created["disk_id"] is None
    assert created["env_vars"] == {"DEBUG": "1"}
    assert created["labels"] == {"provider": "aca", "task": "smoke"}
    assert adapter.ports == [
        {
            "sandbox": adapter.sandbox,
            "port": 8000,
            "anonymous": True,
        }
    ]
    assert any(
        "uvicorn server.app:app" in call["command"] for call in adapter.exec_commands
    )


def test_constructor_image_used_when_start_omits_image(adapter):
    provider = _provider(
        adapter,
        image="disk:constructor-env",
        env_vars={"DEBUG": "1"},
    )

    provider.start_container()

    created = adapter.created[0]
    assert created["disk"] == "constructor-env"
    assert created["env_vars"] == {"DEBUG": "1"}


def test_start_image_overrides_constructor_image(adapter):
    provider = _provider(adapter, image="disk:constructor-env")

    provider.start_container("disk:explicit-env")

    assert adapter.created[0]["disk"] == "explicit-env"


def test_requires_image_from_constructor_or_start(adapter):
    provider = _provider(adapter)

    with pytest.raises(ValueError, match="requires an image"):
        provider.start_container()


def test_provider_imports_from_its_module():
    # Optional cloud providers are imported from their module, not re-exported
    # from the runtime package (matches DaytonaProvider).
    from openenv.core.containers.runtime.aca_provider import (
        ACASandboxProvider as ModuleACASandboxProvider,
    )

    assert ModuleACASandboxProvider is ACASandboxProvider


def test_provider_not_in_runtime_package_all():
    import openenv.core.containers.runtime as runtime_pkg

    # Optional cloud providers are not surfaced at the package root (matches
    # DaytonaProvider): neither in __all__ nor as an attribute.
    assert "ACASandboxProvider" not in runtime_pkg.__all__
    assert not hasattr(runtime_pkg, "ACASandboxProvider")


def test_empty_sources_raise_clear_errors(adapter):
    provider = _provider(adapter)

    with pytest.raises(ValueError, match="non-empty ACA disk"):
        provider.start_container("")
    with pytest.raises(ValueError, match="non-empty ACA disk"):
        provider.start_container("   ")  # whitespace-only
    with pytest.raises(ValueError, match="disk: source"):
        provider.start_container("disk:")
    with pytest.raises(ValueError, match="disk: source"):
        provider.start_container("disk: ")  # whitespace-only name
    with pytest.raises(ValueError, match="disk-id: source"):
        provider.start_container("disk-id:")
    with pytest.raises(ValueError, match="disk-id: source"):
        provider.start_container("disk-id:  ")  # whitespace-only id


@pytest.mark.parametrize("disk_name", ["ubuntu", "python-3.11"])
def test_plain_image_is_treated_as_public_disk(disk_name):
    # "ubuntu" and "python-3.11" (the disk name the live ACA demo uses) are bare
    # public disk names — hyphens/dots are fine; only "/" or ":" are rejected.
    # A fresh provider per disk (no start_container re-use).
    adapter = _FakeAdapter()
    provider = _provider(adapter)

    provider.start_container(disk_name)

    assert adapter.created[0]["disk"] == disk_name


def test_docker_image_reference_is_rejected(adapter):
    """A migration foot-gun: an OCI/Docker image string must fail with guidance,
    not be silently passed to the ACA SDK as a disk name."""
    provider = _provider(adapter)

    for bad in (
        "myregistry.azurecr.io/echo-env:latest",  # registry path + tag
        "echo-env:latest",  # bare image:tag
        "library/ubuntu",  # repo path
    ):
        with pytest.raises(ValueError, match="container/OCI image reference"):
            provider.start_container(bad)

    assert adapter.created == []  # never reached the SDK


def test_disk_id_source(adapter):
    provider = _provider(adapter)

    provider.start_container(
        "disk-id:/subscriptions/sub/resourceGroups/rg/diskimages/img"
    )

    assert (
        adapter.created[0]["disk_id"]
        == "/subscriptions/sub/resourceGroups/rg/diskimages/img"
    )
    assert adapter.created[0]["disk"] is None


def test_port_validation(adapter):
    provider = _provider(adapter)

    with pytest.raises(ValueError, match="only supports port 8000"):
        provider.start_container("ubuntu", port=3000)


def test_anonymous_port_must_be_explicit(adapter):
    with pytest.raises(ValueError, match="requires explicit anonymous_port=True"):
        ACASandboxProvider(_adapter=adapter)


def test_authenticated_aca_ports_are_not_supported(adapter):
    with pytest.raises(ValueError, match="authenticated ACA sandbox ports"):
        ACASandboxProvider(_adapter=adapter, anonymous_port=False)


def test_start_container_cleans_up_on_port_failure(adapter):
    adapter.fail_add_port = True
    provider = _provider(adapter)

    with pytest.raises(RuntimeError, match="port failed"):
        provider.start_container("ubuntu")

    assert adapter.deleted == [adapter.sandbox]
    with pytest.raises(RuntimeError, match="no active base_url"):
        _ = provider.base_url


def test_cleanup_failure_does_not_mask_original_error(adapter):
    """If cleanup (delete_sandbox) also fails, the ORIGINAL start error must
    still propagate, not be replaced by the secondary cleanup error."""
    adapter.fail_add_port = True  # original failure
    adapter.fail_delete = True  # secondary failure during stop_container()
    provider = _provider(adapter)

    with pytest.raises(RuntimeError, match="port failed"):
        provider.start_container("ubuntu")

    assert adapter.deleted == [adapter.sandbox]  # cleanup was still attempted


def test_unsupported_start_kwargs_raise(adapter):
    provider = _provider(adapter)

    with pytest.raises(ValueError, match="Unsupported ACASandboxProvider"):
        provider.start_container("ubuntu", unsupported=True)


def test_wait_for_ready_polls_health(adapter):
    provider = _provider(adapter)
    url = provider.start_container("ubuntu")
    response = MagicMock(status_code=200)

    with patch("requests.get", return_value=response) as get:
        provider.wait_for_ready(url)

    get.assert_called_once_with(f"{url}/health", timeout=5.0)


def test_wait_for_ready_reports_dead_server_log(adapter):
    adapter.dead_process = True
    provider = _provider(adapter, cmd="python -m broken", surface_server_logs=True)
    url = provider.start_container("ubuntu")

    import requests

    with patch("requests.get", side_effect=requests.ConnectionError("nope")):
        with pytest.raises(RuntimeError, match="server process died") as exc_info:
            provider.wait_for_ready(url)

    assert adapter.log in str(exc_info.value)


def test_crash_log_not_surfaced_by_default(adapter):
    """S4: server output is not surfaced in the crash error by default."""
    adapter.dead_process = True
    adapter.log = "TOKEN=topsecret crashed"
    provider = _provider(adapter, cmd="python -m broken")  # surface_server_logs=False
    url = provider.start_container("ubuntu")

    import requests

    with patch("requests.get", side_effect=requests.ConnectionError("nope")):
        with pytest.raises(RuntimeError, match="not surfaced") as exc_info:
            provider.wait_for_ready(url)

    assert "topsecret" not in str(exc_info.value)
    assert "crashed" not in str(exc_info.value)


def test_non_https_error_does_not_leak_url(adapter):
    """S2: the bearer port URL is never interpolated into the error."""
    adapter.port_url = "http://secret-bearer-host.example/abc123"
    provider = _provider(adapter)

    with pytest.raises(RuntimeError, match="non-HTTPS") as exc_info:
        provider.start_container("ubuntu")

    assert "secret-bearer-host.example" not in str(exc_info.value)
    assert "abc123" not in str(exc_info.value)


def test_timeout_error_does_not_leak_url(adapter):
    """S2: the timeout error does not interpolate the bearer URL."""
    provider = _provider(adapter)  # no cmd -> no dead-process check
    url = provider.start_container("ubuntu")

    import requests

    with patch("requests.get", side_effect=requests.ConnectionError("nope")):
        with patch("time.sleep"):
            with pytest.raises(TimeoutError) as exc_info:
                provider.wait_for_ready(url, timeout_s=0.01)

    assert url not in str(exc_info.value)
    assert "aca-sandbox-port.example" not in str(exc_info.value)


def test_start_time_anonymous_port_false_is_rejected(adapter):
    """S2: a start-time anonymous_port=False override is rejected."""
    provider = _provider(adapter)

    with pytest.raises(ValueError, match="authenticated ACA sandbox ports"):
        provider.start_container("ubuntu", anonymous_port=False)


def test_stop_container_clears_redact_values(adapter):
    """Stale injected secret values must not survive across sandboxes."""
    provider = _provider(adapter)
    provider.start_container("ubuntu", env_vars={"TOKEN": "secret-123"})
    assert provider._redact_values == {"secret-123"}

    provider.stop_container()

    assert provider._redact_values == set()


def test_failed_create_auto_clears_redact_values(adapter):
    """create_sandbox failure must AUTO-clear injected secret values: the create
    stage drops them on failure, so no caller action is required."""

    def boom(**kwargs):
        raise RuntimeError("create failed")

    adapter.create_sandbox = boom
    provider = _provider(adapter)

    with pytest.raises(RuntimeError, match="create failed"):
        provider.start_container("ubuntu", env_vars={"TOKEN": "secret-123"})

    assert provider._redact_values == set()
    # A create failure created nothing, so nothing was deleted.
    assert adapter.deleted == []


def test_double_start_raises_and_preserves_existing_sandbox(adapter):
    """Only one sandbox per provider: a second start_container before stop must
    raise and leave the existing sandbox intact (no orphaned/leaked sandbox)."""
    provider = _provider(adapter)
    provider.start_container("ubuntu")
    first_sandbox = provider._sandbox
    first_url = provider._base_url

    with pytest.raises(RuntimeError, match="already has an active sandbox"):
        provider.start_container("python-3.11")

    assert adapter.deleted == []  # the first sandbox is untouched, not orphaned
    assert len(adapter.created) == 1  # the guard fired before a second create
    assert provider._sandbox is first_sandbox
    assert provider._base_url == first_url


def test_start_after_stop_is_allowed(adapter):
    """After stop_container the provider can start a fresh sandbox again."""
    provider = _provider(adapter)
    provider.start_container("ubuntu")
    provider.stop_container()

    # Does not raise; a fresh sandbox is created.
    url = provider.start_container("python-3.11")
    assert url == adapter.port_url
    assert len(adapter.created) == 2  # a real second create occurred after stop


def test_close_deletes_sandbox(adapter):
    provider = _provider(adapter)
    provider.start_container("ubuntu")

    provider.close()

    assert adapter.deleted == [adapter.sandbox]
    assert adapter.closed is False


def test_adapter_owned_close_closes_adapter():
    adapter = _FakeAdapter()
    provider = _provider(adapter)
    provider._owns_adapter = True

    provider.close()

    assert adapter.closed is True


def test_double_close_closes_adapter_once():
    """A second close()/context exit must not close the owned SDK client again
    (a preview SDK's close is not guaranteed idempotent)."""
    adapter = _FakeAdapter()
    provider = _provider(adapter)
    provider._owns_adapter = True

    provider.close()
    provider.close()

    assert adapter.close_count == 1


def test_context_manager_closes_provider():
    """Using the provider as a context manager stops the sandbox and releases
    the owned SDK client on exit (deterministic cleanup)."""
    adapter = _FakeAdapter()
    with ACASandboxProvider(
        _adapter=adapter, anonymous_port=True, egress_policy=_SAFE_EGRESS
    ) as provider:
        provider._owns_adapter = True
        provider.start_container("ubuntu")

    assert adapter.deleted == [adapter.sandbox]
    assert adapter.closed is True


def test_context_manager_closes_on_exception():
    """The point of the context manager: cleanup still happens if the `with`
    body raises."""
    adapter = _FakeAdapter()
    with pytest.raises(RuntimeError, match="boom"):
        with ACASandboxProvider(
            _adapter=adapter, anonymous_port=True, egress_policy=_SAFE_EGRESS
        ) as provider:
            provider._owns_adapter = True
            provider.start_container("ubuntu")
            raise RuntimeError("boom")

    assert adapter.deleted == [adapter.sandbox]
    assert adapter.closed is True


def test_default_adapter_matches_preview_sdk_shape(monkeypatch):
    calls = {}

    class _FakePoller:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _FakeSandboxPort:
        url = "https://aca-port.example"

    class _FakeSandboxClient:
        """Sandbox-scoped operations client.

        Mirrors the real SDK contract: ``begin_create_sandbox(...).result()``
        blocks until the sandbox is *Running* and returns this client (not a
        plain data model), so it carries ``sandbox_id`` plus the
        exec/add_port/begin_delete operations the adapter calls.
        """

        sandbox_id = "sandbox-xyz"

        def __init__(self):
            self.add_port_calls = []
            self.exec_calls = []
            self.deleted = False

        def add_port(self, port, *, anonymous):
            self.add_port_calls.append({"port": port, "anonymous": anonymous})
            return _FakeSandboxPort()

        def exec(self, command, *, working_directory=None):
            self.exec_calls.append(
                {"command": command, "working_directory": working_directory}
            )
            return _ExecResult("ok")

        def begin_delete(self):
            self.deleted = True
            return _FakePoller(None)

    fake_sandbox = _FakeSandboxClient()

    class _FakeEgressPolicy:
        """Mirrors the real SDK EgressPolicy, which exposes ``_to_dict()``."""

        def __init__(self, default):
            self._default = default

        def _to_dict(self):
            return {"default": self._default}

    egress_policy = _FakeEgressPolicy("deny")

    class _FakeSandboxGroupClient:
        def __init__(
            self,
            endpoint,
            credential,
            *,
            subscription_id,
            resource_group,
            sandbox_group,
            **kwargs,
        ):
            calls["client_init"] = {
                "endpoint": endpoint,
                "credential": credential,
                "subscription_id": subscription_id,
                "resource_group": resource_group,
                "sandbox_group": sandbox_group,
                "kwargs": kwargs,
            }

        def begin_create_sandbox(self, **kwargs):
            # The real SDK resolves the egress policy via _to_dict(); a plain
            # dict would raise AttributeError here, so this guards the contract.
            policy = kwargs.get("egress_policy")
            if policy is not None:
                calls["egress_resolved"] = policy._to_dict()
            calls["create_sandbox"] = kwargs
            # `.result()` blocks until Running and returns the sandbox ops
            # client (not a data model) — mirror that here.
            return _FakePoller(fake_sandbox)

        def close(self):
            calls["closed"] = True

    sandbox_module = types.ModuleType("azure.containerapps.sandbox")
    sandbox_module.SandboxGroupClient = _FakeSandboxGroupClient
    sandbox_module.endpoint_for_region = lambda region: f"https://{region}.example"

    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(
        sys.modules, "azure.containerapps", types.ModuleType("azure.containerapps")
    )
    monkeypatch.setitem(sys.modules, "azure.containerapps.sandbox", sandbox_module)

    credential = object()
    adapter = _DefaultACASandboxAdapter(
        subscription_id="sub",
        resource_group="rg",
        sandbox_group="sg",
        region="eastus",
        endpoint=None,
        credential=credential,
        sdk_kwargs={"api_version": "preview"},
    )

    sandbox = adapter.create_sandbox(
        disk="ubuntu",
        disk_id=None,
        env_vars={"A": "B"},
        labels={"openenv": "test"},
        egress_policy=egress_policy,
        cpu="1000m",
        memory="2048Mi",
        disk_size="4Gi",
        auto_suspend_seconds=300,
        auto_suspend_mode="Memory",
        create_timeout=90,
        polling_interval=2,
    )

    assert sandbox is fake_sandbox
    assert calls["client_init"] == {
        "endpoint": "https://eastus.example",
        "credential": credential,
        "subscription_id": "sub",
        "resource_group": "rg",
        "sandbox_group": "sg",
        "kwargs": {"api_version": "preview"},
    }
    assert calls["create_sandbox"] == {
        "disk": "ubuntu",
        "cpu": "1000m",
        "memory": "2048Mi",
        "disk_size": "4Gi",
        "auto_suspend_seconds": 300,
        "auto_suspend_mode": "Memory",
        "labels": {"openenv": "test"},
        "environment": {"A": "B"},
        "egress_policy": egress_policy,
        "polling_timeout": 90,
        "polling_interval": 2,
    }
    assert calls["egress_resolved"] == {"default": "deny"}

    assert (
        adapter.add_port(sandbox, port=8000, anonymous=True)
        == "https://aca-port.example"
    )
    assert fake_sandbox.add_port_calls == [{"port": 8000, "anonymous": True}]

    assert adapter.exec(sandbox, "echo hi", working_directory="/app").stdout == "ok"
    assert fake_sandbox.exec_calls == [
        {"command": "echo hi", "working_directory": "/app"}
    ]

    adapter.delete_sandbox(sandbox)
    adapter.close()
    assert fake_sandbox.deleted is True
    assert calls["closed"] is True


# ---------------------------------------------------------------------------
# Security tests (RFC 002 "Cloud Sandbox Providers" security invariants)
# ---------------------------------------------------------------------------


def test_rejects_non_https_url(adapter):
    """S1: a plaintext port URL must be refused (EnvClient would use ws://)."""
    adapter.port_url = "http://insecure.example"
    provider = _provider(adapter)

    with pytest.raises(RuntimeError, match="non-HTTPS"):
        provider.start_container("ubuntu")

    # Failed start must clean up the sandbox it created.
    assert adapter.deleted == [adapter.sandbox]


def test_warns_on_unrestricted_egress(adapter):
    """S3: starting with no egress policy warns (ACA default is Allow)."""
    provider = ACASandboxProvider(_adapter=adapter, anonymous_port=True)

    with pytest.warns(UserWarning, match="unrestricted egress"):
        provider.start_container("ubuntu")


def test_safe_egress_policy_does_not_warn(adapter, recwarn):
    """A non-None egress policy means no unrestricted-egress warning."""
    provider = _provider(adapter)  # _SAFE_EGRESS sentinel
    provider.start_container("ubuntu")

    assert not [w for w in recwarn.list if "unrestricted egress" in str(w.message)]


def test_deny_all_egress_builds_default_deny_policy(monkeypatch):
    """deny_all_egress() lazily builds a default-deny EgressPolicy with allowlist."""

    class _FakeEgressHostRule:
        def __init__(self, *, pattern, action):
            self.pattern = pattern
            self.action = action

    class _FakeEgressPolicy:
        def __init__(self, *, default_action, host_rules):
            self.default_action = default_action
            self.host_rules = host_rules

    sandbox_module = types.ModuleType("azure.containerapps.sandbox")
    sandbox_module.EgressPolicy = _FakeEgressPolicy
    sandbox_module.EgressHostRule = _FakeEgressHostRule
    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(
        sys.modules, "azure.containerapps", types.ModuleType("azure.containerapps")
    )
    monkeypatch.setitem(sys.modules, "azure.containerapps.sandbox", sandbox_module)

    policy = ACASandboxProvider.deny_all_egress(allow=["my-model.openai.azure.com"])

    assert policy.default_action == "Deny"
    assert [(r.pattern, r.action) for r in policy.host_rules] == [
        ("my-model.openai.azure.com", "Allow")
    ]


def test_wait_for_ready_redacts_injected_secrets(adapter):
    """S4: injected secret values are scrubbed from surfaced server output."""
    adapter.dead_process = True
    adapter.log = "Traceback: TOKEN=supersecret123 failed to load config"
    provider = _provider(adapter, cmd="python -m server", surface_server_logs=True)
    provider.start_container("ubuntu", env_vars={"TOKEN": "supersecret123"})

    import requests as _requests

    with patch("requests.get", side_effect=_requests.ConnectionError("nope")):
        with pytest.raises(RuntimeError) as exc_info:
            provider.wait_for_ready(provider.base_url)

    msg = str(exc_info.value)
    assert "supersecret123" not in msg
    assert "***" in msg
