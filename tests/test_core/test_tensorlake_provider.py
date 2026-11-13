# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TensorlakeProvider. All tests use a fake tensorlake SDK."""

from __future__ import annotations

import sys
import traceback
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.tensorlake_provider import TensorlakeProvider

OPENENV_YAML = b"spec_version: 1\nname: echo\napp: server.app:app\nport: 8000\n"
MANAGEMENT_URL = "https://sbx-1.sandbox.tensorlake.ai"
PUBLIC_URL = "https://8000-sbx-1.sandbox.tensorlake.ai"
SECRET = "s3cret-value"


def _info(status="running"):
    info = MagicMock()
    info.sandbox_id = "sbx-1"
    info.status = status
    return info


def _make_sandbox(yaml_content=OPENENV_YAML):
    """Fake handle returned by SandboxClient.connect()."""
    sandbox = MagicMock()
    if yaml_content is None:
        sandbox.read_file.side_effect = RuntimeError("not found")
    else:
        sandbox.read_file.return_value.value = yaml_content
    sandbox.start_process.return_value.pid = 42
    sandbox.update.return_value.ingress_endpoint = "https://sandbox.tensorlake.ai"
    sandbox.update.return_value.sandbox_id = "sbx-1"
    sandbox.update.return_value.sandbox_url = MANAGEMENT_URL
    sandbox.update.return_value.routing_hint = "hint"
    return sandbox


@pytest.fixture
def sdk():
    """Install a fake ``tensorlake.sandbox`` module and yield it.

    ``sdk.client`` is the fake SandboxClient instance, ``sdk.sandbox`` the
    handle it connects to, and ``sdk.probe`` the handle built for readiness
    probes.
    """
    tensorlake_mod = types.ModuleType("tensorlake")
    sandbox_mod = types.ModuleType("tensorlake.sandbox")
    sandbox_mod.SandboxError = type("SandboxError", (Exception,), {})
    sandbox_mod.SandboxNotFoundError = type(
        "SandboxNotFoundError", (sandbox_mod.SandboxError,), {}
    )
    sandbox_mod.SandboxClient = MagicMock()
    sandbox_mod.client = sandbox_mod.SandboxClient.return_value
    sandbox_mod.client.create.return_value = _info()
    sandbox_mod.client.get.return_value = _info()
    sandbox_mod.sandbox = _make_sandbox()
    sandbox_mod.client.connect.return_value = sandbox_mod.sandbox
    sandbox_mod.Sandbox = MagicMock()
    sandbox_mod.probe = sandbox_mod.Sandbox.return_value.__enter__.return_value
    sandbox_mod.probe.get_process.return_value.status = "running"
    sandbox_mod.sandbox_url_from_ingress_endpoint = lambda endpoint, sandbox_id, port: (
        f"https://{port}-{sandbox_id}.sandbox.tensorlake.ai"
    )
    tensorlake_mod.sandbox = sandbox_mod
    with patch.dict(
        sys.modules, {"tensorlake": tensorlake_mod, "tensorlake.sandbox": sandbox_mod}
    ):
        yield sandbox_mod


def _rendered(exc_info) -> str:
    """Render an exception like the interpreter does, chained causes included."""
    return "".join(traceback.format_exception(exc_info.value))


def test_start_container_launches_server_and_returns_public_url(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env", api_key="key", cpus=2)

    url = provider.start_container(env_vars={"A": "1"})

    assert url == PUBLIC_URL
    sdk.SandboxClient.assert_called_once_with(api_key="key")
    create_kwargs = sdk.client.create.call_args.kwargs
    assert create_kwargs["image"] == "echo-env"
    assert create_kwargs["cpus"] == 2
    assert create_kwargs["timeout_secs"] == 3600
    assert create_kwargs["deny_out"] == ["169.254.0.0/16"]
    assert create_kwargs["allow_out"] is None
    assert create_kwargs["allow_internet_access"] is True
    # The image CMD must not start a second server that races for port 8000.
    assert create_kwargs["entrypoint"] == ["sh", "-c", "exec sleep infinity"]
    sdk.client.connect.assert_called_once_with("sbx-1")
    cmd, args = sdk.sandbox.start_process.call_args.args
    assert cmd == "sh"
    assert args[1] == (
        "cd /app/env && python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"
    )
    assert sdk.sandbox.start_process.call_args.kwargs["env"] == {"A": "1"}
    sdk.sandbox.update.assert_called_once_with(
        allow_unauthenticated_access=True, exposed_ports=[8000]
    )


@pytest.mark.parametrize("public", [None, False])
def test_public_ingress_requires_explicit_opt_in(public):
    with pytest.raises(ValueError, match="public=True"):
        TensorlakeProvider(image="echo-env", public=public)


def test_metadata_egress_stays_blocked_with_custom_policy(sdk):
    provider = TensorlakeProvider(
        public=True,
        image="echo-env",
        allow_out=["huggingface.co"],
        deny_out=["10.0.0.0/8"],
    )

    provider.start_container()

    create_kwargs = sdk.client.create.call_args.kwargs
    assert create_kwargs["allow_out"] == ["huggingface.co"]
    assert create_kwargs["deny_out"] == ["169.254.0.0/16", "10.0.0.0/8"]
    assert create_kwargs["allow_internet_access"] is True


def test_empty_allowlist_blocks_all_egress(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env", allow_out=[])

    provider.start_container()

    create_kwargs = sdk.client.create.call_args.kwargs
    assert create_kwargs["allow_internet_access"] is False
    assert create_kwargs["allow_out"] == []


def test_pending_sandbox_is_polled_until_running(sdk):
    sdk.client.create.return_value = _info("pending")
    sdk.client.get.side_effect = [_info("pending"), _info("running")]
    provider = TensorlakeProvider(public=True, image="echo-env")

    with patch("time.sleep") as sleep:
        assert provider.start_container() == PUBLIC_URL

    assert sdk.client.get.call_count == 2
    assert sleep.call_count == 2
    sdk.client.connect.assert_called_once_with("sbx-1")


def test_pending_timeout_deletes_the_allocation(sdk):
    sdk.client.create.return_value = _info("pending")
    sdk.client.get.return_value = _info("pending")
    provider = TensorlakeProvider(public=True, image="echo-env")
    clock = iter(range(0, 10_000, 100))

    with patch("time.sleep"), patch("time.monotonic", side_effect=clock):
        with pytest.raises(TimeoutError, match="timed out"):
            provider.start_container()

    sdk.client.delete.assert_called_once_with("sbx-1")
    sdk.client.connect.assert_not_called()
    sdk.client.create.return_value = _info("running")
    assert provider.start_container() == PUBLIC_URL


def test_failed_sandbox_is_deleted(sdk):
    sdk.client.create.return_value = _info("failed")
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(RuntimeError, match="failed to start"):
        provider.start_container()

    sdk.client.delete.assert_called_once_with("sbx-1")


def test_allocation_is_deleted_when_connect_fails(sdk):
    # The SDK allocates first and connects second. A failure in between must
    # not leak the allocated sandbox (RFC 002 cloud-provider invariant 5).
    sdk.client.connect.side_effect = sdk.SandboxError(f"no route {SECRET}")
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(RuntimeError, match="withheld") as exc:
        provider.start_container()

    sdk.client.delete.assert_called_once_with("sbx-1")
    assert SECRET not in _rendered(exc)
    sdk.client.connect.side_effect = None
    assert provider.start_container() == PUBLIC_URL


def test_cleanup_failure_does_not_hide_start_error(sdk, caplog):
    sdk.sandbox.read_file.side_effect = sdk.SandboxError(f"read failed {SECRET}")
    sdk.client.delete.side_effect = sdk.SandboxError(f"delete failed {SECRET}")
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(ValueError, match="cmd=") as exc:
        provider.start_container()

    assert "Could not terminate" in caplog.text
    # Neither the sandbox ID nor the SDK messages may reach the logs.
    assert "sbx-1" not in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in _rendered(exc)


def test_sdk_errors_are_withheld_from_the_traceback(sdk):
    sdk.sandbox.start_process.side_effect = sdk.SandboxError(
        f"launch failed {PUBLIC_URL} {SECRET}"
    )
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(RuntimeError, match="withheld") as exc:
        provider.start_container(env_vars={"TOKEN": SECRET})

    rendered = _rendered(exc)
    assert SECRET not in rendered
    assert PUBLIC_URL not in rendered
    sdk.client.delete.assert_called_once_with("sbx-1")


def test_explicit_cmd_skips_discovery(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")

    provider.start_container(cmd="run-server")

    sdk.sandbox.read_file.assert_not_called()
    assert sdk.sandbox.start_process.call_args.args[1] == ["-c", "run-server"]


def test_missing_openenv_yaml_deletes_sandbox(sdk):
    sdk.client.connect.return_value = _make_sandbox(yaml_content=None)
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(ValueError, match="cmd="):
        provider.start_container()

    sdk.client.delete.assert_called_once_with("sbx-1")


def test_rejects_bad_arguments(sdk):
    provider = TensorlakeProvider(public=True)
    with pytest.raises(ValueError, match="requires an image"):
        provider.start_container()
    with pytest.raises(ValueError, match="port 8000"):
        provider.start_container("echo-env", port=9000)
    with pytest.raises(ValueError, match="Unsupported"):
        provider.start_container("echo-env", typo=True)
    sdk.SandboxClient.assert_not_called()


def test_accepts_autoenv_wait_timeout(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    assert provider.start_container(wait_timeout=30.0).startswith("https://")


def test_stop_container_deletes_once(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()

    provider.stop_container()
    provider.stop_container()

    sdk.client.delete.assert_called_once_with("sbx-1")
    sdk.sandbox.close.assert_called_once()


def test_failed_stop_is_retried_and_withholds_sdk_details(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()
    sdk.client.delete.side_effect = [sdk.SandboxError(f"network {SECRET}"), None]

    with pytest.raises(RuntimeError, match="termination failed") as exc:
        provider.stop_container()
    assert SECRET not in _rendered(exc)
    with pytest.raises(RuntimeError, match="already active"):
        provider.start_container()
    provider.stop_container()
    provider.stop_container()

    assert sdk.client.delete.call_count == 2
    assert provider.start_container() == PUBLIC_URL


def test_stop_withholds_handle_close_errors(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()
    sdk.sandbox.close.side_effect = sdk.SandboxError(f"websocket {SECRET}")

    provider.stop_container()

    sdk.client.delete.assert_called_once_with("sbx-1")
    assert provider.start_container() == PUBLIC_URL


def test_stop_on_expired_sandbox_succeeds(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()
    sdk.client.delete.side_effect = sdk.SandboxNotFoundError

    provider.stop_container()

    assert provider.start_container() == PUBLIC_URL


def test_constructor_env_vars_merge_with_call_env_vars(sdk):
    provider = TensorlakeProvider(
        public=True, image="echo-env", env_vars={"TOKEN": "t", "A": "1"}
    )

    provider.start_container(env_vars={"A": "2"})

    assert sdk.sandbox.start_process.call_args.kwargs["env"] == {
        "TOKEN": "t",
        "A": "2",
    }


def test_rejects_non_https_url(sdk):
    sdk.sandbox_url_from_ingress_endpoint = lambda endpoint, sandbox_id, port: (
        "http://insecure"
    )
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(RuntimeError, match="non-HTTPS") as exc:
        provider.start_container()

    assert "insecure" not in str(exc.value)
    sdk.client.delete.assert_called_once_with("sbx-1")


def test_missing_sandbox_url_deletes_sandbox(sdk):
    sdk.sandbox.update.return_value.sandbox_url = None
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(RuntimeError, match="endpoints"):
        provider.start_container()

    sdk.client.delete.assert_called_once_with("sbx-1")


def test_second_start_is_rejected(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()

    with pytest.raises(RuntimeError, match="already active"):
        provider.start_container()

    sdk.client.create.assert_called_once()


def test_context_manager_exit_deletes_sandbox(sdk):
    with TensorlakeProvider(public=True, image="echo-env") as provider:
        provider.start_container()

    sdk.client.delete.assert_called_once_with("sbx-1")


@pytest.mark.parametrize("surface", [False, True])
def test_wait_for_ready_reports_dead_server(sdk, surface):
    import requests

    provider = TensorlakeProvider(
        public=True, image="echo-env", surface_server_logs=surface, api_key="key-1"
    )
    url = provider.start_container(env_vars={"TOKEN": SECRET})
    sdk.probe.get_process.return_value.status = "exited"
    sdk.probe.get_output.return_value.lines = [
        f"token={SECRET}",
        f"listening at {url}",
        f"ws at {url.replace('https://', 'wss://')}",
        "api key-1",
        "ModuleNotFoundError",
    ]

    with patch("requests.get", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="exited") as exc:
            provider.wait_for_ready(url, timeout_s=5)

    message = str(exc.value)
    assert SECRET not in message
    assert url not in message
    assert "wss://" not in message
    assert "key-1" not in message
    assert ("ModuleNotFoundError" in message) is surface


def test_wait_for_ready_returns_on_healthy(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env", api_key="key")
    url = provider.start_container()

    with patch("requests.get", return_value=MagicMock(status_code=200)) as get:
        provider.wait_for_ready(url, timeout_s=30)

    assert get.call_args.args == (f"{url}/health",)
    assert get.call_args.kwargs["timeout"] == 5.0
    # The probe is a separate, short-timeout handle. Exiting its context must
    # close it without deleting the sandbox.
    probe_kwargs = sdk.Sandbox.call_args.kwargs
    assert probe_kwargs["identifier"] == "sbx-1"
    assert probe_kwargs["proxy_url"] == MANAGEMENT_URL
    assert probe_kwargs["routing_hint"] == "hint"
    assert probe_kwargs["api_key"] == "key"
    assert probe_kwargs["request_timeout"] == 5.0
    sdk.Sandbox.return_value.__exit__.assert_called_once()
    sdk.client.delete.assert_not_called()


def test_wait_for_ready_rejects_health_from_other_server(sdk):
    # Another process can answer on port 8000 after our server exits.
    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    sdk.probe.get_process.return_value.status = "exited"

    with patch("requests.get", return_value=MagicMock(status_code=200)):
        with pytest.raises(RuntimeError, match="exited"):
            provider.wait_for_ready(url, timeout_s=5)


def test_wait_for_ready_survives_temporary_sdk_error(sdk):
    import requests

    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    sdk.probe.get_process.side_effect = [
        sdk.SandboxError("proxy 502"),
        MagicMock(status="running"),
    ]
    responses = [requests.ConnectionError, MagicMock(status_code=200)]

    with patch("requests.get", side_effect=responses), patch("time.sleep"):
        provider.wait_for_ready(url, timeout_s=5)

    assert sdk.probe.get_process.call_count == 2


@pytest.mark.parametrize("surface", [False, True])
def test_wait_for_ready_withholds_probe_handle_errors(sdk, surface):
    # The SDK quotes the URL and the API key when the handle cannot be built.
    import requests

    provider = TensorlakeProvider(
        public=True, image="echo-env", surface_server_logs=surface, api_key=SECRET
    )
    url = provider.start_container()
    sdk.Sandbox.side_effect = sdk.SandboxError(f"proxy setup failed: {url} {SECRET}")

    with (
        patch("requests.get", side_effect=requests.ConnectionError),
        patch("time.sleep"),
    ):
        with pytest.raises(TimeoutError) as exc:
            provider.wait_for_ready(url, timeout_s=0.01)

    rendered = _rendered(exc)
    assert SECRET not in rendered
    assert url not in rendered
    assert sdk.Sandbox.call_count >= 1


def test_wait_for_ready_withholds_probe_handle_close_errors(sdk):
    import requests

    provider = TensorlakeProvider(public=True, image="echo-env", api_key=SECRET)
    url = provider.start_container()
    sdk.Sandbox.return_value.__exit__.side_effect = sdk.SandboxError(
        f"close failed: {url} {SECRET}"
    )

    with (
        patch("requests.get", side_effect=requests.ConnectionError),
        patch("time.sleep"),
    ):
        with pytest.raises(TimeoutError) as exc:
            provider.wait_for_ready(url, timeout_s=0.01)

    rendered = _rendered(exc)
    assert SECRET not in rendered
    assert url not in rendered


def test_wait_for_ready_reports_dead_server_when_handle_close_fails(sdk):
    import requests

    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    sdk.probe.get_process.return_value.status = "exited"
    sdk.Sandbox.return_value.__exit__.side_effect = sdk.SandboxError("close failed")

    with patch("requests.get", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="exited"):
            provider.wait_for_ready(url, timeout_s=5)


def test_wait_for_ready_bounds_probes_by_remaining_budget(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    # 2 seconds remain: both the SDK probe and the health request get 2.
    clock = iter([0.0, 118.0, 118.0, 118.0, 118.0])

    with (
        patch("time.monotonic", side_effect=clock),
        patch("requests.get", return_value=MagicMock(status_code=200)) as get,
    ):
        provider.wait_for_ready(url, timeout_s=120)

    assert sdk.Sandbox.call_args.kwargs["request_timeout"] == 2.0
    assert get.call_args.kwargs["timeout"] == 2.0


def test_wait_for_ready_cannot_succeed_after_its_deadline(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    clock = {"now": 0.0}

    def slow_probe(pid):
        # Model an SDK call that outlives the readiness budget.
        clock["now"] += 300.0
        return MagicMock(status="running")

    sdk.probe.get_process.side_effect = slow_probe

    with (
        patch("time.monotonic", side_effect=lambda: clock["now"]),
        patch("requests.get", return_value=MagicMock(status_code=200)) as get,
    ):
        with pytest.raises(TimeoutError):
            provider.wait_for_ready(url, timeout_s=120)

    # No health request once the budget is spent.
    get.assert_not_called()


def test_wait_for_ready_health_request_gets_the_budget_left_after_the_probe(sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    # 20 seconds remain when the loop starts; the probe uses 17 of them.
    clock = {"now": 100.0}

    def slow_probe(pid):
        clock["now"] += 17.0
        return MagicMock(status="running")

    sdk.probe.get_process.side_effect = slow_probe

    with (
        patch("time.monotonic", side_effect=lambda: clock["now"]),
        patch("requests.get", return_value=MagicMock(status_code=200)) as get,
    ):
        provider.wait_for_ready(url, timeout_s=20)

    assert sdk.Sandbox.call_args.kwargs["request_timeout"] == 5.0
    assert get.call_args.kwargs["timeout"] == 3.0


def test_dead_server_message_survives_output_error(sdk):
    import requests

    provider = TensorlakeProvider(
        public=True, image="echo-env", surface_server_logs=True
    )
    url = provider.start_container()
    sdk.probe.get_process.return_value.status = "exited"
    sdk.probe.get_output.side_effect = sdk.SandboxError

    with patch("requests.get", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="exited"):
            provider.wait_for_ready(url, timeout_s=5)
