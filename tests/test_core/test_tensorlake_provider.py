# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TensorlakeProvider. All tests use a fake tensorlake SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.tensorlake_provider import TensorlakeProvider

OPENENV_YAML = b"spec_version: 1\nname: echo\napp: server.app:app\nport: 8000\n"


def _make_sandbox(yaml_content=OPENENV_YAML):
    sandbox = MagicMock()
    sandbox.sandbox_id = "sbx-1"
    if yaml_content is None:
        sandbox.read_file.side_effect = RuntimeError("not found")
    else:
        sandbox.read_file.return_value.value = yaml_content
    sandbox.start_process.return_value.pid = 42
    sandbox.update.return_value.ingress_endpoint = "https://sandbox.tensorlake.ai"
    sandbox.update.return_value.sandbox_id = "sbx-1"
    sandbox.get_process.return_value.status = "running"
    return sandbox


@pytest.fixture
def fake_sdk():
    """Install a fake ``tensorlake.sandbox`` module and yield its Sandbox mock."""
    tensorlake_mod = types.ModuleType("tensorlake")
    sandbox_mod = types.ModuleType("tensorlake.sandbox")
    sandbox_mod.Sandbox = MagicMock()
    sandbox_mod.SandboxError = type("SandboxError", (Exception,), {})
    sandbox_mod.SandboxNotFoundError = type(
        "SandboxNotFoundError", (sandbox_mod.SandboxError,), {}
    )
    sandbox_mod.SandboxClient = MagicMock()
    sandbox_mod.Sandbox.create.return_value = _make_sandbox()
    sandbox_mod.sandbox_url_from_ingress_endpoint = lambda endpoint, sandbox_id, port: (
        f"https://{port}-{sandbox_id}.sandbox.tensorlake.ai"
    )
    tensorlake_mod.sandbox = sandbox_mod
    with patch.dict(
        sys.modules, {"tensorlake": tensorlake_mod, "tensorlake.sandbox": sandbox_mod}
    ):
        yield sandbox_mod.Sandbox


def test_start_container_launches_server_and_returns_public_url(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env", api_key="key", cpus=2)

    url = provider.start_container(env_vars={"A": "1"})

    assert url == "https://8000-sbx-1.sandbox.tensorlake.ai"
    create_kwargs = fake_sdk.create.call_args.kwargs
    assert create_kwargs["image"] == "echo-env"
    assert create_kwargs["api_key"] == "key"
    assert create_kwargs["cpus"] == 2
    assert create_kwargs["timeout_secs"] == 3600
    assert create_kwargs["deny_out"] == ["169.254.0.0/16"]
    assert create_kwargs["allow_out"] is None
    assert create_kwargs["allow_internet_access"] is True
    # The image CMD must not start a second server that races for port 8000.
    assert create_kwargs["entrypoint"] == ["sh", "-c", "exec sleep infinity"]
    sandbox = fake_sdk.create.return_value
    cmd, args = sandbox.start_process.call_args.args
    assert cmd == "sh"
    assert args[1] == (
        "cd /app/env && python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"
    )
    assert sandbox.start_process.call_args.kwargs["env"] == {"A": "1"}
    sandbox.update.assert_called_once_with(
        allow_unauthenticated_access=True, exposed_ports=[8000]
    )


@pytest.mark.parametrize("public", [None, False])
def test_public_ingress_requires_explicit_opt_in(public):
    with pytest.raises(ValueError, match="public=True"):
        TensorlakeProvider(image="echo-env", public=public)


def test_metadata_egress_stays_blocked_with_custom_policy(fake_sdk):
    provider = TensorlakeProvider(
        public=True,
        image="echo-env",
        allow_out=["huggingface.co"],
        deny_out=["10.0.0.0/8"],
    )

    provider.start_container()

    create_kwargs = fake_sdk.create.call_args.kwargs
    assert create_kwargs["allow_out"] == ["huggingface.co"]
    assert create_kwargs["deny_out"] == ["169.254.0.0/16", "10.0.0.0/8"]
    assert create_kwargs["allow_internet_access"] is True


def test_empty_allowlist_blocks_all_egress(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env", allow_out=[])

    provider.start_container()

    create_kwargs = fake_sdk.create.call_args.kwargs
    assert create_kwargs["allow_internet_access"] is False
    assert create_kwargs["allow_out"] == []


def test_cleanup_failure_does_not_hide_start_error(fake_sdk):
    fake_sdk.create.return_value = _make_sandbox(yaml_content=None)
    fake_sdk.create.return_value.terminate.side_effect = ConnectionError("network")
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(ValueError, match="cmd="):
        provider.start_container()


def test_explicit_cmd_skips_discovery(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")

    provider.start_container(cmd="run-server")

    sandbox = fake_sdk.create.return_value
    sandbox.read_file.assert_not_called()
    assert sandbox.start_process.call_args.args[1] == ["-c", "run-server"]


def test_missing_openenv_yaml_terminates_sandbox(fake_sdk):
    fake_sdk.create.return_value = _make_sandbox(yaml_content=None)
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(ValueError, match="cmd="):
        provider.start_container()

    fake_sdk.create.return_value.terminate.assert_called_once()


def test_rejects_bad_arguments(fake_sdk):
    provider = TensorlakeProvider(public=True)
    with pytest.raises(ValueError, match="requires an image"):
        provider.start_container()
    with pytest.raises(ValueError, match="port 8000"):
        provider.start_container("echo-env", port=9000)
    with pytest.raises(ValueError, match="Unsupported"):
        provider.start_container("echo-env", typo=True)
    fake_sdk.create.assert_not_called()


def test_accepts_autoenv_wait_timeout(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    assert provider.start_container(wait_timeout=30.0).startswith("https://")


def test_stop_container_terminates_once(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()
    sandbox = fake_sdk.create.return_value

    provider.stop_container()
    provider.stop_container()

    sandbox.terminate.assert_called_once()


def test_failed_stop_is_retried_by_id(fake_sdk):
    # The real Sandbox.terminate() is one-shot, so a retry must delete by ID.
    provider = TensorlakeProvider(public=True, image="echo-env", api_key="key")
    provider.start_container()
    sandbox = fake_sdk.create.return_value
    # Model the real SDK: the first call fails during delete, and later calls
    # on the same handle return without error but delete nothing.
    sandbox.terminate.side_effect = [ConnectionError("network"), None, None]

    with pytest.raises(ConnectionError):
        provider.stop_container()
    with pytest.raises(RuntimeError, match="already active"):
        provider.start_container()
    provider.stop_container()
    provider.stop_container()

    sandbox.terminate.assert_called_once()
    client = sys.modules["tensorlake.sandbox"].SandboxClient
    client.assert_called_once_with(api_key="key")
    client.return_value.delete.assert_called_once_with("sbx-1")


def test_retry_after_sandbox_is_gone_succeeds(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()
    fake_sdk.create.return_value.terminate.side_effect = ConnectionError("network")
    with pytest.raises(ConnectionError):
        provider.stop_container()
    sdk = sys.modules["tensorlake.sandbox"]
    sdk.SandboxClient.return_value.delete.side_effect = sdk.SandboxNotFoundError

    provider.stop_container()

    assert provider.start_container().startswith("https://")


def test_stop_on_expired_sandbox_succeeds(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()
    fake_sdk.create.return_value.terminate.side_effect = sys.modules[
        "tensorlake.sandbox"
    ].SandboxNotFoundError

    provider.stop_container()

    assert provider.start_container().startswith("https://")


def test_constructor_env_vars_merge_with_call_env_vars(fake_sdk):
    provider = TensorlakeProvider(
        public=True, image="echo-env", env_vars={"TOKEN": "t", "A": "1"}
    )

    provider.start_container(env_vars={"A": "2"})

    sandbox = fake_sdk.create.return_value
    assert sandbox.start_process.call_args.kwargs["env"] == {"TOKEN": "t", "A": "2"}


def test_rejects_non_https_url(fake_sdk):
    sys.modules["tensorlake.sandbox"].sandbox_url_from_ingress_endpoint = (
        lambda endpoint, sandbox_id, port: "http://insecure"
    )
    provider = TensorlakeProvider(public=True, image="echo-env")

    with pytest.raises(RuntimeError, match="non-HTTPS") as exc:
        provider.start_container()

    assert "insecure" not in str(exc.value)
    fake_sdk.create.return_value.terminate.assert_called_once()


def test_second_start_is_rejected(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    provider.start_container()

    with pytest.raises(RuntimeError, match="already active"):
        provider.start_container()

    fake_sdk.create.assert_called_once()


def test_context_manager_exit_terminates_sandbox(fake_sdk):
    with TensorlakeProvider(public=True, image="echo-env") as provider:
        provider.start_container()

    fake_sdk.create.return_value.terminate.assert_called_once()


@pytest.mark.parametrize("surface", [False, True])
def test_wait_for_ready_reports_dead_server(fake_sdk, surface):
    import requests

    provider = TensorlakeProvider(
        public=True, image="echo-env", surface_server_logs=surface
    )
    url = provider.start_container(env_vars={"TOKEN": "s3cret"})
    sandbox = fake_sdk.create.return_value
    sandbox.get_process.return_value.status = "exited"
    sandbox.get_output.return_value.lines = ["token=s3cret", "ModuleNotFoundError"]

    with patch("requests.get", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="exited") as exc:
            provider.wait_for_ready(url, timeout_s=5)

    assert "s3cret" not in str(exc.value)
    assert ("ModuleNotFoundError" in str(exc.value)) is surface


def test_wait_for_ready_returns_on_healthy(fake_sdk):
    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()

    with patch("requests.get", return_value=MagicMock(status_code=200)) as get:
        provider.wait_for_ready(url, timeout_s=5)

    get.assert_called_once_with(f"{url}/health", timeout=5.0)


def test_wait_for_ready_rejects_health_from_other_server(fake_sdk):
    # Another process can answer on port 8000 after our server exits.
    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    fake_sdk.create.return_value.get_process.return_value.status = "exited"

    with patch("requests.get", return_value=MagicMock(status_code=200)):
        with pytest.raises(RuntimeError, match="exited"):
            provider.wait_for_ready(url, timeout_s=5)


def test_wait_for_ready_survives_temporary_sdk_error(fake_sdk):
    import requests

    provider = TensorlakeProvider(public=True, image="echo-env")
    url = provider.start_container()
    sandbox = fake_sdk.create.return_value
    sandbox_error = sys.modules["tensorlake.sandbox"].SandboxError
    sandbox.get_process.side_effect = [
        sandbox_error("proxy 502"),
        MagicMock(status="running"),
    ]
    responses = [requests.ConnectionError, MagicMock(status_code=200)]

    with patch("requests.get", side_effect=responses), patch("time.sleep"):
        provider.wait_for_ready(url, timeout_s=5)

    assert sandbox.get_process.call_count == 2


def test_dead_server_message_survives_output_error(fake_sdk):
    import requests

    provider = TensorlakeProvider(
        public=True, image="echo-env", surface_server_logs=True
    )
    url = provider.start_container()
    sandbox = fake_sdk.create.return_value
    sandbox.get_process.return_value.status = "exited"
    sandbox.get_output.side_effect = sys.modules["tensorlake.sandbox"].SandboxError

    with patch("requests.get", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="exited"):
            provider.wait_for_ready(url, timeout_s=5)
