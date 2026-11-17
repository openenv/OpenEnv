# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BoxdProvider.

The tests inject a duck-typed fake client (``_client=``), so they run without
``pip install boxd``. The fake's methods take only the arguments the provider
passes, and ``test_calls_bind_against_the_real_sdk`` binds the same calls
against the installed SDK's signatures.
"""

from __future__ import annotations

import inspect
import io
import sys
import tarfile
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.boxd_provider import BoxdProvider
from openenv.core.containers.runtime.providers import ContainerProvider

ECHO = "ghcr.io/huggingface/openenv-echo-env:latest"
HOST = "openenv-abc.fra.boxd.sh"
URL = f"https://{HOST}"


class _FakeMachines:
    def __init__(self):
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.bot_protection: list[tuple[str, str]] = []
        self.hibernate_timeouts: list[tuple[str, int]] = []
        self.pinned_ports: list[tuple[str, int]] = []
        self.egress: list[tuple[str, list[str]]] = []
        self.commands: list[str] = []
        self.exec_timeouts: list[float] = []
        self.commands_before_egress: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.files = SimpleNamespace(upload=self._upload)
        self.proxies = SimpleNamespace(set_port=self._set_port)
        self.create_error: BaseException | None = None
        self.fail_ready = False
        self.url = URL
        self.delete_error: Exception | None = None
        self.pull_exit = 0
        self.run_exit = 0
        self.build_exit = 0
        self.container_state = "running"
        self.logs = ""

    def create(
        self,
        name,
        *,
        org,
        isolated,
        labels,
        vcpu,
        memory,
        auto_suspend_timeout,
        delete_after,
    ):
        self.created.append(
            {
                "name": name,
                "org": org,
                "isolated": isolated,
                "labels": labels,
                "vcpu": vcpu,
                "memory": memory,
                "auto_suspend_timeout": auto_suspend_timeout,
                "delete_after": delete_after,
            }
        )
        if self.create_error is not None:
            # The machine exists, but the caller never learns its id.
            raise self.create_error
        return SimpleNamespace(id="m-1", name=name)

    def wait_until_ready(self, machine_id):
        if self.fail_ready:
            raise TimeoutError("not ready")
        return SimpleNamespace(id=machine_id, access=SimpleNamespace(url=self.url))

    def set_bot_protection(self, machine_id, *, mode):
        self.bot_protection.append((machine_id, mode))

    def set_auto_hibernate_timeout(self, machine_id, seconds):
        self.hibernate_timeouts.append((machine_id, seconds))

    def set_egress_allow(self, machine_id, entries):
        self.egress.append((machine_id, entries))
        self.commands_before_egress = list(self.commands)

    def _set_port(self, machine_id, port):
        self.pinned_ports.append((machine_id, port))

    def _upload(self, machine_id, path, data):
        self.uploads[path] = data
        return len(data)

    def exec(self, machine_id, command, *, timeout):
        self.commands.append(command)
        self.exec_timeouts.append(timeout)
        if command.startswith("docker pull"):
            return SimpleNamespace(exit_code=self.pull_exit, stdout="", stderr="denied")
        if "docker run" in command:
            return SimpleNamespace(
                exit_code=self.run_exit, stdout="", stderr="conflict"
            )
        if "docker build" in command:
            return SimpleNamespace(
                exit_code=self.build_exit, stdout="step 1/3", stderr="build broke"
            )
        if command.startswith("docker inspect"):
            return SimpleNamespace(
                exit_code=0, stdout=self.container_state + "\n", stderr=""
            )
        if command.startswith("docker logs"):
            return SimpleNamespace(exit_code=0, stdout=self.logs, stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    def delete(self, machine_id):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(machine_id)


class _FakeClient:
    def __init__(self):
        self.machines = _FakeMachines()
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.fixture
def client():
    return _FakeClient()


def _provider(client, **kwargs):
    return BoxdProvider(public=True, _client=client, **kwargs)


def test_is_a_container_provider(client):
    assert isinstance(_provider(client), ContainerProvider)


def test_public_ingress_is_an_explicit_opt_in(client):
    fake_boxd = types.ModuleType("boxd")
    fake_boxd.Boxd = MagicMock(name="Boxd")
    with patch.dict(sys.modules, {"boxd": fake_boxd}):
        with pytest.raises(ValueError, match="public=True"):
            BoxdProvider(api_key="test-key", image=ECHO)
        with pytest.raises(ValueError, match="public=True"):
            BoxdProvider(_client=client, public=False)
    fake_boxd.Boxd.assert_not_called()
    assert client.machines.created == []


def test_start_creates_an_isolated_machine_and_returns_its_url(client):
    provider = _provider(client, org="acme", vcpu=4, memory="16G")

    assert provider.start_container(ECHO) == URL

    created = client.machines.created[0]
    assert created["isolated"] is True
    assert created["org"] == "acme"
    assert created["vcpu"] == 4 and created["memory"] == "16G"
    assert created["labels"] == {"managed-by": "openenv"}
    assert created["name"].startswith("openenv-")
    assert provider.machine_id == "m-1"


def test_machine_gets_a_one_hour_delete_after_by_default(client):
    _provider(client).start_container(ECHO)
    assert client.machines.created[0]["delete_after"] == 3600


def test_delete_after_is_configurable_and_can_be_turned_off(client):
    _provider(client, delete_after=600).start_container(ECHO)
    other = _FakeClient()
    _provider(other, delete_after=None).start_container(ECHO)
    assert client.machines.created[0]["delete_after"] == 600
    assert other.machines.created[0]["delete_after"] is None


@pytest.mark.parametrize("delete_after", [0, -5, 1.5, True, "1h"])
def test_invalid_delete_after_is_rejected(client, delete_after):
    with pytest.raises(ValueError, match="delete_after"):
        _provider(client, delete_after=delete_after)


def test_idle_suspend_and_hibernation_are_turned_off(client):
    _provider(client).start_container(ECHO)
    assert client.machines.created[0]["auto_suspend_timeout"] == 0
    assert client.machines.hibernate_timeouts == [("m-1", 0)]


def test_default_route_is_pinned_to_port_8000(client):
    _provider(client).start_container(ECHO)
    assert client.machines.pinned_ports == [("m-1", 8000)]


def test_egress_is_left_open_by_default(client):
    _provider(client).start_container(ECHO)
    assert client.machines.egress == []


def test_egress_allow_is_applied_after_the_pull_and_before_the_run(client):
    _provider(client, egress_allow=["api.example.com"]).start_container(ECHO)
    assert client.machines.egress == [("m-1", ["api.example.com"])]
    assert client.machines.commands_before_egress == [f"docker pull -q {ECHO}"]
    assert "docker run" in client.machines.commands[-1]


def test_name_prefix(client):
    _provider(client, name_prefix="rollout").start_container(ECHO)
    assert client.machines.created[0]["name"].startswith("rollout-")


def test_org_defaults_to_the_keys_own_and_ignores_the_environment(client):
    with patch.dict("os.environ", {"BOXD_ORG": "from-env"}):
        _provider(client).start_container(ECHO)
    assert client.machines.created[0]["org"] is None


def test_bot_protection_is_turned_off(client):
    _provider(client).start_container(ECHO)
    assert client.machines.bot_protection == [("m-1", "off")]


def test_docker_run_publishes_port_8000(client):
    _provider(client).start_container(ECHO)
    run = _run_command(client)
    assert "-p 8000:8000" in run
    assert "--name openenv-server" in run
    assert f" {ECHO};" in run


def _run_command(client):
    return next(c for c in client.machines.commands if "docker run" in c)


def test_env_values_go_in_an_env_file_not_on_a_command_line(client):
    _provider(client).start_container(ECHO, env_vars={"API_TOKEN": "s3cret"})

    env_file = "/home/boxd/openenv-server.env"
    assert client.machines.uploads[env_file] == "API_TOKEN=s3cret\n"
    run = _run_command(client)
    assert f"--env-file {env_file}" in run
    assert f"rm -f {env_file}" in run
    assert "s3cret" not in "".join(client.machines.commands)


def test_no_env_file_without_env_vars(client):
    _provider(client).start_container(ECHO)
    assert "--env-file" not in _run_command(client)
    assert client.machines.uploads == {}


@pytest.mark.parametrize(
    "env_vars",
    [{"A;touch /tmp/x": "1"}, {"MY-VAR": "1"}, {"OK": "two\nlines"}],
)
def test_invalid_env_vars_are_rejected_before_create(client, env_vars):
    with pytest.raises(ValueError, match="Invalid env var"):
        _provider(client).start_container(ECHO, env_vars=env_vars)
    assert client.machines.created == []


def test_constructor_image_and_env_vars_are_defaults(client):
    provider = _provider(client, image=ECHO, env_vars={"A": "1"})
    provider.start_container()
    assert f" {ECHO};" in _run_command(client)
    assert client.machines.uploads["/home/boxd/openenv-server.env"] == "A=1\n"


def test_requires_an_image(client):
    with pytest.raises(ValueError, match="requires an image"):
        _provider(client).start_container()
    assert client.machines.created == []


@pytest.mark.parametrize("port", [None, 8000])
def test_port_8000_is_accepted(client, port):
    assert _provider(client).start_container(ECHO, port=port) == URL


def test_other_ports_are_rejected(client):
    with pytest.raises(ValueError, match="only supports port 8000"):
        _provider(client).start_container(ECHO, port=9000)
    assert client.machines.created == []


def test_unknown_options_are_rejected(client):
    with pytest.raises(TypeError, match="Unsupported"):
        _provider(client).start_container(ECHO, gpu=True)
    assert client.machines.created == []


def test_second_start_raises_instead_of_orphaning(client):
    provider = _provider(client)
    provider.start_container(ECHO)
    with pytest.raises(RuntimeError, match="already has an active machine"):
        provider.start_container(ECHO)
    assert len(client.machines.created) == 1


def test_failed_docker_pull_deletes_the_machine(client):
    client.machines.pull_exit = 1
    with pytest.raises(RuntimeError, match="docker pull failed") as err:
        _provider(client).start_container(ECHO)
    assert "denied" in str(err.value)
    assert client.machines.deleted == ["m-1"]
    assert not any("docker run" in c for c in client.machines.commands)


def test_failed_docker_run_deletes_the_machine(client):
    client.machines.run_exit = 125
    provider = _provider(client)
    with pytest.raises(RuntimeError, match="docker run failed"):
        provider.start_container(ECHO)
    assert client.machines.deleted == ["m-1"]
    with pytest.raises(RuntimeError, match="no active machine"):
        provider.machine_id


def test_failed_readiness_deletes_the_machine(client):
    client.machines.fail_ready = True
    with pytest.raises(TimeoutError):
        _provider(client).start_container(ECHO)
    assert client.machines.deleted == ["m-1"]


def test_stop_deletes_by_id_and_is_idempotent(client):
    provider = _provider(client)
    provider.start_container(ECHO)
    provider.stop_container()
    provider.stop_container()
    assert client.machines.deleted == ["m-1"]


def test_close_deletes_and_closes_the_client(client):
    with _provider(client) as provider:
        provider.start_container(ECHO)
    assert client.machines.deleted == ["m-1"]
    assert client.closed == 1


class _BoxdError(Exception):
    pass


class _NotFoundError(_BoxdError):
    pass


class _AuthenticationError(_BoxdError):
    pass


class _PermissionDeniedError(_BoxdError):
    pass


class _APIConnectionError(_BoxdError):
    pass


@pytest.fixture
def fake_boxd_errors():
    errors = types.ModuleType("boxd.errors")
    errors.BoxdError = _BoxdError
    errors.NotFoundError = _NotFoundError
    errors.AuthenticationError = _AuthenticationError
    errors.PermissionDeniedError = _PermissionDeniedError
    errors.APIConnectionError = _APIConnectionError
    boxd = types.ModuleType("boxd")
    boxd.errors = errors
    with patch.dict(sys.modules, {"boxd": boxd, "boxd.errors": errors}):
        yield


def test_failed_delete_keeps_the_id_so_stop_can_retry(client, fake_boxd_errors):
    provider = _provider(client)
    provider.start_container(ECHO)
    client.machines.delete_error = ConnectionError("network down")
    with pytest.raises(ConnectionError):
        provider.stop_container()
    assert provider.machine_id == "m-1"

    client.machines.delete_error = None
    provider.stop_container()
    assert client.machines.deleted == ["m-1"]
    with pytest.raises(RuntimeError, match="no active machine"):
        provider.machine_id


def test_already_deleted_machine_counts_as_stopped(client, fake_boxd_errors):
    provider = _provider(client)
    provider.start_container(ECHO)
    client.machines.delete_error = _NotFoundError("gone")
    provider.stop_container()
    with pytest.raises(RuntimeError, match="no active machine"):
        provider.machine_id


def test_failed_start_with_failed_cleanup_raises_the_start_error(
    client, fake_boxd_errors, caplog
):
    client.machines.run_exit = 125
    client.machines.delete_error = ConnectionError("network down")
    provider = _provider(client)
    with pytest.raises(RuntimeError, match="docker run failed"):
        provider.start_container(ECHO)
    assert "Could not delete the boxd machine" in caplog.text
    # Neither the id nor the name (the secret part of the URL) is logged.
    assert "m-1" not in caplog.text
    assert client.machines.created[0]["name"] not in caplog.text
    assert provider.machine_id == "m-1"


@pytest.mark.parametrize(
    "error", [TimeoutError("deadline exceeded"), KeyboardInterrupt]
)
def test_machine_created_by_a_failed_create_is_deleted_by_name(client, error):
    client.machines.create_error = error
    provider = _provider(client, delete_after=None)
    with pytest.raises(type(error) if isinstance(error, Exception) else error):
        provider.start_container(ECHO)
    assert client.machines.deleted == [client.machines.created[0]["name"]]
    with pytest.raises(RuntimeError, match="no active machine"):
        provider.machine_id


def test_failed_create_that_made_no_machine_counts_as_deleted(client, fake_boxd_errors):
    client.machines.create_error = ValueError("bad size")
    client.machines.delete_error = _NotFoundError("gone")
    provider = _provider(client)
    with pytest.raises(ValueError, match="bad size"):
        provider.start_container(ECHO)
    with pytest.raises(RuntimeError, match="no active machine"):
        provider.machine_id


def test_failed_create_with_failed_cleanup_keeps_the_name_for_a_retry(
    client, fake_boxd_errors, caplog
):
    client.machines.create_error = TimeoutError("deadline exceeded")
    client.machines.delete_error = ConnectionError("network down")
    provider = _provider(client)
    with pytest.raises(TimeoutError):
        provider.start_container(ECHO)
    name = client.machines.created[0]["name"]
    assert name not in caplog.text
    with pytest.raises(RuntimeError, match="already has an active machine"):
        provider.start_container(ECHO)

    client.machines.delete_error = None
    provider.close()
    assert client.machines.deleted == [name]


def test_interrupted_start_deletes_the_machine(client):
    client.machines.set_bot_protection = MagicMock(side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        _provider(client).start_container(ECHO)
    assert client.machines.deleted == ["m-1"]


def _health(status):
    response = MagicMock()
    response.status_code = status
    return response


def test_wait_for_ready_returns_on_healthy(client):
    provider = _provider(client)
    url = provider.start_container(ECHO)
    with patch("requests.get", return_value=_health(200)) as get:
        provider.wait_for_ready(url)
    get.assert_called_once_with(f"{URL}/health", timeout=5.0)


def test_wait_for_ready_raises_when_the_container_exited(client):
    client.machines.container_state = "exited"
    client.machines.logs = "token=s3cret crashed"
    provider = _provider(client)
    url = provider.start_container(ECHO, env_vars={"T": "s3cret"})
    with patch("requests.get", return_value=_health(502)):
        with pytest.raises(RuntimeError, match="exited during startup") as err:
            provider.wait_for_ready(url, timeout_s=5)
    assert "s3cret" not in str(err.value)
    assert "crashed" not in str(err.value)


def test_surface_server_logs_includes_redacted_output(client):
    client.machines.container_state = "exited"
    client.machines.logs = "token=s3cret crashed"
    provider = _provider(client, surface_server_logs=True)
    url = provider.start_container(ECHO, env_vars={"T": "s3cret"})
    with patch("requests.get", return_value=_health(502)):
        with pytest.raises(RuntimeError) as err:
            provider.wait_for_ready(url, timeout_s=5)
    assert "token=*** crashed" in str(err.value)


def test_surface_server_logs_redacts_the_machine_hostname(client):
    client.machines.container_state = "exited"
    client.machines.logs = f"GET /health Host: {HOST}\ntoken={HOST}-suffix boom\n"
    provider = _provider(client, surface_server_logs=True)
    # An env value that contains the hostname is replaced first, whole.
    url = provider.start_container(ECHO, env_vars={"T": f"{HOST}-suffix"})
    with patch("requests.get", return_value=_health(502)):
        with pytest.raises(RuntimeError) as err:
            provider.wait_for_ready(url, timeout_s=5)
    assert "Host: ***\ntoken=*** boom" in str(err.value)
    assert HOST not in str(err.value)


def test_wait_for_ready_times_out(client):
    provider = _provider(client)
    url = provider.start_container(ECHO)
    with (
        patch("requests.get", return_value=_health(502)),
        patch("time.sleep"),
        patch("time.monotonic", side_effect=[0.0, 0.0, 0.0, 10.0, 10.0]),
    ):
        with pytest.raises(TimeoutError, match="did not become ready") as err:
            provider.wait_for_ready(url, timeout_s=5)
    # The URL is the environment's address, so it stays out of errors.
    assert url not in str(err.value)


def test_wait_for_ready_probes_stay_inside_the_budget(client):
    provider = _provider(client)
    url = provider.start_container(ECHO)
    clock = SimpleNamespace(now=0.0)
    http_timeouts = []

    def get(_url, *, timeout):
        http_timeouts.append(timeout)
        clock.now += timeout  # the probe hangs for its whole timeout
        return _health(502)

    original = client.machines.exec

    def exec_(machine_id, command, *, timeout):
        clock.now += timeout  # so does the exec
        return original(machine_id, command, timeout=timeout)

    client.machines.exec = exec_
    client.machines.exec_timeouts.clear()
    with (
        patch("requests.get", side_effect=get),
        patch("time.sleep"),
        patch("time.monotonic", side_effect=lambda: clock.now),
    ):
        with pytest.raises(TimeoutError):
            provider.wait_for_ready(url, timeout_s=7)
    assert clock.now == 7
    assert http_timeouts == [5.0]
    assert client.machines.exec_timeouts == [2.0]


def _clocked_wait(client, provider, url, timeout_s, inspect):
    """Run wait_for_ready on a fake clock. Health checks answer 502 at once;
    ``inspect(clock, timeout)`` stands in for the ``docker inspect`` exec."""
    clock = SimpleNamespace(now=0.0)
    original = client.machines.exec

    def exec_(machine_id, command, *, timeout):
        if command.startswith("docker inspect"):
            client.machines.exec_timeouts.append(timeout)
            return inspect(clock, timeout)
        return original(machine_id, command, timeout=timeout)

    client.machines.exec = exec_
    client.machines.exec_timeouts.clear()
    with (
        patch("requests.get", return_value=_health(502)),
        patch("time.sleep", side_effect=lambda s: setattr(clock, "now", clock.now + s)),
        patch("time.monotonic", side_effect=lambda: clock.now),
    ):
        provider.wait_for_ready(url, timeout_s=timeout_s)


def test_probe_timeout_near_the_deadline_raises_timeout_error(client, fake_boxd_errors):
    provider = _provider(client)
    url = provider.start_container(ECHO)

    def inspect(clock, timeout):
        # The exec hangs until its own timeout, then the SDK gives up on it.
        clock.now += timeout
        raise _BoxdError(f"exec timed out after {timeout}s on {HOST}")

    with pytest.raises(TimeoutError, match="did not become ready") as err:
        _clocked_wait(client, provider, url, 2.5, inspect)
    assert HOST not in str(err.value)
    assert err.value.__cause__ is None and err.value.__context__ is None
    assert client.machines.exec_timeouts == [2.5]


def test_probe_transport_error_keeps_polling(client, fake_boxd_errors):
    provider = _provider(client)
    url = provider.start_container(ECHO)
    states = iter([_APIConnectionError("connection reset"), "exited"])

    def inspect(clock, timeout):
        state = next(states)
        if isinstance(state, Exception):
            raise state
        return SimpleNamespace(exit_code=0, stdout=state + "\n", stderr="")

    with pytest.raises(RuntimeError, match="exited during startup"):
        _clocked_wait(client, provider, url, 30, inspect)
    assert len(client.machines.exec_timeouts) == 2


@pytest.mark.parametrize(
    "error", [_NotFoundError("gone"), _AuthenticationError("bad key")]
)
def test_definite_probe_errors_are_raised(client, fake_boxd_errors, error):
    provider = _provider(client)
    url = provider.start_container(ECHO)

    def inspect(clock, timeout):
        raise error

    with pytest.raises(type(error)):
        _clocked_wait(client, provider, url, 30, inspect)


def test_no_probe_with_less_than_a_second_left(client):
    provider = _provider(client)
    url = provider.start_container(ECHO)

    def inspect(clock, timeout):
        clock.now += 1.0
        return SimpleNamespace(exit_code=0, stdout="running\n", stderr="")

    with pytest.raises(TimeoutError):
        _clocked_wait(client, provider, url, 2.5, inspect)
    # Probes at 0 s (2.5 s left) and 2 s (0.5 s left): only the first runs.
    assert client.machines.exec_timeouts == [2.5]


def test_container_log_read_is_bounded(client):
    client.machines.container_state = "exited"
    provider = _provider(client, surface_server_logs=True)
    url = provider.start_container(ECHO)
    with patch("requests.get", return_value=_health(502)):
        with pytest.raises(RuntimeError):
            provider.wait_for_ready(url, timeout_s=5)
    assert client.machines.commands[-1].startswith("docker logs")
    assert client.machines.exec_timeouts[-1] == 10.0


def test_failed_container_log_read_still_reports_the_exit(client, fake_boxd_errors):
    client.machines.container_state = "exited"
    provider = _provider(client, surface_server_logs=True)
    url = provider.start_container(ECHO)
    original = client.machines.exec

    def exec_(machine_id, command, *, timeout):
        if command.startswith("docker logs"):
            raise _BoxdError(f"exec timed out after {timeout}s on {HOST}")
        return original(machine_id, command, timeout=timeout)

    client.machines.exec = exec_
    with patch("requests.get", return_value=_health(502)):
        with pytest.raises(RuntimeError, match="exited during startup") as err:
            provider.wait_for_ready(url, timeout_s=5)
    assert "could not be read" in str(err.value)
    assert HOST not in str(err.value)


def test_missing_url_is_refused_and_the_machine_deleted(client):
    # The SDK returns "" while the machine's domain is not known.
    client.machines.url = ""
    with pytest.raises(RuntimeError, match="non-HTTPS"):
        _provider(client).start_container(ECHO)
    assert client.machines.deleted == ["m-1"]
    assert client.machines.commands == []


def _env_dir(tmp_path):
    env = tmp_path / "my_env"
    (env / "server").mkdir(parents=True)
    (env / "server" / "Dockerfile").write_text(
        'FROM python:3.11\nCOPY . /app/env\nCMD ["server"]\n'
    )
    (env / "app.py").write_text("print('hi')\n")
    (env / ".venv").mkdir()
    (env / ".venv" / "big").write_text("x")
    return env


def test_image_from_dockerfile_returns_a_uri(tmp_path):
    env = _env_dir(tmp_path)
    uri = BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))
    assert uri == f"dockerfile:{(env / 'server' / 'Dockerfile').resolve()}"


def test_image_from_dockerfile_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BoxdProvider.image_from_dockerfile(str(tmp_path / "Dockerfile"))


def test_image_from_dockerfile_bad_context(tmp_path):
    env = _env_dir(tmp_path)
    with pytest.raises(ValueError, match="context_dir does not exist"):
        BoxdProvider.image_from_dockerfile(
            str(env / "server" / "Dockerfile"), context_dir=str(tmp_path / "nope")
        )


def test_image_from_dockerfile_missing_copy_source(tmp_path):
    env = _env_dir(tmp_path)
    (env / "server" / "Dockerfile").write_text("FROM python:3.11\nCOPY missing /x\n")
    with pytest.raises(ValueError, match="COPY source 'missing'"):
        BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))


def test_dockerfile_image_is_built_in_the_machine(client, tmp_path):
    env = _env_dir(tmp_path)
    image = BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))

    _provider(client).start_container(image)

    uploads = client.machines.uploads
    dockerfile = uploads["/home/boxd/openenv-build.Dockerfile"]
    assert dockerfile.startswith(b"FROM python:3.11")
    tar = tarfile.open(
        fileobj=io.BytesIO(uploads["/home/boxd/openenv-build.tar.gz"]), mode="r:gz"
    )
    names = tar.getnames()
    assert "./app.py" in names and "./server/Dockerfile" in names
    assert not any(".venv" in n for n in names)

    commands = client.machines.commands
    build = next(i for i, c in enumerate(commands) if "docker build" in c)
    run = next(i for i, c in enumerate(commands) if "docker run" in c)
    assert build < run
    assert " openenv-env:latest;" in commands[run]
    assert not any(c.startswith("docker pull") for c in commands)


def test_egress_allow_is_applied_after_the_build(client, tmp_path):
    env = _env_dir(tmp_path)
    image = BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))
    _provider(client, egress_allow=["none"]).start_container(image)
    assert client.machines.egress == [("m-1", ["none"])]
    assert "docker build" in client.machines.commands_before_egress[-1]
    assert "docker run" in client.machines.commands[-1]


def test_failed_build_deletes_the_machine(client, tmp_path):
    env = _env_dir(tmp_path)
    image = BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))
    client.machines.build_exit = 1
    with pytest.raises(RuntimeError, match="docker build failed") as err:
        _provider(client).start_container(image)
    assert client.machines.deleted == ["m-1"]
    # Build output comes from the environment's own RUN steps: opt-in only.
    assert "build broke" not in str(err.value)
    assert "surface_server_logs=True" in str(err.value)


def test_failed_build_output_with_surface_server_logs(client, tmp_path):
    env = _env_dir(tmp_path)
    image = BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))
    client.machines.build_exit = 1
    with pytest.raises(RuntimeError, match="build broke"):
        _provider(client, surface_server_logs=True).start_container(image)


def test_unregistered_dockerfile_uri(client):
    with pytest.raises(ValueError, match="image_from_dockerfile"):
        _provider(client).start_container("dockerfile:/nowhere/Dockerfile")
    assert client.machines.created == []


def test_real_client_is_built_from_api_key_and_region():
    fake_boxd = types.ModuleType("boxd")
    fake_boxd.Boxd = MagicMock(name="Boxd")
    with patch.dict(sys.modules, {"boxd": fake_boxd}):
        BoxdProvider(public=True, api_key="test-key", region="nyc")
    fake_boxd.Boxd.assert_called_once_with(api_key="test-key", region="nyc")


def test_calls_bind_against_the_real_sdk():
    """Every SDK call the provider makes exists with the arguments it passes."""
    boxd = pytest.importorskip("boxd")
    from boxd.resources.machines import Files, Machines, Proxies

    calls = [
        (boxd.Boxd.__init__, (), {"api_key": "k", "region": "nyc"}),
        (
            Machines.create,
            ("openenv-0123456789ab",),
            {
                "org": None,
                "isolated": True,
                "labels": {"managed-by": "openenv"},
                "vcpu": 4,
                "memory": "16G",
                "auto_suspend_timeout": 0,
                "delete_after": 3600,
            },
        ),
        (Machines.wait_until_ready, ("m-1",), {}),
        (Machines.set_auto_hibernate_timeout, ("m-1", 0), {}),
        (Proxies.set_port, ("m-1", 8000), {}),
        (Machines.set_bot_protection, ("m-1",), {"mode": "off"}),
        (Machines.set_egress_allow, ("m-1", ["api.example.com"]), {}),
        (Machines.exec, ("m-1", "true"), {"timeout": 30.0}),
        (Files.upload, ("m-1", "/p", b"x"), {}),
        (Files.upload, ("m-1", "/p", "x"), {}),
        (Machines.delete, ("m-1",), {}),
        (boxd.Boxd.close, (), {}),
    ]
    for func, args, kwargs in calls:
        inspect.signature(func).bind(object(), *args, **kwargs)
    assert issubclass(boxd.errors.NotFoundError, Exception)
