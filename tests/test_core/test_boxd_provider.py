# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for BoxdProvider.

The tests inject a duck-typed fake client (``_client=``), so they run without
``pip install boxd``. One test installs a minimal fake ``boxd`` module to pin
how the real client is constructed.
"""

from __future__ import annotations

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
URL = "https://openenv-abc.fra.boxd.sh"


class _FakeMachines:
    def __init__(self):
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.bot_protection: list[tuple[str, str]] = []
        self.commands: list[str] = []
        self.exec_envs: list[dict | None] = []
        self.uploads: dict[str, bytes] = {}
        self.files = SimpleNamespace(upload=self._upload)
        self.fail_ready = False
        self.delete_error: Exception | None = None
        self.run_exit = 0
        self.build_exit = 0
        self.container_state = "running"
        self.logs = ""

    def create(self, name, **kwargs):
        self.created.append({"name": name, **kwargs})
        return SimpleNamespace(id="m-1", name=name)

    def wait_until_ready(self, machine_id):
        if self.fail_ready:
            raise TimeoutError("not ready")
        return SimpleNamespace(id=machine_id, access=SimpleNamespace(url=URL))

    def set_bot_protection(self, machine_id, *, mode):
        self.bot_protection.append((machine_id, mode))

    def _upload(self, machine_id, path, data):
        self.uploads[path] = data
        return len(data)

    def exec(self, machine_id, command, *, env=None, timeout=None):
        self.commands.append(command)
        self.exec_envs.append(env)
        if "docker run" in command:
            return SimpleNamespace(exit_code=self.run_exit, stdout="", stderr="denied")
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
    return BoxdProvider(_client=client, **kwargs)


def test_is_a_container_provider(client):
    assert isinstance(_provider(client), ContainerProvider)


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


def test_name_prefix(client):
    _provider(client, name_prefix="rollout").start_container(ECHO)
    assert client.machines.created[0]["name"].startswith("rollout-")


def test_org_falls_back_to_env(client):
    with patch.dict("os.environ", {"BOXD_ORG": "from-env"}):
        _provider(client).start_container(ECHO)
    assert client.machines.created[0]["org"] == "from-env"


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

    assert client.machines.uploads["/home/boxd/openenv-server.env"] == (
        "API_TOKEN=s3cret\n"
    )
    run = _run_command(client)
    assert "--env-file /home/boxd/openenv-server.env" in run
    assert "rm -f /home/boxd/openenv-server.env" in run
    assert "s3cret" not in "".join(client.machines.commands)
    # The SDK turns exec env into a prefix of the command text, so the
    # provider never passes env to exec.
    assert client.machines.exec_envs == [None] * len(client.machines.exec_envs)


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
    assert _run_command(client).endswith(
        f"{ECHO}; rc=$?; rm -f /home/boxd/openenv-server.env; exit $rc"
    )
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


class _NotFoundError(Exception):
    pass


@pytest.fixture
def fake_boxd_errors():
    errors = types.ModuleType("boxd.errors")
    errors.NotFoundError = _NotFoundError
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
    assert "m-1" in caplog.text
    assert provider.machine_id == "m-1"


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
        provider.wait_for_ready(url, timeout_s=5)
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


def test_wait_for_ready_times_out(client):
    provider = _provider(client)
    url = provider.start_container(ECHO)
    with (
        patch("requests.get", return_value=_health(502)),
        patch("time.sleep"),
        patch("time.time", side_effect=[0.0, 0.0, 10.0]),
    ):
        with pytest.raises(TimeoutError, match="did not become ready"):
            provider.wait_for_ready(url, timeout_s=5)


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


def test_failed_build_deletes_the_machine(client, tmp_path):
    env = _env_dir(tmp_path)
    image = BoxdProvider.image_from_dockerfile(str(env / "server" / "Dockerfile"))
    client.machines.build_exit = 1
    with pytest.raises(RuntimeError, match="docker build failed"):
        _provider(client).start_container(image)
    assert client.machines.deleted == ["m-1"]


def test_unregistered_dockerfile_uri(client):
    with pytest.raises(ValueError, match="image_from_dockerfile"):
        _provider(client).start_container("dockerfile:/nowhere/Dockerfile")
    assert client.machines.created == []


def test_real_client_is_built_from_api_key_and_region():
    fake_boxd = types.ModuleType("boxd")
    fake_boxd.Boxd = MagicMock(name="Boxd")
    with patch.dict(sys.modules, {"boxd": fake_boxd}):
        BoxdProvider(api_key="test-key", region="nyc")
    fake_boxd.Boxd.assert_called_once_with(api_key="test-key", region="nyc")
