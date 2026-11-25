import re
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openenv.core.containers.runtime import DockerSwarmProvider, LocalDockerProvider
from openenv.core.generic_client import GenericEnvClient
from websockets.protocol import State


def make_provider(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="container-id\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    provider = LocalDockerProvider()
    monkeypatch.setattr(
        provider, "_generate_container_name", lambda _: "test-container"
    )
    return provider, commands


def test_start_container_forwards_volume_mounts(monkeypatch):
    provider, commands = make_provider(monkeypatch)

    base_url = provider.start_container(
        "retro-env:latest",
        port=8123,
        volumes={
            "/host/roms": {
                "bind": "/roms",
                "mode": "ro",
            },
            "game-data": {
                "bind": "/data",
            },
        },
    )

    assert base_url == "http://localhost:8123"
    assert commands[0] == ["docker", "version"]
    assert commands[1] == [
        "docker",
        "run",
        "-d",
        "--name",
        "test-container",
        "-p",
        "8123:8000",
        "--volume",
        "/host/roms:/roms:ro",
        "--volume",
        "game-data:/data",
        "retro-env:latest",
    ]


def test_start_container_rejects_volume_without_bind(monkeypatch):
    provider, _ = make_provider(monkeypatch)

    with pytest.raises(
        ValueError,
        match="Volume config for '/host/roms' must include a non-empty 'bind'",
    ):
        provider.start_container(
            "retro-env:latest",
            port=8123,
            volumes={"/host/roms": {}},
        )


@pytest.mark.parametrize(
    "kwargs", [{"workers": 4}, {"environment": {"EXAMPLE": "value"}}]
)
def test_start_container_preserves_ignored_options(monkeypatch, kwargs):
    provider, commands = make_provider(monkeypatch)

    assert provider.start_container("retro-env:latest", port=8123, **kwargs) == (
        "http://localhost:8123"
    )
    assert commands[1][-1] == "retro-env:latest"
    assert len(commands[1]) == 8


@pytest.mark.asyncio
async def test_from_env_keeps_workers_compatible_with_default_docker(monkeypatch):
    provider, commands = make_provider(monkeypatch)
    monkeypatch.setattr("openenv.core.env_client.LocalDockerProvider", lambda: provider)
    monkeypatch.setattr(provider, "wait_for_ready", lambda _: None)
    websocket = Mock(state=State.OPEN, close=AsyncMock(), send=AsyncMock())
    monkeypatch.setattr(
        "openenv.core.env_client.ws_connect", AsyncMock(return_value=websocket)
    )

    client = await GenericEnvClient.from_env("audit/test-env", port=8123, workers=4)
    try:
        assert commands[1][-1] == "registry.hf.space/audit-test-env:latest"
        assert len(commands[1]) == 8
    finally:
        await client.close()


@pytest.mark.parametrize("provider_cls", [LocalDockerProvider, DockerSwarmProvider])
def test_env_values_reach_only_the_container(monkeypatch, provider_cls):
    env_vars = {"HF_TOKEN": "fake-token-value", "DOCKER_HOST": "tcp://fake-host:1"}
    seen = {}

    def fake_run(command, **kwargs):
        if command[1] in ("run", "service") and "--env-file" in command:
            seen["env_file"] = command[command.index("--env-file") + 1]
            seen["stdin"] = kwargs.get("input")
            seen["command"] = command
            seen["env"] = kwargs.get("env")
            raise subprocess.CalledProcessError(125, command, "", "no such image")
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(provider_cls, "__init__", lambda self: None)
    provider = provider_cls()
    if provider_cls is DockerSwarmProvider:
        provider._overlay_network = None

    with pytest.raises(RuntimeError) as error:
        provider.start_container("echo-env:latest", env_vars=env_vars)

    # Docker reads them from its stdin: no file, so a confined CLI gets them too.
    assert seen["env_file"] == "/dev/stdin"
    assert seen["stdin"] == (
        "HF_TOKEN=fake-token-value\nDOCKER_HOST=tcp://fake-host:1\n"
    )
    assert "fake-token-value" not in " ".join(seen["command"])
    assert "fake-token-value" not in str(error.value)
    # The docker CLI keeps its own environment: DOCKER_HOST is the container's.
    assert seen["env"] is None


@pytest.mark.parametrize(
    "provider_cls, name_method",
    [
        (LocalDockerProvider, "_generate_container_name"),
        (DockerSwarmProvider, "_generate_service_name"),
    ],
)
@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/openenv/echo-env@sha256:" + "a" * 64,
        "ghcr.io/openenv/echo-env:1.0@sha256:" + "b" * 64,
    ],
)
def test_generated_name_drops_image_digest(provider_cls, name_method, image):
    provider = provider_cls.__new__(provider_cls)

    name = getattr(provider, name_method)(image)

    assert re.fullmatch(r"echo-env-(swarm-)?\d+", name)


@pytest.mark.parametrize(
    "env_vars", [{"KEY": "two\nlines"}, {"A=B": "x"}, {"": "x"}, {"#KEY": "x"}]
)
def test_env_values_the_env_file_cannot_hold_are_rejected(monkeypatch, env_vars):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0)
    )
    with pytest.raises(ValueError):
        LocalDockerProvider().start_container("echo-env:latest", env_vars=env_vars)
