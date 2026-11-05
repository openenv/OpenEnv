import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openenv.core.containers.runtime import LocalDockerProvider
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
