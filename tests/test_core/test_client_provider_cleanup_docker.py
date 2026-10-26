# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in EnvClient cleanup tests with real, fault-injected Docker resources.

Run with OPENENV_DOCKER_INTEGRATION=1 and a locally cached Alpine image:
    PYTHONPATH=src:envs uv run pytest tests/test_core/test_client_provider_cleanup_docker.py -v

OPENENV_DOCKER_TEST_IMAGE defaults to alpine:latest. No images are pulled,
ports published, or host directories mounted. These tests verify container
ownership and cleanup, not the environment server or WebSocket protocol.
"""

import os
import subprocess
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from openenv.core.containers.runtime.providers import ContainerProvider
from openenv.core.generic_client import GenericEnvClient

pytestmark = [
    pytest.mark.docker,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENENV_DOCKER_INTEGRATION") != "1",
        reason="Set OPENENV_DOCKER_INTEGRATION=1 to run real Docker lifecycle tests",
    ),
]


def docker(*args):
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


class DockerLifecycleProvider(ContainerProvider):
    """Single-handle provider with injectable readiness and stop failures."""

    def __init__(self, image):
        self.image = image
        self.label = f"openenv-cleanup-test={uuid4().hex}"
        self.container_id = None
        self.started = []
        self.stop_calls = 0
        self.fail_readiness = False
        self.fail_stop = True

    def start_container(self):
        self.container_id = docker(
            "run",
            "--detach",
            "--pull=never",
            "--network=none",
            "--label",
            self.label,
            self.image,
            "sleep",
            "300",
        )
        self.started.append(self.container_id)
        # Every tested operation fails before opening a WebSocket.
        return "http://localhost:8000"

    def wait_for_ready(self, base_url, timeout_s=30.0):
        assert (
            docker("inspect", "--format={{.State.Running}}", self.container_id)
            == "true"
        )
        if self.fail_readiness:
            raise TimeoutError("injected readiness failure")

    def stop_container(self):
        self.stop_calls += 1
        if self.fail_stop:
            raise OSError("injected Docker cleanup failure")
        docker("stop", "--time=1", self.container_id)
        docker("rm", self.container_id)
        self.container_id = None

    def remaining_containers(self):
        return docker(
            "ps", "--all", "--quiet", "--no-trunc", "--filter", f"label={self.label}"
        ).splitlines()


@pytest.fixture
def docker_provider():
    image = os.environ.get("OPENENV_DOCKER_TEST_IMAGE", "alpine:latest")
    # Opting in with a missing daemon/image is a failure, not a silent skip.
    docker("image", "inspect", image)
    provider = DockerLifecycleProvider(image)
    try:
        yield provider
    finally:
        # Independent of the provider's single handle: remove even resources
        # orphaned by the regression, and only this fixture's unique label.
        remaining = provider.remaining_containers()
        if remaining:
            docker("rm", "--force", *remaining)
        assert provider.remaining_containers() == []


class ConstructorFailureClient(GenericEnvClient):
    def __init__(self, base_url=None, **kwargs):
        if base_url is not None:
            raise ValueError("injected constructor failure")
        super().__init__(base_url=base_url, **kwargs)


@pytest.mark.parametrize("failure", ["readiness", "constructor"])
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.asyncio
async def test_failed_cleanup_never_orphans_a_real_container(
    docker_provider, failure, sync
):
    provider = docker_provider
    provider.fail_readiness = failure == "readiness"
    client_type = (
        GenericEnvClient if provider.fail_readiness else ConstructorFailureClient
    )
    parent = client_type(provider=provider)
    client = parent.sync() if sync else parent

    async def invoke(method):
        result = getattr(client, method)()
        if not sync:
            return await result
        return result

    with patch("openenv.core.env_client.ws_connect", AsyncMock()) as connect:
        try:
            with pytest.raises((TimeoutError, ValueError), match="injected"):
                await invoke("new_session")
            original = provider.container_id
            assert provider.remaining_containers() == [original]

            with pytest.raises(Exception) as retry_error:
                await invoke("new_session")
            assert provider.started == [original]
            assert provider.stop_calls == 2
            assert provider.remaining_containers() == [original]
            assert str(retry_error.value) == "injected Docker cleanup failure"
            assert isinstance(retry_error.value, OSError)

            provider.fail_stop = False
            with pytest.raises((TimeoutError, ValueError), match="injected"):
                await invoke("new_session")
            assert len(provider.started) == 2
            assert provider.started[1] != original
            assert provider.remaining_containers() == []
            assert provider.stop_calls == 4

            await invoke("close")
            await invoke("close")
            assert provider.stop_calls == 4
            connect.assert_not_called()
        finally:
            provider.fail_stop = False
            # On an unfixed client close can only release the last handle;
            # the fixture finalizer independently catches older orphans.
            await invoke("close")
