"""Docker-local network enforcement against the pinned fixture image."""

import json
import os
import subprocess
import urllib.request
import uuid
from pathlib import Path

import pytest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.network import NetworkPolicyGrader
from openenv.validation.manifest import NetworkPolicy, ResourceDeclaration
from openenv.validation.providers import _netns, docker
from openenv.validation.providers.docker import DockerValidationProvider
from openenv.validation.runtime.contracts import LaunchSpec
from openenv.validation.types import CheckStatus
from websockets.sync.client import connect

pytestmark = pytest.mark.docker


@pytest.fixture
def image():
    image = os.environ.get("OPENENV_VALIDATION_IMAGE")
    if not image:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required Docker acceptance needs OPENENV_VALIDATION_IMAGE")
        pytest.skip("Run the validation lab to supply its pinned fixture image")
    return image


def launch(image, mode):
    return LaunchSpec(
        image_ref=image,
        resources=ResourceDeclaration(
            cpu=1, memory_mb=512, disk_mb=64, episode_timeout_s=10
        ),
        network=NetworkPolicy(mode=mode),
        run_id="network-" + uuid.uuid4().hex[:12],
        startup_timeout_s=60,
    )


def owned(run_id):
    listing = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={docker._LABEL}={run_id}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return listing.stdout.split()


class _Manifest:
    def __init__(self, mode):
        self.network = NetworkPolicy(mode=mode)


def grade(mode, measured):
    subject = Subject(
        root=Path("."),
        manifest=_Manifest(mode),
        image_ref=None,
        running=None,
        outputs_dir=Path("."),
        network_evidence_json=json.dumps(measured),
    )
    return NetworkPolicyGrader().run(subject)


def test_no_network_subject_is_served_only_through_the_helper(image):
    spec = launch(image, "no-network")
    running = DockerValidationProvider().start(spec)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(running.base_url + "/health", timeout=20) as response:
            assert response.status == 200
        ws_url = running.base_url.replace("http://", "ws://") + "/ws"
        with connect(ws_url, proxy=None, open_timeout=20, close_timeout=2) as socket:
            socket.send(json.dumps({"type": "reset", "data": {}}))
            assert json.loads(socket.recv(timeout=20))["type"] == "observation"
        assert running.inspect()["limits"]["NetworkMode"] == "none"
    finally:
        running.stop()
    assert owned(spec.run_id) == []


@pytest.mark.parametrize("mode", ["public", "no-network"])
def test_good_network_policy(image, mode):
    spec = launch(image, mode)
    running = DockerValidationProvider().start(spec)
    try:
        measured = running.measure_network()
    finally:
        running.stop()
    assert owned(spec.run_id) == []
    result = grade(mode, measured)
    assert result.status is CheckStatus.PASS, result.evidence
    assert "tcp" in result.measured["counted_probes"]


class _LeakyProvider(DockerValidationProvider):
    """Simulates broken enforcement: launches on the bridge but reports no-network."""

    def start(self, spec):
        running = super().start(spec.model_copy(update={"network": NetworkPolicy()}))
        running._mode = "no-network"
        return running


def test_network_leak_is_detected_from_measurement(image):
    spec = launch(image, "no-network")
    running = _LeakyProvider().start(spec)
    try:
        measured = running.measure_network()
    finally:
        running.stop()
    assert owned(spec.run_id) == []
    result = grade("no-network", measured)
    assert result.status is CheckStatus.FAIL
    assert any("reachable" in line for line in result.evidence)


def test_helper_cannot_bridge_subject_egress(image):
    spec = launch(image, "no-network")
    running = DockerValidationProvider().start(spec)
    sink = f"network-test-sink-{uuid.uuid4().hex[:8]}"
    try:
        measured = running.measure_network()
        # The helper shares the namespace but adds no listening socket of its own.
        assert set(measured["namespace"]["listening_ports"]) == {8000}
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                sink,
                "--network",
                "bridge",
                "--label",
                f"{docker._LABEL}={spec.run_id}",
                *_netns.HARDENING,
                _netns.HELPER_IMAGE,
                "python3",
                "-c",
                _netns.SINK_SCRIPT,
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        inspected = subprocess.run(
            ["docker", "inspect", sink],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        sink_ip = docker._container_ip(inspected.stdout)
        attempt = running.exec(
            [
                "python",
                "-c",
                "import socket, sys\n"
                "try:\n"
                f"    socket.create_connection(({sink_ip!r}, {_netns.SINK_TCP_PORT}), timeout=5)\n"
                "    print('reachable')\n"
                "except OSError as exc:\n"
                "    print(exc.errno)",
            ],
            30,
        )
        assert attempt.stdout.strip() == "101", attempt
    finally:
        subprocess.run(["docker", "rm", "-f", sink], capture_output=True, timeout=60)
        running.stop()
    assert owned(spec.run_id) == []
