"""Real lifecycle acceptance using the lab's exact-wheel, pinned-base image."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest
from openenv.validation.manifest import NetworkPolicy, ResourceDeclaration
from openenv.validation.providers import ProviderError, StartupError
from openenv.validation.providers.docker import DockerValidationProvider
from openenv.validation.runtime.contracts import LaunchSpec
from websockets.sync.client import connect


pytestmark = pytest.mark.docker


@pytest.fixture
def spec():
    image = os.environ.get("OPENENV_VALIDATION_IMAGE")
    if not image:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required Docker acceptance needs OPENENV_VALIDATION_IMAGE")
        pytest.skip("Run the validation lab to supply its pinned fixture image")
    return LaunchSpec(
        image_ref=image,
        resources=ResourceDeclaration(
            cpu=1, memory_mb=512, disk_mb=64, episode_timeout_s=10
        ),
        network=NetworkPolicy(),
        run_id="provider-" + uuid.uuid4().hex[:12],
        startup_timeout_s=30,
    )


def record_cleanup(name, run_id, details=None):
    result = subprocess.run(
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"label=org.openenv.validation.run={run_id}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert not result.stdout.strip(), "Validation leaked an owned container"
    artifacts = os.environ.get("OPENENV_VALIDATION_ARTIFACTS")
    if artifacts:
        target = Path(artifacts) / "cleanup-provider.json"
        previous = json.loads(target.read_text()) if target.exists() else []
        previous.append(
            {
                "case": name,
                "run_id": run_id,
                "remaining_container_ids": [],
                "inspection": details,
            }
        )
        target.write_text(json.dumps(previous, indent=2) + "\n")


def test_docker_lifecycle_effective_limits_and_owned_cleanup(spec):
    running = DockerValidationProvider().start(spec)
    details = None
    try:
        details = running.inspect()
        assert details["image_id"] == spec.image_ref
        assert details["running"] is True
        assert details["user"] == "65532:65532"
        assert details["limits"]["ReadonlyRootfs"] is True
        assert details["limits"]["Memory"] == 512 * 1024**2
        assert details["limits"]["MemorySwap"] == 512 * 1024**2
        assert details["limits"]["NanoCpus"] == 1_000_000_000
        assert details["limits"]["PidsLimit"] == 256
        # Engines report dropped capabilities differently (Docker: ["ALL"]; Podman:
        # each name), so the evidence is only required to be present.
        assert details["limits"]["CapDrop"]
        assert any(
            "no-new-privileges" in value for value in details["limits"]["SecurityOpt"]
        )
        result = running.exec(["python", "-c", "import os; print(os.getuid())"], 5)
        assert result.exit_code == 0 and result.stdout.strip() == "65532"
        # A non-root CapEff is zero regardless; the bounding set is what
        # `--cap-drop ALL` empties, and it is observable on every engine.
        bounding = running.exec(
            [
                "python",
                "-c",
                "print(next(line.split()[1] for line in open('/proc/self/status')"
                " if line.startswith('CapBnd:')))",
            ],
            5,
        )
        assert bounding.exit_code == 0 and int(bounding.stdout.strip(), 16) == 0
        assert (
            running.exec(
                ["python", "-c", "open('/root-write', 'w').write('no')"], 5
            ).exit_code
            != 0
        )
        assert (
            running.exec(
                ["python", "-c", "open('/tmp/probe', 'w').write('yes')"], 5
            ).exit_code
            == 0
        )
        with connect(
            running.base_url.replace("http://", "ws://") + "/ws",
            proxy=None,
            open_timeout=5,
            close_timeout=1,
            max_size=65536,
        ) as socket:
            socket.send(
                json.dumps(
                    {"type": "reset", "data": {"episode_id": "integration", "seed": 7}}
                )
            )
            reset = json.loads(socket.recv(timeout=5))
            assert reset["type"] == "observation"
            assert reset["data"]["observation"]["counter"] == 0
            socket.send(json.dumps({"type": "step", "data": {"increment": 1}}))
            step = json.loads(socket.recv(timeout=5))
            assert step["type"] == "observation"
            assert step["data"]["observation"]["counter"] == 1
            socket.send(json.dumps({"type": "state"}))
            state = json.loads(socket.recv(timeout=5))
            assert state["type"] == "state"
            assert state["data"]["episode_id"] == "integration"
            assert state["data"]["step_count"] == 1
            socket.send(json.dumps({"type": "close"}))
        artifacts = os.environ.get("OPENENV_VALIDATION_ARTIFACTS")
        if artifacts:
            log = Path(artifacts) / "logs/provider-lifecycle.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(running.logs(max_bytes=16384))
    finally:
        running.stop()
    running.stop()
    record_cleanup("lifecycle", spec.run_id, details)


def test_docker_unhealthy_startup_removes_container(spec):
    failed = spec.model_copy(
        update={
            "env_vars": {"VALIDATION_FAULT": "startup_failure"},
            "startup_timeout_s": 2,
        }
    )
    with pytest.raises(StartupError, match="deadline"):
        DockerValidationProvider().start(failed)
    record_cleanup("startup_failure", spec.run_id)


def test_docker_exec_timeout_removes_process_tree(spec):
    running = DockerValidationProvider().start(spec)
    try:
        with pytest.raises(ProviderError, match="deadline"):
            running.exec(
                [
                    "python",
                    "-c",
                    "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)",
                ],
                timeout_s=0.2,
            )
    finally:
        running.stop()
    record_cleanup("exec_timeout", spec.run_id)
