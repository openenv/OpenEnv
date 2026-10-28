"""Docker-local host containment, measured against the pinned fixture image."""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.containment import HostContainmentGrader
from openenv.validation.manifest import NetworkPolicy, ResourceDeclaration
from openenv.validation.providers import docker
from openenv.validation.providers.docker import DockerValidationProvider
from openenv.validation.runtime.contracts import LaunchSpec
from openenv.validation.types import CheckStatus

pytestmark = pytest.mark.docker


@pytest.fixture
def image():
    image = os.environ.get("OPENENV_VALIDATION_IMAGE")
    if not image:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required Docker acceptance needs OPENENV_VALIDATION_IMAGE")
        pytest.skip("Run the validation lab to supply its pinned fixture image")
    return image


def launch(image, mode="public"):
    return LaunchSpec(
        image_ref=image,
        resources=ResourceDeclaration(
            cpu=1, memory_mb=512, disk_mb=64, episode_timeout_s=10
        ),
        network=NetworkPolicy(mode=mode),
        run_id="containment-" + uuid.uuid4().hex[:12],
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


def grade(measured):
    subject = Subject(
        root=Path("."),
        manifest=None,
        image_ref=None,
        running=None,
        outputs_dir=Path("."),
        containment_evidence_json=json.dumps(measured),
    )
    return HostContainmentGrader().run(subject)


@pytest.mark.parametrize("mode", ["public", "no-network"])
def test_good_host_containment(image, mode):
    spec = launch(image, mode)
    running = DockerValidationProvider().start(spec)
    try:
        measured = running.measure_containment()
    finally:
        running.stop()
    assert owned(spec.run_id) == []
    result = grade(measured)
    assert result.status is CheckStatus.PASS, result.evidence
    assert sorted(measured["allotted_writable"]) == ["/dev/shm", "/tmp"]
    assert {p["name"] for p in measured["processes"]} >= {"docker-init"}
    mounts = measured["mounts"]
    assert [m["read_only"] for m in mounts if m["mount_point"] == "/"] == [True]
    # The allotted roots together hold the declared writable-storage budget.
    sizes = {m["mount_point"]: m["size_bytes"] for m in mounts if not m["read_only"]}
    assert sizes["/tmp"] + sizes["/dev/shm"] == 64 * 1024 * 1024


def test_writable_image_root_is_detected(image, monkeypatch):
    # Fault injection: the effective launch loses --read-only.
    original = docker._command

    def without_read_only(argv, *args, **kwargs):
        if argv[:2] == ["docker", "create"]:
            argv = [arg for arg in argv if arg != "--read-only"]
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(docker, "_command", without_read_only)
    spec = launch(image)
    running = DockerValidationProvider().start(spec)
    try:
        # A world-writable directory below root-owned top-level directories.
        subprocess.run(
            ["docker", "exec", "--user", "0", running.name, "sh", "-c"]
            + ["mkdir -p /opt/nested/data && chmod 777 /opt/nested/data"],
            capture_output=True,
            timeout=30,
            check=True,
        )
        escape = running.exec(
            [
                "python",
                "-c",
                "with open('/opt/nested/data/escape', 'wb') as f:\n"
                " for _ in range(65): f.write(bytes(2**20))",
            ],
            30,
        )
        measured = running.measure_containment()
    finally:
        running.stop()
    assert owned(spec.run_id) == []
    assert escape.exit_code == 0, escape.stderr
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "the image root is writable" in result.evidence


@pytest.mark.parametrize("storage", ["directory", "file"])
def test_nested_device_storage_is_detected(image, storage):
    spec = launch(image)
    running = DockerValidationProvider().start(spec)
    path = "/dev/nested/data"
    setup = "mkdir -p /dev/nested && "
    setup += (
        f"mkdir {path} && chmod 777 {path}"
        if storage == "directory"
        else f"touch {path} && chmod 666 {path}"
    )
    target = path + "/escape" if storage == "directory" else path
    try:
        subprocess.run(
            ["docker", "exec", "--user", "0", running.name, "sh", "-c", setup],
            capture_output=True,
            timeout=30,
            check=True,
        )
        escape = running.exec(
            ["python", "-c", f"open({target!r}, 'wb').write(bytes(2**20))"], 30
        )
        measured = running.measure_containment()
    finally:
        running.stop()
    assert owned(spec.run_id) == []
    assert escape.exit_code == 0, escape.stderr
    result = grade(measured)
    assert result.status is CheckStatus.FAIL, result.evidence
    assert any(path in line for line in result.evidence)


def test_root_process_in_the_subject_is_detected(image):
    spec = launch(image)
    running = DockerValidationProvider().start(spec)
    intruder = subprocess.Popen(
        ["docker", "exec", "--user", "0", running.name, "sleep", "60"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            measured = running.measure_containment()
            if any(0 in p["uids"] for p in measured["processes"]):
                break
            time.sleep(0.5)
    finally:
        intruder.kill()
        running.stop()
    assert owned(spec.run_id) == []
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert any("runs as root" in line for line in result.evidence)
