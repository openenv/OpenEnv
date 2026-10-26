"""Docker-local resource bounds, measured against the pinned fixture image."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.resources import ResourceBoundsGrader
from openenv.validation.manifest import NetworkPolicy, ResourceDeclaration
from openenv.validation.providers import docker
from openenv.validation.providers.docker import DockerValidationProvider
from openenv.validation.runtime.contracts import LaunchSpec
from openenv.validation.types import CheckStatus

pytestmark = pytest.mark.docker

LAUNCHED = ResourceDeclaration(cpu=1, memory_mb=256, disk_mb=64, episode_timeout_s=10)


@pytest.fixture
def image():
    image = os.environ.get("OPENENV_VALIDATION_IMAGE")
    if not image:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required Docker acceptance needs OPENENV_VALIDATION_IMAGE")
        pytest.skip("Run the validation lab to supply its pinned fixture image")
    return image


def owned(run_id):
    listing = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={docker._LABEL}={run_id}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return listing.stdout.split()


def measure(image):
    spec = LaunchSpec(
        image_ref=image,
        resources=LAUNCHED,
        network=NetworkPolicy(),
        run_id="resources-" + uuid.uuid4().hex[:12],
        startup_timeout_s=60,
    )
    running = DockerValidationProvider().start(spec)
    try:
        measured = running.measure_resources()
    finally:
        running.stop()
    # The disposable qualification subject shares the run label.
    assert owned(spec.run_id) == []
    return measured


def grade(declared, measured):
    class Manifest:
        resources = declared

    subject = Subject(
        root=Path("."),
        manifest=Manifest(),
        image_ref=None,
        running=None,
        outputs_dir=Path("."),
        resource_evidence_json=json.dumps(measured),
    )
    return ResourceBoundsGrader().run(subject)


def test_good_resource_bounds(image):
    measured = measure(image)
    assert measured["cgroup"] == {
        "memory_max": 256 * 1024 * 1024,
        "swap_max": 0,
        "cpu_quota": 100000,
        "cpu_period": 100000,
        "pids_max": 256,
    }
    result = grade(LAUNCHED, measured)
    assert result.status is CheckStatus.PASS, result.evidence
    assert set(result.measured["qualification"].values()) == {"contained"}


def test_limit_above_the_declaration_is_detected(image):
    declared = LAUNCHED.model_copy(update={"memory_mb": 128, "disk_mb": 32})
    result = grade(declared, measure(image))
    assert result.status is CheckStatus.FAIL
    evidence = " ".join(result.evidence)
    assert "exceeds 128 MiB" in evidence
    assert "exceeds 32 MiB" in evidence
