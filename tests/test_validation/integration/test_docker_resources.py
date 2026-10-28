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


def test_good_resource_bounds(image, monkeypatch):
    created = []
    original = DockerValidationProvider.start

    def start(self, spec):
        running = original(self, spec)
        created.append(running.name)
        return running

    monkeypatch.setattr(DockerValidationProvider, "start", start)
    measured = measure(image)
    assert len(created) == len(set(created)) == 5
    assert measured["cgroup"] == {
        "memory_max": 256 * 1024 * 1024,
        "swap_max": 0,
        "cpu_quota": 100000,
        "cpu_period": 100000,
        "pids_max": 256,
        "unavailable": [],
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


@pytest.mark.parametrize("emit_start", [False, True])
def test_exit_137_without_enforcement_evidence_is_inconclusive(
    image, monkeypatch, emit_start
):
    script = "import sys\n"
    if emit_start:
        script += "print('openenv-probe-start:' + sys.argv[3], flush=True)\n"
    script += "print('contained', flush=True)\nsys.exit(137)"
    monkeypatch.setattr(docker, "QUALIFY_SCRIPT", script)
    measured = measure(image)
    assert {probe["result"] for probe in measured["qualification"]} == {"inconclusive"}
    assert grade(LAUNCHED, measured).status is CheckStatus.SKIP


def test_writable_root_is_not_accounted_as_only_tmpfs(image, monkeypatch):
    original = docker._command

    def writable_root(argv, *args, **kwargs):
        if argv[:2] == ["docker", "create"]:
            argv = [arg for arg in argv if arg != "--read-only"]
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(docker, "_command", writable_root)
    result = grade(LAUNCHED, measure(image))
    assert result.status is CheckStatus.FAIL
    assert "the image root is writable" in result.evidence


@pytest.mark.parametrize("storage", ["directory", "file"])
def test_nested_device_storage_is_accounted(image, storage):
    spec = LaunchSpec(
        image_ref=image,
        resources=LAUNCHED,
        network=NetworkPolicy(),
        run_id="resources-dev-" + uuid.uuid4().hex[:12],
        startup_timeout_s=60,
    )
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
        assert escape.exit_code == 0, escape.stderr
        measured = running.measure_resources()
    finally:
        running.stop()
    assert owned(spec.run_id) == []
    assert "/dev" not in measured["containment"]["writable"]
    result = grade(LAUNCHED, measured)
    assert result.status is CheckStatus.FAIL, result.evidence
    assert result.measured["writable_bytes"] == 128 * 1024 * 1024


@pytest.mark.parametrize("failed_resource", ["subject", "helper", "startup"])
def test_disposable_cleanup_failure_is_reported_and_retried(
    image, monkeypatch, tmp_path, failed_resource
):
    import shutil

    import yaml
    from conftest import FIXTURES
    from openenv.validation import runner
    from openenv.validation.types import Level, ProviderCapability
    from support.runtime import evidence

    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    manifest = yaml.safe_load((package / "openenv.yaml").read_text())
    manifest["validation"]["network"] = {"mode": "no-network"}
    (package / "openenv.yaml").write_text(yaml.safe_dump(manifest))

    class Provider(DockerValidationProvider):
        capabilities = frozenset(
            {
                ProviderCapability.IMAGE_BUILD,
                ProviderCapability.EXEC,
                ProviderCapability.RESOURCE_ENFORCEMENT,
            }
        )

        def build(self, *args):
            return image

    created = []
    removals = []
    original_start = DockerValidationProvider.start
    original_command = docker._command
    original_helper_exec = docker.DockerRunningSubject._helper_exec

    def start(self, spec):
        subject = original_start(self, spec)
        created.append(subject)
        return subject

    def command(argv, *args, **kwargs):
        if created and created[0]._disposables and argv[:2] == ["docker", "rm"]:
            disposable = created[0]._disposables[0]
            target = (
                disposable.name
                if failed_resource in {"subject", "startup"}
                else disposable._auxiliary[0]
            )
            if argv[-1] == target:
                removals.append(target)
                return 1, "", "injected removal failure"
        return original_command(argv, *args, **kwargs)

    def helper_exec(self, script, *args):
        if (
            failed_resource == "startup"
            and created
            and self is not created[0]
            and script == docker.WAIT_READY_SCRIPT
        ):
            return 1, "", "injected readiness failure"
        return original_helper_exec(self, script, *args)

    monkeypatch.setattr(docker.DockerRunningSubject, "_helper_exec", helper_exec)
    monkeypatch.setattr(DockerValidationProvider, "start", start)
    monkeypatch.setattr(docker, "_command", command)
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())
    try:
        report = runner.run_validation(
            package,
            max_level=Level.RUNTIME,
            provider=Provider(),
            artifacts_dir=tmp_path / "artifacts",
        )
        cleanup = json.loads((tmp_path / "artifacts" / "cleanup.json").read_text())
        assert cleanup["completed"] is False
        startup = next(
            result for result in report.results if result.check_id == "runtime.startup"
        )
        assert startup.status is CheckStatus.ERROR
        assert len(removals) >= 2
        assert owned(created[0].run_id)
    finally:
        monkeypatch.setattr(docker, "_command", original_command)
        for subject in reversed(created):
            subject.stop()
    assert owned(created[0].run_id) == []


def test_recovered_exec_removal_failure_remains_fatal(image, tmp_path, monkeypatch):
    import shutil

    import yaml
    from conftest import FIXTURES
    from openenv.validation import runner
    from openenv.validation.types import Level, ProviderCapability
    from support.runtime import evidence

    subjects = []
    failed = []
    original_start = docker.DockerValidationProvider.start
    original_command = docker._command

    class Provider(docker.DockerValidationProvider):
        capabilities = frozenset(
            {
                ProviderCapability.IMAGE_BUILD,
                ProviderCapability.EXEC,
                ProviderCapability.RESOURCE_ENFORCEMENT,
            }
        )

        def build(self, *args):
            return image

    def start(self, spec):
        subject = original_start(self, spec)
        subjects.append(subject)
        return subject

    def command(argv, *args, **kwargs):
        if argv[:2] == ["docker", "rm"] and subjects and not failed:
            disposables = subjects[0]._disposables
            if len(disposables) == 4 and argv[-1] == disposables[-1].name:
                failed.append(argv[-1])
                return 1, "", "injected first exec-timeout removal failure"
        return original_command(argv, *args, **kwargs)

    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    path = package / "openenv.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["validation"]["resources"].update(memory_mb=256, disk_mb=64)
    path.write_text(yaml.safe_dump(manifest))
    monkeypatch.setattr(docker.DockerValidationProvider, "start", start)
    monkeypatch.setattr(docker, "_command", command)
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())
    try:
        report = runner.run_validation(
            package,
            max_level=Level.RUNTIME,
            provider=Provider(),
            artifacts_dir=tmp_path / "artifacts",
        )
        cleanup = json.loads((tmp_path / "artifacts" / "cleanup.json").read_text())
        results = {result.check_id: result for result in report.results}
        listing = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={docker._LABEL}={subjects[0].run_id}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert failed
        assert listing.stdout.split() == []
        assert results["runtime.startup"].status is CheckStatus.ERROR
        assert results["runtime.resource_bounds"].status is CheckStatus.SKIP
        assert cleanup["completed"] is True
        assert any(
            "cleanup" in item.lower() or "teardown" in item.lower()
            for result in report.results
            for item in result.evidence
        )
    finally:
        monkeypatch.setattr(docker, "_command", original_command)
        for subject in reversed(subjects):
            subject.stop()
