"""Resource collection preserves trusted evidence and fatal cleanup provenance."""

import json
import shutil

import pytest
from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.manifest import NetworkPolicy, ResourceDeclaration
from openenv.validation.providers import CleanupError, docker, ProviderError
from openenv.validation.runtime.contracts import LaunchSpec
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence
from test_host_containment import contained
from test_resource_bounds import bounded, grade


@pytest.mark.parametrize("fault", ["cleanup", "escaped_text", "counterfeit_descendant"])
def test_resource_collection_rejects_untrusted_outcomes(tmp_path, monkeypatch, fault):
    alive = set()
    subjects = []
    counters = {}
    removal_attempts = {}
    failed_names = set()
    image = "sha256:" + "a" * 64

    def start(provider, spec):
        name = "review-subject-" + str(len(subjects))
        subject = docker.DockerRunningSubject(name, spec)
        subject.base_url = "http://127.0.0.1:8000"
        if provider._cleanup_subjects is not None:
            provider._cleanup_subjects.append(subject)
        subjects.append(subject)
        alive.add(name)
        return subject

    def command(argv, *args, **kwargs):
        if argv[1] == "rm":
            name = argv[-1]
            removal_attempts[name] = removal_attempts.get(name, 0) + 1
            if name in failed_names and removal_attempts[name] == 1:
                return 1, "", "injected one-shot disposable removal failure"
            alive.discard(name)
            return 0, "", ""
        assert argv[:3] == ["docker", "exec", "--"], argv
        if argv[-1] == "pass":
            return 0, "", ""
        kind, nonce = argv[-5], argv[-3]
        if kind == "exec_timeout":
            if fault == "cleanup":
                failed_names.add(argv[3])
            raise docker._OperationTimeout("openenv-probe-descendant:" + nonce + "\n")
        outcome = "escaped" if fault == "escaped_text" else "contained"
        return 0, "openenv-probe-start:" + nonce + "\n" + outcome + "\n", ""

    def pidns(subject, suffix, *args):
        if suffix == "limits":
            return json.dumps(
                {
                    "cgroup": {
                        "memory_max": 1024 * 2**20,
                        "swap_max": 0,
                        "cpu_quota": 100000,
                        "cpu_period": 100000,
                        "pids_max": 256,
                    }
                }
            )
        if suffix == "descendants":
            return "{}"
        assert suffix == "counters"
        n = counters.get(subject.name, 0)
        counters[subject.name] = n + (fault != "escaped_text")
        return json.dumps(
            {
                "oom_kill": n,
                "pids_max": n,
                "free_bytes": {
                    "/tmp": 0 if n else 32 * 2**20,
                    "/dev/shm": 0 if n else 32 * 2**20,
                },
            }
        )

    monkeypatch.setattr(docker, "_ensure_image", lambda *a: None)
    monkeypatch.setattr(docker, "_command", command)
    monkeypatch.setattr(docker.DockerValidationProvider, "start", start)
    monkeypatch.setattr(
        docker.DockerRunningSubject,
        "_owner",
        lambda subject, name=None: (
            subject.run_id if (name or subject.name) in alive else None
        ),
    )
    monkeypatch.setattr(
        docker.DockerRunningSubject,
        "_raw_inspect",
        lambda *a: {"Config": {"User": "65532:65532"}, "State": {"Running": True}},
    )
    monkeypatch.setattr(docker.DockerRunningSubject, "_pidns_probe", pidns)
    monkeypatch.setattr(
        docker.DockerRunningSubject, "measure_containment", lambda *a: contained()
    )
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())

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

    package = tmp_path / "subject"
    fixture = FIXTURES / "runtime" / "served_probe"
    shutil.copytree(fixture, package)
    report = runner.run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=Provider(),
        artifacts_dir=tmp_path / "artifacts",
    )
    results = {result.check_id: result for result in report.results}
    cleanup = json.loads((tmp_path / "artifacts" / "cleanup.json").read_text())
    if fault == "cleanup":
        assert failed_names and all(
            removal_attempts[name] == 2 for name in failed_names
        )
        assert results["runtime.startup"].status is CheckStatus.ERROR
        assert any(
            "cleanup" in item.lower() or "teardown" in item.lower()
            for result in report.results
            for item in result.evidence
        )
    else:
        assert results["runtime.startup"].status is CheckStatus.PASS
    assert not alive
    assert results["runtime.resource_bounds"].status is CheckStatus.SKIP
    assert cleanup["completed"] is True


@pytest.mark.parametrize("memory_mb", [1024, 2048])
def test_storage_failure_preserves_collected_limits(monkeypatch, memory_mb):
    spec = LaunchSpec(
        image_ref="sha256:" + "a" * 64,
        resources=ResourceDeclaration(
            cpu=1, memory_mb=1024, disk_mb=64, episode_timeout_s=10
        ),
        network=NetworkPolicy(),
        run_id="test-resource-limits",
    )
    subject = docker.DockerRunningSubject("test-resource-limits", spec)
    monkeypatch.setattr(docker, "_ensure_image", lambda *a: None)
    monkeypatch.setattr(
        subject, "_raw_inspect", lambda *a: {"Config": {"User": "65532:65532"}}
    )
    limits = bounded(memory_max=memory_mb * 2**20)["cgroup"]
    monkeypatch.setattr(
        subject, "_pidns_probe", lambda *a: json.dumps({"cgroup": limits})
    )

    def unavailable(*args, **kwargs):
        raise ProviderError("independent storage probe failed")

    monkeypatch.setattr(subject, "measure_containment", unavailable)
    monkeypatch.setattr(docker.DockerValidationProvider, "start", unavailable)
    measured = subject.measure_resources()
    assert measured["cgroup"]["memory_max"] == memory_mb * 2**20
    assert measured["containment"] is None
    result = grade(measured)
    assert result.status is (CheckStatus.FAIL if memory_mb > 1024 else CheckStatus.SKIP)

    def cleanup_failed(*args):
        raise CleanupError("storage helper teardown failed")

    monkeypatch.setattr(subject, "measure_containment", cleanup_failed)
    with pytest.raises(CleanupError, match="storage helper teardown"):
        subject.measure_resources()


@pytest.mark.parametrize(
    "observation", ["fork", "none", "malformed", "unavailable", "late"]
)
def test_exec_timeout_requires_a_witness_before_timeout(monkeypatch, observation):
    import threading

    spec = LaunchSpec(
        image_ref="sha256:" + "a" * 64,
        resources=ResourceDeclaration(
            cpu=1, memory_mb=256, disk_mb=64, episode_timeout_s=10
        ),
        network=NetworkPolicy(),
        run_id="test-witness",
    )
    subject = docker.DockerRunningSubject("test-witness", spec)
    stopped = []
    release = threading.Event()

    def command(*args):
        assert release.wait(1)
        raise docker._OperationTimeout("openenv-probe-descendant:fresh-nonce\n")

    def observe(*args):
        if observation == "late":
            release.set()
            # Wait for the worker to report its timeout before returning a witness.
            import time

            time.sleep(0.02)
        else:
            threading.Timer(0.02, release.set).start()
        if observation == "unavailable":
            raise ProviderError("observer unavailable")
        if observation == "malformed":
            return "invalid json"
        return json.dumps(
            {"parent_pid": 100, "descendant_pid": 101} if observation != "none" else {}
        )

    monkeypatch.setattr(docker, "_command", command)
    monkeypatch.setattr(subject, "_pidns_probe", observe)
    monkeypatch.setattr(subject, "stop", lambda: stopped.append(True))
    measured = subject._measure_exec_timeout(
        ["python", "-c", "stimulus"], "fresh-nonce", "65532", 10
    )
    assert stopped == [True]
    assert measured["removed"] is True
    assert measured.get("result") == ("contained" if observation == "fork" else None)
