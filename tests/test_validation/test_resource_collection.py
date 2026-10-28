"""Resource measurement retains fatal cleanup provenance after successful recovery."""

import json
import shutil

from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.providers import docker
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence
from test_host_containment import contained


def test_recovered_exec_cleanup_error_remains_fatal(tmp_path, monkeypatch):
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
            failed_names.add(argv[3])
            raise docker._OperationTimeout("openenv-probe-descendant:" + nonce + "\n")
        return 0, "openenv-probe-start:" + nonce + "\ncontained\n", ""

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
        assert suffix == "counters"
        n = counters.get(subject.name, 0)
        counters[subject.name] = n + 1
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
    assert failed_names and all(removal_attempts[name] == 2 for name in failed_names)
    assert not alive
    assert results["runtime.startup"].status is CheckStatus.ERROR
    assert results["runtime.resource_bounds"].status is CheckStatus.SKIP
    assert cleanup["completed"] is True
    assert any(
        "cleanup" in item.lower() or "teardown" in item.lower()
        for result in report.results
        for item in result.evidence
    )
