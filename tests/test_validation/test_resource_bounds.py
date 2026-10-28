"""`runtime.resource_bounds` judges measured limits, and the runner supplies them."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.resources import ResourceBoundsGrader
from openenv.validation.manifest import ResourceDeclaration
from openenv.validation.providers import ProviderError
from openenv.validation.runner import run_validation
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence, FakeRunningSubject, FakeRuntimeProvider
from test_host_containment import contained

MIB = 1024 * 1024


def bounded(**cgroup):
    """Evidence for a subject declared with 1 CPU, 1024 MiB memory and 64 MiB disk."""
    limits = {
        "memory_max": 1024 * MIB,
        "swap_max": 0,
        "cpu_quota": 100000,
        "cpu_period": 100000,
        "pids_max": 256,
    }
    limits.update(cgroup)
    return {
        "schema_version": 1,
        "cgroup": limits,
        "containment": contained(),
        "writable": [
            {"path": "/tmp", "size_bytes": 32 * MIB},
            {"path": "/dev/shm", "size_bytes": 32 * MIB},
        ],
        "qualification": [
            {
                "kind": kind,
                "result": "contained",
                "started": True,
                "reason": reason,
                "counter_before": 0,
                "counter_after": 1,
                "removed": True,
                "free_bytes_before": {"/tmp": 32 * MIB, "/dev/shm": 32 * MIB},
                "free_bytes": {"/tmp": 0, "/dev/shm": 0},
            }
            for kind, reason in [
                ("memory", "cgroup_oom_kill"),
                ("pids", "cgroup_pids_max"),
                ("disk", "filesystem_full"),
                ("exec_timeout", "exec_timeout"),
            ]
        ],
    }


class _Manifest:
    resources = ResourceDeclaration(
        cpu=1.0, memory_mb=1024, disk_mb=64, episode_timeout_s=30
    )


def grade(measured):
    subject = Subject(
        root=Path("."),
        manifest=_Manifest(),
        image_ref="sha256:" + "a" * 64,
        running=None,
        outputs_dir=Path("."),
        resource_evidence_json=None if measured is None else json.dumps(measured),
    )
    return ResourceBoundsGrader().run(subject)


def test_enforced_budget_passes():
    result = grade(bounded())
    assert result.status is CheckStatus.PASS, result.evidence
    assert result.measured["writable_bytes"] == 64 * MIB


@pytest.mark.parametrize(
    "cgroup, expected",
    [
        ({"memory_max": None}, "memory is unlimited"),
        ({"memory_max": 2048 * MIB}, "exceeds 1024 MiB"),
        ({"swap_max": None}, "swap is not disabled"),
        ({"swap_max": 512 * MIB}, "swap is not disabled"),
        ({"cpu_quota": None}, "CPU is unlimited"),
        ({"cpu_quota": 200000}, "CPU limit 2 exceeds 1"),
        ({"pids_max": None}, "process count is unlimited"),
    ],
)
def test_missing_limit_fails(cgroup, expected):
    result = grade(bounded(**cgroup))
    assert result.status is CheckStatus.FAIL
    assert expected in " ".join(result.evidence)
    assert result.remediation


def test_writable_storage_over_budget_fails():
    measured = bounded()
    measured["containment"]["mounts"][-1]["size_bytes"] = 64 * MIB
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "exceeds 64 MiB" in " ".join(result.evidence)


def test_escaped_probe_fails_even_with_correct_settings():
    measured = bounded()
    measured["qualification"][3]["result"] = "escaped"
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "exec_timeout probe escaped" in " ".join(result.evidence)


@pytest.mark.parametrize("result", ["inconclusive", None])
def test_unproven_enforcement_is_a_skip(result):
    measured = bounded()
    if result is None:
        measured["qualification"].pop(0)
    else:
        measured["qualification"][0]["result"] = result
    outcome = grade(measured)
    assert outcome.status is CheckStatus.SKIP
    assert "not proven for memory" in " ".join(outcome.evidence)


def test_unreadable_cgroup_is_a_skip_unless_a_violation_is_proven():
    measured = bounded()
    measured["cgroup"] = None
    assert grade(measured).status is CheckStatus.SKIP
    measured["qualification"][0]["result"] = "escaped"
    assert grade(measured).status is CheckStatus.FAIL


def test_missing_or_malformed_evidence():
    assert grade(None).status is CheckStatus.SKIP
    assert grade({"schema_version": 1}).status is CheckStatus.FAIL


@dataclass
class MeasuredSubject(FakeRunningSubject):
    measurement: object = None
    measured: int = 0

    def measure_resources(self, timeout_s: float = 180) -> dict:
        self.measured += 1
        if isinstance(self.measurement, Exception):
            raise self.measurement
        return self.measurement


ENFORCING = frozenset(
    {
        ProviderCapability.IMAGE_BUILD,
        ProviderCapability.EXEC,
        ProviderCapability.RESOURCE_ENFORCEMENT,
    }
)


def validate(tmp_path, monkeypatch, subject, capabilities=ENFORCING):
    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())
    provider = FakeRuntimeProvider(capabilities=capabilities, subject=subject)
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    return {r.check_id: r for r in report.results}["runtime.resource_bounds"]


def test_runner_grades_provider_measurement_against_the_manifest(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=bounded())
    result = validate(tmp_path, monkeypatch, subject)
    assert subject.measured == 1
    assert result.status is CheckStatus.PASS, result.evidence


def test_failed_measurement_leaves_the_check_unproven(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=ProviderError("The limits probe failed"))
    result = validate(tmp_path, monkeypatch, subject)
    assert result.status is CheckStatus.SKIP
    assert "unavailable" in " ".join(result.evidence)


def test_provider_without_enforcement_is_not_asked(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=bounded())
    result = validate(
        tmp_path,
        monkeypatch,
        subject,
        frozenset({ProviderCapability.IMAGE_BUILD, ProviderCapability.EXEC}),
    )
    assert subject.measured == 0
    assert result.status is CheckStatus.SKIP
    assert "resource_enforcement" in " ".join(result.evidence)


def test_writable_image_root_cannot_pass_resources():
    measured = bounded()
    measured["containment"] = contained()
    measured["containment"]["mounts"][0]["read_only"] = False
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "image root is writable" in " ".join(result.evidence)


def test_missing_storage_inventory_is_unproven():
    measured = bounded()
    measured.pop("containment")
    result = grade(measured)
    assert result.status is CheckStatus.SKIP
    assert "inventory" in " ".join(result.evidence)


def test_unreadable_limit_preserves_a_known_violation():
    measured = bounded(memory_max=2048 * MIB)
    measured["cgroup"]["unavailable"] = ["swap_max"]
    measured["cgroup"]["swap_max"] = None
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "exceeds 1024 MiB" in " ".join(result.evidence)
    assert "swap is not disabled" not in result.evidence


@pytest.mark.parametrize("kind", ["memory", "pids", "disk", "exec_timeout"])
def test_claimed_enforcement_without_start_or_cause_is_unproven(kind):
    measured = bounded()
    probe = next(probe for probe in measured["qualification"] if probe["kind"] == kind)
    probe["started"] = False
    assert grade(measured).status is CheckStatus.SKIP
    probe["started"] = True
    probe["reason"] = None
    assert grade(measured).status is CheckStatus.SKIP


def test_unknown_storage_capacity_and_partial_cgroup_are_unproven():
    measured = bounded()
    measured["containment"]["mounts"][-1]["size_bytes"] = None
    assert grade(measured).status is CheckStatus.SKIP
    measured = bounded(swap_max=None, unavailable=["swap_max"])
    assert grade(measured).status is CheckStatus.SKIP


def test_shared_filesystem_is_counted_once():
    measured = bounded()
    duplicate = dict(measured["containment"]["mounts"][-1], mount_point="/dev/cache")
    measured["containment"]["mounts"].append(duplicate)
    result = grade(measured)
    assert result.status is CheckStatus.PASS, result.evidence
    assert result.measured["writable_bytes"] == 64 * MIB


@pytest.mark.parametrize("path", ["/dev/nested/data", "/dev/existing-file"])
def test_device_storage_counts_capacity_even_when_mount_root_is_not_writable(path):
    measured = bounded()
    measured["containment"]["writable"] = [path]
    measured["containment"]["device_storage"]["0:180"]["writable"] = True
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    # /dev and its masked /proc aliases share one 64 MiB filesystem.
    assert result.measured["writable_bytes"] == 128 * MIB


@pytest.mark.parametrize(
    "inventory", [{}, {"0:180": {"complete": False, "writable": False}}]
)
def test_incomplete_device_storage_cannot_pass(inventory):
    measured = bounded()
    measured["containment"]["device_storage"] = inventory
    assert grade(measured).status is CheckStatus.SKIP


def test_partial_device_storage_retains_a_known_budget_violation():
    measured = bounded()
    measured["containment"]["device_storage"]["0:180"] = {
        "complete": False,
        "writable": True,
    }
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert result.measured["writable_bytes"] == 128 * MIB


def test_allotted_mount_on_device_filesystem_is_always_accounted():
    measured = bounded()
    shm = next(
        m for m in measured["containment"]["mounts"] if m["mount_point"] == "/dev/shm"
    )
    shm.update(device="0:180", size_bytes=64 * MIB)
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert result.measured["writable_bytes"] == 96 * MIB


def test_usable_device_storage_with_unknown_capacity_is_inconclusive():
    measured = bounded()
    measured["containment"]["device_storage"]["0:180"]["writable"] = True
    for mount in measured["containment"]["mounts"]:
        if mount["device"] == "0:180":
            mount["size_bytes"] = None
    assert grade(measured).status is CheckStatus.SKIP


def test_incomplete_usable_device_inventory_within_budget_is_inconclusive():
    measured = bounded()
    measured["containment"]["device_storage"]["0:180"] = {
        "complete": False,
        "writable": True,
    }
    for mount in measured["containment"]["mounts"]:
        if mount["device"] == "0:180":
            mount["size_bytes"] = MIB
        elif mount["mount_point"] in measured["containment"]["allotted"]:
            mount["size_bytes"] = MIB
    assert grade(measured).status is CheckStatus.SKIP


@pytest.mark.parametrize("cpu", ["100000 100000", "100000 0", "invalid", ""])
def test_limits_script_retains_independently_readable_values(monkeypatch, capsys, cpu):
    import io

    from openenv.validation.providers._pidns import LIMITS_SCRIPT

    values = {
        "memory.max": str(2048 * MIB),
        "cpu.max": cpu,
        "pids.max": "256",
    }

    def read(path):
        name = path.rsplit("/", 1)[-1]
        if name not in values:
            raise PermissionError(name)
        return io.StringIO(values[name])

    monkeypatch.setattr("builtins.open", read)
    exec(LIMITS_SCRIPT, {})
    measured = json.loads(capsys.readouterr().out)["cgroup"]
    assert measured["memory_max"] == 2048 * MIB
    assert measured["pids_max"] == 256
    expected = (
        ["swap_max"]
        if cpu == "100000 100000"
        else ["swap_max", "cpu_quota", "cpu_period"]
    )
    assert measured["unavailable"] == expected


def test_disposable_cleanup_failure_is_fatal_and_retried(tmp_path, monkeypatch):
    from openenv.validation.providers import CleanupError

    class FailedCleanup(MeasuredSubject):
        calls = 0

        def stop(self):
            self.calls += 1
            raise CleanupError("disposable helper remains")

        def measure_resources(self, timeout_s=180):
            self.stop()

    subject = FailedCleanup(measurement=CleanupError("disposable teardown failed"))
    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=FakeRuntimeProvider(capabilities=ENFORCING, subject=subject),
        artifacts_dir=tmp_path / "artifacts",
    )
    startup = next(
        result for result in report.results if result.check_id == "runtime.startup"
    )
    assert startup.status is CheckStatus.ERROR
    assert subject.calls == 2
    cleanup = json.loads((tmp_path / "artifacts" / "cleanup.json").read_text())
    assert cleanup["completed"] is False


@pytest.mark.parametrize("kind", ["memory", "pids", "disk", "exec_timeout"])
def test_missing_trusted_cause_cannot_qualify_a_claimed_result(kind):
    measured = bounded()
    probe = next(probe for probe in measured["qualification"] if probe["kind"] == kind)
    probe.update(counter_after=0, free_bytes_before={}, removed=False)
    assert grade(measured).status is CheckStatus.SKIP
