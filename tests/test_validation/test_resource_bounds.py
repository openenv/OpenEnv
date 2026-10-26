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
        "writable": [
            {"path": "/tmp", "size_bytes": 32 * MIB},
            {"path": "/dev/shm", "size_bytes": 32 * MIB},
        ],
        "qualification": [
            {"kind": kind, "result": "contained"}
            for kind in ("memory", "pids", "disk", "deadline")
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
    measured["writable"][0]["size_bytes"] = 64 * MIB
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "exceeds 64 MiB" in " ".join(result.evidence)


def test_escaped_probe_fails_even_with_correct_settings():
    measured = bounded()
    measured["qualification"][3]["result"] = "escaped"
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert "deadline probe escaped" in " ".join(result.evidence)


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
