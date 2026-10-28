"""`runtime.host_containment` judges measured isolation, and the runner supplies it."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime.containment import HostContainmentGrader
from openenv.validation.providers import ProviderError
from openenv.validation.runner import run_validation
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence, FakeRunningSubject, FakeRuntimeProvider


def contained(**changes):
    """Evidence shaped like a hardened Docker-local subject."""
    measured = {
        "schema_version": 1,
        "launch": {
            "privileged": False,
            "host_namespaces": [],
            "cap_add": [],
            "devices": [],
            "unconfined": [],
        },
        "processes": [
            {
                "pid": 1,
                "name": "docker-init",
                "uids": [65532] * 4,
                "cap_eff": 0,
                "no_new_privs": True,
                "seccomp": 2,
            },
            {
                "pid": 7,
                "name": "python",
                "uids": [65532] * 4,
                "cap_eff": 0,
                "no_new_privs": True,
                "seccomp": 2,
            },
        ],
        "mounts": [
            {"mount_point": "/", "fstype": "overlay", "read_only": True},
            {"mount_point": "/proc", "fstype": "proc", "read_only": False},
            {"mount_point": "/proc/kcore", "fstype": "tmpfs", "read_only": True},
            {"mount_point": "/dev", "fstype": "tmpfs", "read_only": False},
            {"mount_point": "/dev/shm", "fstype": "tmpfs", "read_only": False},
            {"mount_point": "/sys/fs/cgroup", "fstype": "cgroup2", "read_only": True},
            {"mount_point": "/etc/hosts", "fstype": "ext4", "read_only": False},
            {"mount_point": "/sbin/docker-init", "fstype": "ext4", "read_only": True},
            {"mount_point": "/tmp", "fstype": "tmpfs", "read_only": False},
        ],
        "allotted": ["/tmp", "/dev/shm"],
        "allotted_writable": ["/tmp", "/dev/shm"],
        "writable": [],
    }
    measured.update(changes)
    return measured


def grade(measured):
    subject = Subject(
        root=Path("."),
        manifest=None,
        image_ref="sha256:" + "a" * 64,
        running=None,
        outputs_dir=Path("."),
        containment_evidence_json=None if measured is None else json.dumps(measured),
    )
    return HostContainmentGrader().run(subject)


def test_hardened_subject_passes():
    result = grade(contained())
    assert result.status is CheckStatus.PASS, result.evidence
    assert result.measured["processes"] == 2


@pytest.mark.parametrize(
    "launch, expected",
    [
        ({"privileged": True}, "privileged"),
        ({"host_namespaces": ["pid"]}, "host pid namespace"),
        ({"cap_add": ["SYS_ADMIN"]}, "SYS_ADMIN"),
        ({"devices": ["/dev/kvm"]}, "/dev/kvm"),
        ({"unconfined": ["seccomp=unconfined"]}, "seccomp=unconfined"),
    ],
)
def test_unsafe_inspection_fails(launch, expected):
    measured = contained()
    measured["launch"].update(launch)
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert expected in " ".join(result.evidence)
    assert result.remediation


@pytest.mark.parametrize(
    "process, expected",
    [
        ({"uids": [0, 0, 0, 0]}, "runs as root"),
        ({"cap_eff": 0x2000}, "capabilities 0x2000"),
        ({"no_new_privs": False}, "can gain privileges"),
        ({"seccomp": 0}, "without seccomp"),
    ],
)
def test_unsafe_process_fails(process, expected):
    measured = contained()
    measured["processes"][1].update(process)
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert expected in " ".join(result.evidence)


def test_host_mount_and_out_of_root_write_fail():
    measured = contained(writable=["/app"])
    measured["mounts"].append(
        {"mount_point": "/var/run/docker.sock", "fstype": "tmpfs", "read_only": False}
    )
    measured["mounts"].append(
        {"mount_point": "/dev/host", "fstype": "ext4", "read_only": False}
    )
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    evidence = " ".join(result.evidence)
    assert "/var/run/docker.sock" in evidence
    assert "/dev/host (ext4)" in evidence
    assert "outside its allotted roots: /app" in evidence


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"processes": []}, "no subject process"),
        ({"mounts": []}, "mount table"),
        ({"allotted_writable": ["/dev/shm"]}, "write denial is unproven"),
    ],
)
def test_inconclusive_measurement_is_unproven(changes, reason):
    result = grade(contained(**changes))
    assert result.status is CheckStatus.SKIP
    assert reason in " ".join(result.evidence)


def test_failed_control_cannot_hide_a_proven_violation():
    result = grade(contained(allotted_writable=[], writable=["/"]))
    assert result.status is CheckStatus.FAIL


def test_missing_or_malformed_evidence():
    assert grade(None).status is CheckStatus.SKIP
    assert grade({"schema_version": 1}).status is CheckStatus.FAIL


@dataclass
class MeasuredSubject(FakeRunningSubject):
    measurement: object = None
    measured: int = 0

    def measure_containment(self, timeout_s: float = 60) -> dict:
        self.measured += 1
        if isinstance(self.measurement, Exception):
            raise self.measurement
        return self.measurement


def validate(tmp_path, monkeypatch, subject, capabilities):
    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    monkeypatch.setattr(runner, "collect_runtime_evidence", lambda *a, **k: evidence())
    provider = FakeRuntimeProvider(capabilities=capabilities, subject=subject)
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    return {r.check_id: r for r in report.results}["runtime.host_containment"]


INSPECTING = frozenset(
    {
        ProviderCapability.IMAGE_BUILD,
        ProviderCapability.EXEC,
        ProviderCapability.INSPECTED_ISOLATION,
    }
)


def test_runner_grades_provider_measurement(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=contained())
    result = validate(tmp_path, monkeypatch, subject, INSPECTING)
    assert subject.measured == 1
    assert result.status is CheckStatus.PASS


def test_failed_measurement_leaves_the_check_unproven(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=ProviderError("Containment probe failed"))
    result = validate(tmp_path, monkeypatch, subject, INSPECTING)
    assert result.status is CheckStatus.SKIP
    assert "unavailable" in " ".join(result.evidence)


def test_provider_without_inspection_is_not_asked(tmp_path, monkeypatch):
    subject = MeasuredSubject(measurement=contained())
    result = validate(
        tmp_path,
        monkeypatch,
        subject,
        frozenset({ProviderCapability.IMAGE_BUILD, ProviderCapability.EXEC}),
    )
    assert subject.measured == 0
    assert result.status is CheckStatus.SKIP
    assert "inspected_isolation" in " ".join(result.evidence)
