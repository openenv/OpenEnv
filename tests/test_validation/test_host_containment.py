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

MIB = 1024 * 1024


def mount(mount_point, fstype, device, read_only, size_bytes=None):
    return {
        "mount_point": mount_point,
        "fstype": fstype,
        "device": device,
        "read_only": read_only,
        "size_bytes": size_bytes,
    }


def contained(**changes):
    """Evidence measured from a hardened Docker-local subject."""
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
            mount("/", "overlay", "0:105", True, 60000 * MIB),
            mount("/proc", "proc", "0:179", False),
            mount("/proc/sys", "proc", "0:179", True),
            # Docker masks files in /proc by binding /dev/null over them.
            mount("/proc/kcore", "tmpfs", "0:180", False, 64 * MIB),
            mount("/proc/scsi", "tmpfs", "0:249", True, 4096),
            mount("/dev", "tmpfs", "0:180", False, 64 * MIB),
            mount("/dev/pts", "devpts", "0:181", False),
            mount("/dev/mqueue", "mqueue", "0:176", False),
            mount("/dev/shm", "tmpfs", "0:184", False, 32 * MIB),
            mount("/sys", "sysfs", "0:183", True),
            mount("/sys/fs/cgroup", "cgroup2", "0:41", True),
            mount("/usr/sbin/docker-init", "overlay", "0:26", True, 60000 * MIB),
            mount("/etc/hosts", "ext4", "254:1", True, 60000 * MIB),
            mount("/tmp", "tmpfs", "0:185", False, 32 * MIB),
        ],
        "allotted": ["/tmp", "/dev/shm"],
        "allotted_writable": ["/tmp", "/dev/shm"],
        # Creating a file in /dev/mqueue creates a message queue, not stored data.
        "writable": ["/dev/mqueue"],
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


# Docker's init lands under /usr/sbin only where /sbin links there.
@pytest.mark.parametrize("init", ["/sbin/docker-init", "/usr/sbin/docker-init"])
def test_init_binary_mount_passes(init):
    measured = contained()
    measured["mounts"] = [
        mount(init, "overlay", "0:26", True, 60000 * MIB)
        if m["mount_point"] == "/usr/sbin/docker-init"
        else m
        for m in measured["mounts"]
    ]
    result = grade(measured)
    assert result.status is CheckStatus.PASS, result.evidence


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
        mount("/var/run/docker.sock", "tmpfs", "0:300", False, 64 * MIB)
    )
    measured["mounts"].append(mount("/dev/host", "ext4", "254:1", False, 60000 * MIB))
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    evidence = " ".join(result.evidence)
    assert "/var/run/docker.sock" in evidence
    assert "/dev/host (ext4)" in evidence
    assert "outside its allotted roots: /app" in evidence


def test_writable_image_root_fails():
    measured = contained()
    measured["mounts"][0]["read_only"] = False
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["the image root is writable"]
    assert result.remediation


@pytest.mark.parametrize(
    "extra, expected",
    [
        # A tmpfs below /dev passes the mount check yet can hold subject files.
        (
            mount("/dev/cache", "tmpfs", "0:300", False, 1024 * MIB),
            "/dev/cache (tmpfs)",
        ),
        # Runtime bind mounts are expected only read-only.
        (
            mount("/etc/resolv.conf", "ext4", "254:1", False, 60000 * MIB),
            "/etc/resolv.conf (ext4)",
        ),
    ],
)
def test_writable_filesystem_outside_the_allotted_roots_fails(extra, expected):
    measured = contained()
    measured["mounts"].append(extra)
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert result.evidence == [
        f"writable filesystem outside the allotted roots: {expected}"
    ]


def test_subject_writable_dev_fails():
    result = grade(contained(writable=["/dev", "/dev/mqueue"]))
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["subject can write outside its allotted roots: /dev"]


def test_allotted_root_backed_by_host_storage_fails():
    measured = contained()
    measured["mounts"][-1] = mount("/tmp", "ext4", "254:1", False, 60000 * MIB)
    result = grade(measured)
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["unexpected mount at /tmp (ext4)"]


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"processes": []}, "no subject process"),
        # Write candidates come from the mount table, so none are tried without it.
        ({"mounts": [], "writable": []}, "mount table"),
        ({"mounts": contained()["mounts"][1:]}, "mount table"),
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
