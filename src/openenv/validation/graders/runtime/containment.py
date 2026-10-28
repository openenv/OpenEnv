"""`runtime.host_containment`: judge measured isolation, never provider claims."""

import time

from ...report import CheckResult
from ...runtime.contracts import ContainmentEvidence
from ...types import CheckStatus, Level, ProviderCapability

# Mounts every container runtime adds. Anything else could expose host state.
_RUNTIME_MOUNTS = {
    "/",
    "/etc/hosts",
    "/etc/hostname",
    "/etc/resolv.conf",
    # Docker binds its init at /sbin/docker-init, which resolves to
    # /usr/sbin/docker-init in images where /sbin links to /usr/sbin.
    "/sbin/docker-init",
    "/usr/sbin/docker-init",
    "/run/podman-init",
}
_PSEUDO_ROOTS = ("/proc", "/sys", "/dev")
_PSEUDO_FS = {"proc", "sysfs", "tmpfs", "devpts", "mqueue", "cgroup", "cgroup2"}
# Read-write filesystems of a hardened Docker launch that hold no subject files.
_NOT_STORAGE = {
    # Kernel interfaces; Docker makes the sensitive entries read-only or masks them.
    ("/proc", "proc"),
    # Pseudo-terminal device nodes.
    ("/dev/pts", "devpts"),
    # POSIX message queues, bounded by the subject's own IPC namespace limits.
    ("/dev/mqueue", "mqueue"),
}


class HostContainmentGrader:
    """Check that the API-boundary subject cannot reach host state or privileges."""

    check_id = "runtime.host_containment"
    level = Level.RUNTIME
    requires_capabilities = frozenset()
    requires_provider = frozenset({ProviderCapability.INSPECTED_ISOLATION})
    depends_on = ("runtime.startup",)

    def applies_to(self, manifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        status, problems, measured = self._grade(subject)
        return CheckResult(
            check_id=self.check_id,
            status=status,
            evidence=problems or ["measured isolation matches the agent boundary"],
            measured=measured,
            remediation=(
                "Run the subject unprivileged from a read-only image, without host "
                "namespaces, mounts or devices, and with writes confined to its "
                "allotted roots."
                if status is CheckStatus.FAIL
                else None
            ),
            duration_s=time.monotonic() - started,
        )

    @staticmethod
    def _grade(subject):
        if subject.containment_evidence_json is None:
            return CheckStatus.SKIP, ["containment evidence is unavailable"], {}
        try:
            evidence = ContainmentEvidence.model_validate_json(
                subject.containment_evidence_json
            )
        except ValueError:  # includes pydantic's ValidationError
            return CheckStatus.FAIL, ["malformed containment evidence"], {}

        launch = evidence.launch
        problems = []
        if launch.privileged:
            problems.append("subject runs privileged")
        problems += [
            f"subject shares the host {ns} namespace" for ns in launch.host_namespaces
        ]
        problems += [f"capability added: {cap}" for cap in launch.cap_add]
        problems += [f"host device passed in: {device}" for device in launch.devices]
        problems += [f"confinement disabled: {option}" for option in launch.unconfined]
        for process in evidence.processes:
            label = f"process {process.pid} ({process.name})"
            if 0 in process.uids:
                problems.append(f"{label} runs as root")
            if process.cap_eff:
                problems.append(f"{label} holds capabilities {process.cap_eff:#x}")
            if not process.no_new_privs:
                problems.append(f"{label} can gain privileges")
            if process.seccomp == 0:
                problems.append(f"{label} runs without seccomp")
        for mount in evidence.mounts:
            pseudo = mount.fstype in _PSEUDO_FS and any(
                mount.mount_point == root or mount.mount_point.startswith(root + "/")
                for root in _PSEUDO_ROOTS
            )
            allotted = (
                mount.mount_point in evidence.allotted and mount.fstype == "tmpfs"
            )
            if mount.mount_point not in _RUNTIME_MOUNTS and not allotted and not pseudo:
                problems.append(
                    f"unexpected mount at {mount.mount_point} ({mount.fstype})"
                )
        # The read-write mounts are every filesystem the subject could store data in.
        if any(m.mount_point == "/" and not m.read_only for m in evidence.mounts):
            problems.append("the image root is writable")
        # Docker's root-owned /dev tmpfs, also bound as /dev/null over masked /proc
        # files. The write probe below shows whether the subject can use it.
        dev = {
            m.device
            for m in evidence.mounts
            if (m.mount_point, m.fstype) == ("/dev", "tmpfs")
        }
        for mount in evidence.mounts:
            if (
                mount.read_only
                or mount.mount_point in ("/", *evidence.allotted)
                or (mount.mount_point, mount.fstype) in _NOT_STORAGE
                or mount.device in dev
            ):
                continue
            problems.append(
                "writable filesystem outside the allotted roots: "
                f"{mount.mount_point} ({mount.fstype})"
            )
        not_storage = {
            m.mount_point
            for m in evidence.mounts
            if (m.mount_point, m.fstype) in _NOT_STORAGE
        }
        problems += [
            f"subject can write outside its allotted roots: {path}"
            for path in evidence.writable
            if path not in not_storage
        ]
        measured = {
            "processes": len(evidence.processes),
            "mounts": len(evidence.mounts),
            "allotted_writable": evidence.allotted_writable,
        }
        # A failed control cannot erase an independently proven violation.
        if problems:
            return CheckStatus.FAIL, problems, measured
        if not evidence.processes:
            return CheckStatus.SKIP, ["no subject process could be measured"], measured
        if not any(m.mount_point == "/" for m in evidence.mounts):
            return (
                CheckStatus.SKIP,
                ["the subject's mount table could not be read in full"],
                measured,
            )
        if sorted(evidence.allotted_writable) != sorted(evidence.allotted):
            return (
                CheckStatus.SKIP,
                [
                    "the probe could not write an allotted root, so write denial is unproven"
                ],
                measured,
            )
        return CheckStatus.PASS, [], measured
