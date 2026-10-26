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
    "/usr/sbin/docker-init",
    "/run/podman-init",
}
_PSEUDO_ROOTS = ("/proc", "/sys", "/dev")
_PSEUDO_FS = {"proc", "sysfs", "tmpfs", "devpts", "mqueue", "cgroup", "cgroup2"}


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
                "Run the subject unprivileged, without host namespaces, mounts or "
                "devices, and with writes confined to its allotted roots."
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
            if (
                mount.mount_point not in _RUNTIME_MOUNTS
                and mount.mount_point not in evidence.allotted
                and not pseudo
            ):
                problems.append(
                    f"unexpected mount at {mount.mount_point} ({mount.fstype})"
                )
        problems += [
            f"subject can write outside its allotted roots: {path}"
            for path in evidence.writable
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
        if not evidence.mounts:
            return (
                CheckStatus.SKIP,
                ["the subject's mount table could not be read"],
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
