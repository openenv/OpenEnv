"""`runtime.resource_bounds`: measured limits and enforcement against the declared budget."""

import time

from ...report import CheckResult
from ...runtime.contracts import ResourceEvidence
from ...types import CheckStatus, Level, ProviderCapability

_MIB = 1024 * 1024
_PROBES = {
    "memory": "cgroup_oom_kill",
    "pids": "cgroup_pids_max",
    "disk": "filesystem_full",
    "exec_timeout": "exec_timeout",
}


class ResourceBoundsGrader:
    """Check measured resource ceilings and Docker exec descendant cleanup."""

    check_id = "runtime.resource_bounds"
    level = Level.RUNTIME
    requires_capabilities = frozenset()
    requires_provider = frozenset({ProviderCapability.RESOURCE_ENFORCEMENT})
    depends_on = ("runtime.startup",)

    def applies_to(self, manifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        status, problems, measured = self._grade(subject)
        return CheckResult(
            check_id=self.check_id,
            status=status,
            evidence=problems
            or [
                "measured CPU, memory, PID and storage ceilings match the declaration; "
                "Docker exec timeout removes descendants; episode deadline is not qualified"
            ],
            measured=measured,
            remediation=(
                "Run the subject under finite CPU, memory, swap, process and "
                "writable-storage limits within its declared budget."
                if status is CheckStatus.FAIL
                else None
            ),
            duration_s=time.monotonic() - started,
        )

    @staticmethod
    def _grade(subject):
        if subject.resource_evidence_json is None:
            return CheckStatus.SKIP, ["resource evidence is unavailable"], {}
        try:
            evidence = ResourceEvidence.model_validate_json(
                subject.resource_evidence_json
            )
        except ValueError:  # includes pydantic's ValidationError
            return CheckStatus.FAIL, ["malformed resource evidence"], {}

        declared = subject.manifest.resources
        cgroup = evidence.cgroup
        problems = []
        if cgroup is not None:
            if "memory_max" in cgroup.unavailable:
                pass
            elif cgroup.memory_max is None:
                problems.append("memory is unlimited")
            elif cgroup.memory_max > declared.memory_mb * _MIB:
                problems.append(
                    f"memory limit {cgroup.memory_max} B exceeds {declared.memory_mb} MiB"
                )
            if "swap_max" not in cgroup.unavailable and cgroup.swap_max != 0:
                problems.append("swap is not disabled")
            if {"cpu_quota", "cpu_period"} & set(cgroup.unavailable):
                pass
            elif cgroup.cpu_quota is None:
                problems.append("CPU is unlimited")
            elif cgroup.cpu_period is None:
                problems.append("CPU period is unavailable")
            elif cgroup.cpu_quota / cgroup.cpu_period > declared.cpu + 1e-9:
                problems.append(
                    f"CPU limit {cgroup.cpu_quota / cgroup.cpu_period:g} exceeds {declared.cpu:g}"
                )
            if "pids_max" not in cgroup.unavailable and cgroup.pids_max is None:
                problems.append("process count is unlimited")
        inventory = evidence.containment
        storage_unknown = inventory is None or not any(
            mount.mount_point == "/" for mount in inventory.mounts
        )
        filesystems = {}
        if inventory is not None:
            # Docker's root-owned device tmpfs is also bound over masked /proc files.
            dev = {
                mount.device
                for mount in inventory.mounts
                if (mount.mount_point, mount.fstype) == ("/dev", "tmpfs")
            }
            for mount in inventory.mounts:
                if mount.mount_point == "/" and not mount.read_only:
                    problems.append("the image root is writable")
                if mount.read_only or (mount.mount_point, mount.fstype) in {
                    ("/proc", "proc"),
                    ("/dev/pts", "devpts"),
                    ("/dev/mqueue", "mqueue"),
                }:
                    continue
                if mount.device in dev and "/dev" not in inventory.writable:
                    continue
                if mount.size_bytes is None:
                    storage_unknown = True
                else:
                    filesystems[mount.device] = mount.size_bytes
            if sorted(inventory.allotted_writable) != sorted(inventory.allotted):
                storage_unknown = True
        writable = sum(filesystems.values())
        if writable > declared.disk_mb * _MIB:
            problems.append(
                f"writable storage {writable} B exceeds {declared.disk_mb} MiB"
            )
        results = {probe.kind: probe.result for probe in evidence.qualification}
        problems += [
            f"the {kind} probe escaped its limit"
            for kind, result in results.items()
            if result == "escaped"
        ]
        measured = {
            "cgroup": cgroup.model_dump() if cgroup else None,
            "writable_bytes": writable,
            "qualification": results,
        }
        # A failed control cannot erase an independently proven violation.
        if problems:
            return CheckStatus.FAIL, problems, measured
        if cgroup is None:
            return CheckStatus.SKIP, ["cgroup v2 limits could not be read"], measured
        if cgroup.unavailable:
            return (
                CheckStatus.SKIP,
                ["unreadable cgroup limits: " + ", ".join(cgroup.unavailable)],
                measured,
            )
        if storage_unknown:
            return (
                CheckStatus.SKIP,
                ["complete writable-filesystem inventory is unavailable"],
                measured,
            )
        qualified = {probe.kind: probe for probe in evidence.qualification}
        unproven = []
        for kind, reason in _PROBES.items():
            probe = qualified.get(kind)
            if (
                probe is None
                or not probe.started
                or probe.result != "contained"
                or probe.reason != reason
            ):
                unproven.append(kind)
                continue
            increased = (
                probe.counter_before is not None
                and probe.counter_after is not None
                and probe.counter_after > probe.counter_before
            )
            corroborated = {
                "memory": probe.oom_killed or increased,
                "pids": increased,
                "disk": all(
                    probe.free_bytes.get(path) == 0
                    and (probe.free_bytes_before.get(path) or 0) > 0
                    for path in inventory.allotted
                ),
                "exec_timeout": probe.removed,
            }[kind]
            if not corroborated:
                unproven.append(kind)
        if unproven:
            return (
                CheckStatus.SKIP,
                [f"enforcement not proven for {', '.join(unproven)}"],
                measured,
            )
        return CheckStatus.PASS, [], measured
