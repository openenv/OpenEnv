"""`runtime.resource_bounds`: measured limits and enforcement against the declared budget."""

import time

from ...report import CheckResult
from ...runtime.contracts import ResourceEvidence
from ...types import CheckStatus, Level, ProviderCapability

_MIB = 1024 * 1024
_PROBES = ("memory", "pids", "disk", "deadline")


class ResourceBoundsGrader:
    """Check that one episode cannot exceed its declared CPU, memory, storage or time."""

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
            evidence=problems or ["measured limits enforce the declared budget"],
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
            if cgroup.memory_max is None:
                problems.append("memory is unlimited")
            elif cgroup.memory_max > declared.memory_mb * _MIB:
                problems.append(
                    f"memory limit {cgroup.memory_max} B exceeds {declared.memory_mb} MiB"
                )
            if cgroup.swap_max != 0:
                problems.append("swap is not disabled")
            if cgroup.cpu_quota is None or cgroup.cpu_period is None:
                problems.append("CPU is unlimited")
            elif cgroup.cpu_quota / cgroup.cpu_period > declared.cpu + 1e-9:
                problems.append(
                    f"CPU limit {cgroup.cpu_quota / cgroup.cpu_period:g} exceeds {declared.cpu:g}"
                )
            if cgroup.pids_max is None:
                problems.append("process count is unlimited")
        writable = sum(root.size_bytes for root in evidence.writable)
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
        unproven = [kind for kind in _PROBES if results.get(kind) != "contained"]
        if unproven:
            return (
                CheckStatus.SKIP,
                [f"enforcement not proven for {', '.join(unproven)}"],
                measured,
            )
        return CheckStatus.PASS, [], measured
