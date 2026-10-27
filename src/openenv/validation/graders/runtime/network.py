"""`runtime.network_policy`: judge measured reachability, never provider claims."""

import time

from ...report import CheckResult
from ...runtime.contracts import NetworkEvidence
from ...types import CheckStatus, Level, ProviderCapability


class NetworkPolicyGrader:
    """Compare the subject namespace's measured reachability with its declared mode."""

    check_id = "runtime.network_policy"
    level = Level.RUNTIME
    requires_capabilities = frozenset()
    requires_provider = frozenset({ProviderCapability.NETWORK_POLICY})
    depends_on = ("runtime.startup",)

    def applies_to(self, manifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        status, problems, measured = self._grade(subject)
        return CheckResult(
            check_id=self.check_id,
            status=status,
            evidence=problems or ["measured network access matches the declaration"],
            measured=measured,
            remediation=(
                "Run the subject under its declared network mode."
                if status is CheckStatus.FAIL
                else None
            ),
            duration_s=time.monotonic() - started,
        )

    def _grade(self, subject):
        declared = subject.manifest.network.mode
        if declared not in {"public", "no-network"}:
            return (
                CheckStatus.SKIP,
                [f"{declared} network grading is not implemented"],
                {},
            )
        if subject.network_evidence_json is None:
            return CheckStatus.SKIP, ["network evidence is unavailable"], {}
        try:
            evidence = NetworkEvidence.model_validate_json(
                subject.network_evidence_json
            )
        except ValueError:  # includes pydantic's ValidationError
            return CheckStatus.FAIL, ["malformed network evidence"], {}

        if evidence.requested_mode != declared:
            return (
                CheckStatus.FAIL,
                [f"evidence measured {evidence.requested_mode}, declared {declared}"],
                {},
            )
        # A probe kind is evidence only if it reaches the sink from the control side.
        counted = [p for p in evidence.probes if p.control.result == "reachable"]
        measured = {"counted_probes": [p.kind for p in counted]}
        result = (
            self._public(evidence, counted, measured)
            if declared == "public"
            else self._no_network(evidence, counted, measured)
        )
        # A failed control cannot erase an independently proven policy violation.
        if result[0] is CheckStatus.FAIL:
            return result
        if not any(p.kind == "tcp" for p in counted):
            return (
                CheckStatus.SKIP,
                [
                    "the controlled TCP sink was not reachable from the control namespace"
                ],
                measured,
            )
        return result

    @staticmethod
    def _public(evidence, counted, measured):
        problems = []
        if evidence.subject_network_mode == "none":
            problems.append("subject network mode is none, but public was declared")
        problems += [
            f"{p.kind} to the sink was denied from the subject namespace"
            for p in counted
            if p.subject.result == "denied"
        ]
        if problems:
            return CheckStatus.FAIL, problems, measured
        unproven = [p.kind for p in counted if p.subject.result != "reachable"]
        if unproven:
            return (
                CheckStatus.SKIP,
                [f"egress not proven for {', '.join(unproven)}"],
                measured,
            )
        return CheckStatus.PASS, [], measured

    @staticmethod
    def _no_network(evidence, counted, measured):
        namespace = evidence.namespace
        problems = [
            f"{p.kind} to the sink was reachable from the subject namespace"
            for p in counted
            if p.subject.result == "reachable"
        ]
        if evidence.subject_network_mode != "none":
            problems.append(
                f"subject network mode is {evidence.subject_network_mode}, not none"
            )
        if sorted(namespace.interfaces) != ["lo"]:
            problems.append(
                f"namespace has interfaces other than lo: {sorted(namespace.interfaces)}"
            )
        if namespace.default_route:
            problems.append("namespace has a default route")
        if namespace.ipv6_non_loopback:
            problems.append("namespace has IPv6 addresses outside lo")
        if problems:
            return CheckStatus.FAIL, problems, measured
        unproven = [p.kind for p in counted if p.subject.result != "denied"]
        if unproven:
            return (
                CheckStatus.SKIP,
                [f"denial not proven for {', '.join(unproven)}"],
                measured,
            )
        return CheckStatus.PASS, [], measured
