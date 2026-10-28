"""`runtime.episode_isolation`: a reset must leave no trace of the previous episode."""

import time

from ...report import CheckResult
from ...types import CheckStatus, Level


def _trace(evidence):
    return [(x.operation, x.request_json, x.response_json) for x in evidence.exchanges]


def _divergence(left, right):
    """Describe the first differing exchange, or `None` when the traces match."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"exchange {index} ({a[0]})"
    if len(left) != len(right):
        return f"exchange {min(len(left), len(right))} (trace length)"
    return None


class EpisodeIsolationGrader:
    """Compare an episode that follows a same-session reset with a fresh episode."""

    check_id = "runtime.episode_isolation"
    level = Level.RUNTIME
    requires_capabilities = frozenset()
    requires_provider = frozenset()
    depends_on = ("runtime.startup",)

    def applies_to(self, manifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        status, problems, measured = self._grade(subject)
        return CheckResult(
            check_id=self.check_id,
            status=status,
            evidence=problems or ["an episode after reset matches a fresh episode"],
            measured=measured,
            remediation=(
                "Reset all episode state, including files, caches and rubric "
                "state, at the start of every episode."
                if status is CheckStatus.FAIL
                else None
            ),
            duration_s=time.monotonic() - started,
        )

    @staticmethod
    def _grade(subject):
        isolation = subject.isolation_evidence
        reference = subject.runtime_evidence
        if isolation is None or reference is None:
            return CheckStatus.SKIP, ["isolation evidence is unavailable"], {}
        if isolation.sequential.failure_reason:
            return (
                CheckStatus.FAIL,
                [
                    "a second episode on the same session failed: "
                    + isolation.sequential.failure_reason
                ],
                {},
            )
        if isolation.fresh.failure_reason:
            return (
                CheckStatus.SKIP,
                [
                    "the fresh reference episode failed: "
                    + isolation.fresh.failure_reason
                ],
                {},
            )
        sequential = _trace(isolation.sequential)
        resets = [i for i, x in enumerate(sequential) if x[0] == "reset"]
        if len(resets) < 2:
            return CheckStatus.SKIP, ["no second episode was recorded"], {}
        after_reset = sequential[resets[1] :]
        fresh = _trace(isolation.fresh)
        measured = {"exchanges": len(after_reset)}
        # Isolation is only observable when fixed inputs give a fixed trace.
        drift = _divergence(_trace(reference), fresh)
        if drift:
            return (
                CheckStatus.SKIP,
                [f"two fresh episodes differ at {drift}, so isolation is unproven"],
                measured,
            )
        leak = _divergence(after_reset, fresh)
        if leak:
            return (
                CheckStatus.FAIL,
                [
                    f"the episode after a same-session reset differs from a fresh one at {leak}"
                ],
                measured,
            )
        return CheckStatus.PASS, [], measured
