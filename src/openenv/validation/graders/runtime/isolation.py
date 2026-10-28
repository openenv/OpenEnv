"""`runtime.episode_isolation`: a reset must leave no trace of the previous episode."""

import time

from ...report import CheckResult
from ...runtime.json_values import loads, typed_json
from ...types import CheckStatus, Level


def _canonical(text):
    """Compare exact decoded values while preserving JSON types and malformed text."""
    try:
        return typed_json(loads(text))
    except ValueError:
        return text  # Malformed text is compared exactly as it was sent.


def _trace(evidence):
    return [
        (x.operation, _canonical(x.request_json), _canonical(x.response_json))
        for x in evidence.exchanges
    ]


def _divergence(left, right, complete=True):
    """
    Describe the first differing exchange, or `None` when the traces match.

    When `left` is an incomplete recording, only its recorded exchanges count.
    """
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"exchange {index} ({a[0]})"
    if complete and len(left) != len(right):
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
        if isolation.fresh.failure_reason:
            return (
                CheckStatus.SKIP,
                [
                    "the fresh reference episode failed: "
                    + isolation.fresh.failure_reason
                ],
                {},
            )
        failure = isolation.sequential.failure_reason
        try:
            sequential = _trace(isolation.sequential)
            fresh = _trace(isolation.fresh)
            baseline = _trace(reference)
        except RecursionError:
            return (
                CheckStatus.SKIP,
                ["JSON nesting exceeds the parser depth, so isolation is unproven"],
                {},
            )
        except ArithmeticError:
            return (
                CheckStatus.SKIP,
                [
                    "JSON numeric range exceeds Decimal support, so isolation is unproven"
                ],
                {},
            )
        resets = [i for i, x in enumerate(sequential) if x[0] == "reset"]
        if len(resets) < 2 and not failure:
            return CheckStatus.SKIP, ["no second episode was recorded"], {}
        split = resets[1] if len(resets) > 1 else len(sequential)
        first, after_reset = sequential[:split], sequential[split:]
        measured = {"exchanges": len(after_reset)}
        # Isolation is only observable when fixed inputs give a fixed trace, so
        # the first episode of every session must agree.
        drift = _divergence(baseline, fresh) or _divergence(
            first, baseline, complete=not failure
        )
        if drift:
            return (
                CheckStatus.SKIP,
                [f"two fresh episodes differ at {drift}, so isolation is unproven"],
                measured,
            )
        leak = _divergence(after_reset, fresh, complete=not failure)
        if leak:
            return (
                CheckStatus.FAIL,
                [
                    f"the episode after a same-session reset differs from a fresh one at {leak}"
                ],
                measured,
            )
        if failure:
            # A deadline, size bound or transport failure that left no
            # differing response is a gap in the experiment, not a leak.
            return (
                CheckStatus.SKIP,
                [f"the same-session episodes did not complete: {failure}"],
                measured,
            )
        return CheckStatus.PASS, [], measured
