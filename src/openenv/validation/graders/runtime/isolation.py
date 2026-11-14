"""`runtime.episode_isolation`: a reset must leave no trace of the previous episode."""

import time
from json import JSONDecodeError

from ...report import CheckResult
from ...runtime.json_values import loads, typed_json
from ...types import CheckStatus, Level


def _canonical(text):
    """Compare exact decoded values while preserving JSON types and malformed text."""
    try:
        value = loads(text)
    except JSONDecodeError:
        return text
    try:
        return typed_json(value)
    except ValueError:
        return text  # Malformed text is compared exactly as it was sent.


def _trace(evidence):
    trace, problems = [], []
    for x in evidence.exchanges:
        values = [x.operation]
        for text in (x.request_json, x.response_json):
            try:
                values.append(_canonical(text))
            except (RecursionError, ArithmeticError, ValueError) as exc:
                # Unknown values are neither equal nor different, even when
                # their raw bytes match. Keep the supported fields comparable.
                values.append(None)
                if isinstance(exc, RecursionError):
                    problem = "JSON nesting exceeds the parser depth"
                elif isinstance(exc, ArithmeticError):
                    problem = "JSON numeric range exceeds Decimal support"
                else:
                    problem = "JSON integer range exceeds parser support"
                if problem not in problems:
                    problems.append(problem)
        trace.append(tuple(values))
    return trace, problems


def _divergence(left, right, complete=True, controls=()):
    """
    Describe the first supported difference, or `None` when none is established.

    When `left` is an incomplete recording, only its recorded exchanges count.
    Unknown values cannot establish equality or a difference. A leak also
    requires supported fresh controls at the differing position.
    """
    fixed_inputs = True
    for index, (a, b) in enumerate(zip(left, right)):
        if controls:
            fixed_inputs = (
                fixed_inputs
                and None not in a[:2]
                and None not in b[:2]
                and a[:2] == b[:2]
                and all(
                    index < len(control) and control[index][:2] == b[:2]
                    for control in controls
                )
            )
        if controls and (
            not fixed_inputs
            or None in a
            or None in b
            or any(index >= len(control) or control[index] != b for control in controls)
        ):
            continue
        if any(x is not None and y is not None and x != y for x, y in zip(a, b)):
            return f"exchange {index} ({a[0]})"
    if complete and fixed_inputs and len(left) != len(right):
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
        failure = isolation.sequential.failure_reason
        fresh_failure = isolation.fresh.failure_reason
        sequential, sequential_problems = _trace(isolation.sequential)
        fresh, fresh_problems = _trace(isolation.fresh)
        baseline, baseline_problems = _trace(reference)
        problems = sequential_problems + fresh_problems + baseline_problems
        if fresh_failure and not fresh_problems:
            return (
                CheckStatus.SKIP,
                ["the fresh reference episode failed: " + fresh_failure],
                {},
            )
        resets = [
            i
            for i, x in enumerate(isolation.sequential.exchanges)
            if x.operation == "reset"
        ]
        if len(resets) < 2 and not failure:
            return CheckStatus.SKIP, ["no second episode was recorded"], {}
        split = resets[1] if len(resets) > 1 else len(sequential)
        first, after_reset = sequential[:split], sequential[split:]
        measured = {"exchanges": len(after_reset)}
        # Isolation is only observable when fixed inputs give a fixed trace, so
        # the first episode of every session must agree.
        # A validator limit may leave a usable fresh prefix, but never proves
        # equality with the unrecorded tail of a complete reference.
        drift = _divergence(baseline, fresh, complete=not fresh_failure) or _divergence(
            first, baseline, complete=not failure
        )
        if drift:
            return (
                CheckStatus.SKIP,
                [f"two fresh episodes differ at {drift}, so isolation is unproven"],
                measured,
            )
        leak = _divergence(
            after_reset,
            fresh,
            complete=not failure and not problems,
            controls=(baseline, first),
        )
        if leak:
            return (
                CheckStatus.FAIL,
                [
                    f"the episode after a same-session reset differs from a fresh one at {leak}"
                ],
                measured,
            )
        inputs = _divergence(
            [x[:2] for x in after_reset], [x[:2] for x in fresh], complete=False
        )
        if inputs:
            return (
                CheckStatus.SKIP,
                [
                    f"the episode after reset used different inputs at {inputs}, so isolation is unproven"
                ],
                measured,
            )
        if problems:
            return (
                CheckStatus.SKIP,
                [
                    problem + ", so isolation is unproven"
                    for problem in dict.fromkeys(problems)
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
