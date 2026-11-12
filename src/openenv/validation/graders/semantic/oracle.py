"""Verifier sanity over oracle, do-nothing and canary rewards (#778 tests #25, #28).

The execution binding measures the rewards; these graders only compare them with the
manifest's declared range, tolerance and margin. Comparisons use exact rationals so a
tiny floor cannot round into the margin.
"""

import time
from fractions import Fraction

from ...report import CheckResult
from ...types import CheckStatus, Level


class _SemanticGrader:
    level = Level.SEMANTIC
    requires_capabilities = frozenset()
    requires_provider = frozenset()
    # RFC 008: episode determinism is a precondition of every semantic check.
    depends_on = ("runtime.episode_determinism",)
    remediation = ""

    def applies_to(self, manifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        evidence = subject.oracle_evidence
        if evidence is None:
            return CheckResult(
                check_id=self.check_id,
                status=CheckStatus.ERROR,
                evidence=["oracle evidence is unavailable"],
                duration_s=0,
            )
        reward = subject.manifest.reward
        low, high = (Fraction(str(value)) for value in reward.range)
        problems, measured = self.check(
            evidence,
            maximum=high,
            width=high - low,
            tolerance=Fraction(str(reward.oracle_tolerance)),
            margin=Fraction(str(reward.floor_margin)),
        )
        return CheckResult(
            check_id=self.check_id,
            status=CheckStatus.FAIL if problems else CheckStatus.PASS,
            measured=measured,
            evidence=problems or [self.passed],
            remediation=self.remediation if problems else None,
            duration_s=time.monotonic() - started,
        )


class OracleMaxGrader(_SemanticGrader):
    """Every oracle run exits cleanly and meets the range-normalized tolerance."""

    check_id = "semantic.oracle_max"
    passed = "every oracle run reached the declared maximum within tolerance"
    remediation = (
        "Fix the oracle or the verifier so a correct solution exits cleanly and "
        "earns the declared maximum reward."
    )

    def check(self, evidence, *, maximum, width, tolerance, margin):
        threshold = maximum - tolerance * width
        problems = []
        for run, (reward, code) in enumerate(
            zip(evidence.oracle, evidence.oracle_exit_codes, strict=True), start=1
        ):
            if code != 0:
                problems.append(f"oracle run {run} exited with code {code}")
            if reward < threshold:
                problems.append(
                    f"oracle run {run} earned {float(reward)}, below {float(threshold)}"
                )
        return problems, {
            "oracle_rewards": [float(reward) for reward in evidence.oracle],
            "oracle_exit_codes": list(evidence.oracle_exit_codes),
            "declared_max": float(maximum),
            "oracle_tolerance": float(tolerance),
        }


class FloorGapGrader(_SemanticGrader):
    """
    The do-nothing reward sits at least `floor_margin` of the declared range below
    the maximum and every oracle run. The oracle may sit up to the tolerance below
    the maximum, so both gaps are required.
    """

    check_id = "semantic.floor_gap"
    passed = "the do-nothing reward leaves the declared margin"
    remediation = (
        "The verifier rewards doing nothing. Check for vacuous success conditions, "
        "such as comparisons against empty output."
    )

    def check(self, evidence, *, maximum, width, tolerance, margin):
        floor = evidence.floor
        required_gap = margin * width
        problems = []
        if maximum - floor < required_gap:
            problems.append(
                f"do-nothing reward {float(floor)} is within {float(maximum - floor)} "
                f"of the maximum, below the required gap {float(required_gap)}"
            )
        lowest = min(evidence.oracle)
        if lowest - floor < required_gap:
            problems.append(
                f"oracle reward {float(lowest)} is within {float(lowest - floor)} "
                f"of the do-nothing reward, below the required gap {float(required_gap)}"
            )
        return problems, {
            "measured_floor": float(floor),
            "oracle_reward": float(lowest),
            "declared_max": float(maximum),
            "declared_margin": float(margin),
        }


class CanaryFloorGrader(_SemanticGrader):
    """Every declared canary, a known-wrong solution, earns no more than doing nothing."""

    check_id = "semantic.canary_floor"
    requires_capabilities = frozenset({"canaries"})
    passed = "every canary stayed at the do-nothing reward"
    remediation = (
        "The verifier rewards a known-wrong solution. Tighten the tests so each "
        "canary earns no more than doing nothing."
    )

    def check(self, evidence, *, maximum, width, tolerance, margin):
        problems = [
            f"canary {name} earned {float(reward)}, above the do-nothing reward "
            f"{float(evidence.floor)}"
            for name, reward in evidence.canaries
            if reward > evidence.floor
        ]
        return problems, {
            "measured_floor": float(evidence.floor),
            "canary_rewards": {
                name: float(reward) for name, reward in evidence.canaries
            },
        }
