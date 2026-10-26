"""Deterministic dependency ordering and capability-aware grader execution."""

import time
from graphlib import CycleError, TopologicalSorter

from ..policy import PolicyError
from ..report import CheckResult
from ..types import CheckStatus


def order_graders(graders: list) -> list:
    """Topologically order selected graders; reject duplicates and dependency cycles."""
    by_id = {grader.check_id: grader for grader in graders}
    if len(by_id) != len(graders):
        raise PolicyError("duplicate grader IDs")
    graph = {
        check_id: sorted(dep for dep in grader.depends_on if dep in by_id)
        for check_id, grader in sorted(by_id.items())
    }
    try:
        return [by_id[check_id] for check_id in TopologicalSorter(graph).static_order()]
    except CycleError as exc:
        raise PolicyError(f"grader dependency cycle at {exc.args[1][0]}") from exc


def execute_graders(graders, subject, *, provider_capabilities=frozenset(), prior=()):
    """Run independent checks despite failures, and name unmet dependencies in SKIPs."""
    results = []
    outcomes = {result.check_id: result for result in prior}
    for grader in order_graders(graders):
        blocked = [
            name
            for name in grader.depends_on
            if name not in outcomes or outcomes[name].status is not CheckStatus.PASS
        ]
        missing = sorted(
            capability.value
            for capability in grader.requires_provider - provider_capabilities
        )
        if blocked or missing:
            reasons = []
            if blocked:
                reasons.append("unmet dependencies: " + ", ".join(blocked))
            if missing:
                reasons.append("missing provider capabilities: " + ", ".join(missing))
            result = CheckResult(
                check_id=grader.check_id,
                status=CheckStatus.SKIP,
                evidence=reasons,
                duration_s=0,
            )
        else:
            started = time.monotonic()
            try:
                result = grader.run(subject)
                if (
                    not isinstance(result, CheckResult)
                    or result.check_id != grader.check_id
                ):
                    raise ValueError("grader returned the wrong check result")
            except Exception as exc:
                result = CheckResult(
                    check_id=grader.check_id,
                    status=CheckStatus.ERROR,
                    evidence=[f"grader failed ({type(exc).__name__})"],
                    duration_s=time.monotonic() - started,
                )
        outcomes[grader.check_id] = result
        results.append(result)
    return results
