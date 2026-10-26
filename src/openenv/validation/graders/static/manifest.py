"""The `static.manifest` grader."""

import time

from ...manifest import NormalizedManifest
from ...policy import DeclarationBounds
from ...report import CheckResult
from ...types import CheckStatus, Level, ProviderCapability


class StaticManifestGrader:
    """Checks declared tolerances against the severity policy bounds."""

    check_id = "static.manifest"
    level = Level.STATIC
    requires_capabilities: frozenset[str] = frozenset()
    requires_provider: frozenset[ProviderCapability] = frozenset()
    depends_on: tuple[str, ...] = ()

    def __init__(self, bounds: DeclarationBounds):
        self._bounds = bounds

    def applies_to(self, manifest: NormalizedManifest) -> bool:
        return True

    def run(self, subject) -> CheckResult:
        started = time.monotonic()
        reward = subject.manifest.reward
        bounds = self._bounds
        problems = []
        if reward.oracle_tolerance > bounds.max_oracle_tolerance:
            problems.append(
                f"declared oracle_tolerance {reward.oracle_tolerance} exceeds the "
                f"policy maximum {bounds.max_oracle_tolerance}"
            )
        if reward.floor_margin < bounds.min_floor_margin:
            problems.append(
                f"declared floor_margin {reward.floor_margin} is below the "
                f"policy minimum {bounds.min_floor_margin}"
            )
        if (
            reward.variance_tolerance is not None
            and reward.variance_tolerance > bounds.max_variance_tolerance
        ):
            problems.append(
                f"declared variance_tolerance {reward.variance_tolerance} exceeds "
                f"the policy maximum {bounds.max_variance_tolerance}"
            )
        timeout = subject.manifest.resources.episode_timeout_s
        if timeout > bounds.max_episode_timeout_s:
            problems.append(
                f"declared episode_timeout_s {timeout} exceeds the "
                f"policy maximum {bounds.max_episode_timeout_s}"
            )
        return CheckResult(
            check_id=self.check_id,
            status=CheckStatus.FAIL if problems else CheckStatus.PASS,
            measured={
                "oracle_tolerance": reward.oracle_tolerance,
                "floor_margin": reward.floor_margin,
                "variance_tolerance": reward.variance_tolerance,
                "episode_timeout_s": timeout,
                "bounds": bounds.model_dump(),
            },
            evidence=problems or ["manifest is schema-valid and within policy bounds"],
            remediation=(
                "declare tolerances within the severity policy's bounds"
                if problems
                else None
            ),
            duration_s=time.monotonic() - started,
        )
