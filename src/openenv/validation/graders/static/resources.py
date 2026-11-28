"""Static checks for the declared resource budget and episode timeout."""

import time

from ...manifest import NormalizedManifest
from ...policy import DeclarationBounds
from ...report import CheckResult
from ...types import CheckStatus, Level, ProviderCapability
from .. import Subject


class ResourceDeclarationGrader:
    """Report the required resource budget from a parsed manifest."""

    check_id = "static.resource_declaration"
    level = Level.STATIC
    requires_capabilities: frozenset[str] = frozenset()
    requires_provider: frozenset[ProviderCapability] = frozenset()
    depends_on: tuple[str, ...] = ()

    def applies_to(self, manifest: NormalizedManifest) -> bool:
        return True

    def run(self, subject: Subject) -> CheckResult:
        started = time.monotonic()
        resources = subject.manifest.resources
        measured = resources.model_dump(
            include={"cpu", "memory_mb", "disk_mb", "episode_timeout_s"}
        )
        return CheckResult(
            check_id=self.check_id,
            status=CheckStatus.PASS,
            measured=measured,
            duration_s=time.monotonic() - started,
        )


class TimeoutCeilingGrader:
    """Check the declared timeout against the active policy."""

    check_id = "static.timeout_ceiling"
    level = Level.STATIC
    requires_capabilities: frozenset[str] = frozenset()
    requires_provider: frozenset[ProviderCapability] = frozenset()
    depends_on = ("static.resource_declaration",)

    def __init__(self, bounds: DeclarationBounds):
        self._maximum = bounds.max_episode_timeout_s

    def applies_to(self, manifest: NormalizedManifest) -> bool:
        return True

    def run(self, subject: Subject) -> CheckResult:
        started = time.monotonic()
        timeout = subject.manifest.resources.episode_timeout_s
        exceeds_maximum = timeout > self._maximum
        return CheckResult(
            check_id=self.check_id,
            status=CheckStatus.FAIL if exceeds_maximum else CheckStatus.PASS,
            measured={
                "episode_timeout_s": timeout,
                "max_episode_timeout_s": self._maximum,
            },
            evidence=[
                f"declared episode timeout {timeout}s exceeds the policy maximum {self._maximum}s"
                if exceeds_maximum
                else "episode timeout is within the policy maximum"
            ],
            remediation=(
                f"set resources.episode_timeout_s to at most {self._maximum}s"
                if exceeds_maximum
                else None
            ),
            duration_s=time.monotonic() - started,
        )
