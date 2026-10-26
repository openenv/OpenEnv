"""Severity policy: the only place a check acquires a severity.

The policy is a versioned data artifact (`policies/severity-v1.json`), independent of
code. It maps every known check id — including reserved hub/statistical ids — to a
lane and severity, and bounds what tolerances an author may declare.
"""

import json
from importlib.resources import files

from pydantic import BaseModel, ConfigDict, model_validator

from .report import CheckResult
from .types import CheckStatus, Lane, Level, Severity, Verdict


class PolicyError(Exception):
    """Internal policy error (unknown check id, missing policy file). CLI exit code 3."""


class PolicyEntry(BaseModel):
    """One check id's lane and severity under this policy version."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    level: Level
    lane: Lane
    severity: Severity


class DeclarationBounds(BaseModel):
    """
    Ceilings on author-declared manifest values.

    Authors declare tolerances in the manifest; the policy bounds what is declarable;
    reports carry the declared values so hubs can apply stricter ceilings.
    """

    model_config = ConfigDict(extra="forbid")

    max_oracle_tolerance: float
    min_floor_margin: float
    max_variance_tolerance: float
    max_episode_timeout_s: float


class SeverityPolicy(BaseModel):
    """A versioned severity policy: every known check id, exactly one lane each."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str
    entries: list[PolicyEntry]
    bounds: DeclarationBounds

    @model_validator(mode="after")
    def _unique_check_ids(self) -> "SeverityPolicy":
        ids = [entry.check_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate check ids in policy")
        return self

    def entries_for_lane(self, lane: Lane) -> dict[str, PolicyEntry]:
        """
        Return the applicable entries keyed by check id.

        Hub-lane entries are filtered out entirely for `lane=LOCAL` — an author never
        sees a check they cannot red-to-green. The hub lane is a superset: operators
        run the local checks plus the hub-only ones.

        Args:
            lane ([`~openenv.validation.types.Lane`]):
                The lane the run executes in.

        Returns:
            `dict[str, PolicyEntry]`: applicable entries keyed by check id.
        """
        if lane is Lane.LOCAL:
            applicable = [e for e in self.entries if e.lane is Lane.LOCAL]
        else:
            applicable = list(self.entries)
        return {e.check_id: e for e in applicable}


def load_policy(version: str = "v1") -> SeverityPolicy:
    """
    Load a committed severity policy by version.

    Args:
        version (`str`, *optional*, defaults to `"v1"`):
            The policy version to load.

    Returns:
        [`~openenv.validation.policy.SeverityPolicy`]: the parsed policy.
    """
    resource = files("openenv.validation").joinpath(
        "policies", f"severity-{version}.json"
    )
    try:
        raw = resource.read_text()
    except (FileNotFoundError, OSError) as exc:
        raise PolicyError(f"no severity policy for version {version!r}") from exc
    return SeverityPolicy.model_validate(json.loads(raw))


def apply_policy(
    results: list[CheckResult], policy: SeverityPolicy, lane: Lane
) -> Verdict:
    """
    Map check results to the run verdict. The only severity-assigning code path.

    FAIL if any policy-fail check FAILed; ERROR results fail closed; WARN if the run
    contains a SKIP or the only findings carry warn/advisory severity. Results with
    ids unknown to the policy (or outside the run's lane) are an internal error.

    Args:
        results (`list` of [`~openenv.validation.report.CheckResult`]):
            Grader outputs for this run.
        policy ([`~openenv.validation.policy.SeverityPolicy`]):
            The pinned policy version.
        lane ([`~openenv.validation.types.Lane`]):
            The lane the run executes in.

    Returns:
        [`~openenv.validation.types.Verdict`]: the overall verdict.
    """
    entries = policy.entries_for_lane(lane)
    classified_results = []
    for result in results:
        entry = entries.get(result.check_id)
        if entry is None:
            raise PolicyError(
                f"check id {result.check_id!r} is unknown to policy "
                f"{policy.policy_version!r} in lane {lane.value!r}"
            )
        classified_results.append((result, entry))

    verdict = Verdict.PASS
    for result, entry in classified_results:
        if result.status is CheckStatus.ERROR:
            return Verdict.FAIL
        if result.status is CheckStatus.FAIL:
            if entry.severity is Severity.FAIL:
                return Verdict.FAIL
            verdict = Verdict.WARN
        if result.status is CheckStatus.SKIP:
            verdict = Verdict.WARN
    return verdict
