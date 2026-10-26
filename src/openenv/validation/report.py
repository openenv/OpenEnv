"""Check results and the validation report.

The report is the cross-operator contract: schema-versioned JSON embedding the
normalized manifest verbatim. Local-lane reports contain only local-lane check ids —
an author is never shown a check they cannot red-to-green. Hub and statistical check
ids are reserved in this schema so operator reports and local reports share it.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .manifest import NormalizedManifest, NormalizedManifestV2
from .types import CheckStatus, Lane, Level, SignatureKind, Verdict


class CheckResult(BaseModel):
    """
    Outcome of one check, as emitted by its grader.

    Attributes:
        check_id (`str`):
            Stable, policy-addressed id. Local runs use `static.*`/`runtime.*`/
            `semantic.*`; the `hub.*` and `statistical.*` namespaces are reserved for
            operator lanes.
        status ([`~openenv.validation.types.CheckStatus`]):
            The grader's finding. Severity is assigned by the policy, never here.
        measured (`dict[str, Any]`, *optional*):
            Measured values, e.g. `{"oracle_reward": 0.98, "declared_tolerance": 0.05}`.
        evidence (`list[str]`, *optional*):
            Human-readable, credential-safe evidence lines.
        remediation (`str`, *optional*):
            What the author changes to go green.
        duration_s (`float`):
            Wall-clock time this check took.
    """

    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(pattern=r"^[a-z_]+\.[a-z_]+$")
    status: CheckStatus
    measured: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    remediation: str | None = None
    duration_s: float = Field(ge=0.0)


class ValidationReport(BaseModel):
    """
    One report per validation run, maximum information per run.

    Pin `policy_version` and a local run reproduces the hub verdict, modulo hub-lane
    checks.

    `manifest` is `None` only when the package's declarations failed the manifest
    schema. The report then carries the `static.manifest` FAIL explaining why.
    """

    model_config = ConfigDict(extra="forbid")

    report_schema_version: Literal["1"]
    target: str
    source_digest: str
    signature: SignatureKind
    manifest: NormalizedManifest | None
    policy_version: str
    lane: Lane
    levels_run: list[Level]
    results: list[CheckResult]
    verdict: Verdict


class ValidationReportV2(ValidationReport):
    """
    Report version 2 preserves runtime declarations and parse-failure reporting.

    A runtime request may target a version 1 package without an execution binding,
    or fail before producing any manifest. Both remain representable so the report
    can explain the unmet prerequisite or manifest error.
    """

    report_schema_version: Literal["2"]
    manifest: NormalizedManifestV2 | NormalizedManifest | None


def write_report(report: ValidationReport, path: Path | None = None) -> str:
    """
    Serialize a validation report to schema-versioned JSON.

    Args:
        report ([`~openenv.validation.report.ValidationReport`]):
            The completed report.
        path (`Path`, *optional*):
            When set, the JSON is also written to this file.

    Returns:
        `str`: the JSON payload, without a trailing newline.
    """
    payload = report.model_dump_json(indent=2)
    if path is not None:
        path.write_text(payload + "\n")
    return payload
