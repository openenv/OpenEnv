"""`runtime.oracle_containment`: the oracle must not be readable through the API."""

import json
import time
from pathlib import Path

from ...report import CheckResult
from ...types import CheckStatus, Level

MAX_ORACLE_BYTES = 1024 * 1024
MAX_ORACLE_FILES = 64
_MIN_FINGERPRINT = 16


def _fingerprints(root: Path, location: str) -> list[str] | None:
    """Distinctive oracle text, or `None` when the artifact cannot be read safely."""
    package = root.resolve()
    target = (package / location).resolve()
    if not target.is_relative_to(package) or not target.exists():
        return None
    files = (
        sorted(p for p in target.rglob("*") if p.is_file())
        if target.is_dir()
        else [target]
    )
    texts, total = [], 0
    for path in files[:MAX_ORACLE_FILES]:
        if path.is_symlink():
            return None
        data = path.read_bytes()[: MAX_ORACLE_BYTES - total]
        total += len(data)
        texts.append(data.decode("utf-8", "ignore"))
        if total >= MAX_ORACLE_BYTES:
            break
    found = set()
    for text in texts:
        if len(text.strip()) >= _MIN_FINGERPRINT:
            found.add(text.strip())
        found.update(
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) >= _MIN_FINGERPRINT
        )
    return sorted(found)


def _appears(fingerprint: str, text: str) -> bool:
    # Responses are JSON, so the oracle may appear in escaped form.
    return fingerprint in text or json.dumps(fingerprint)[1:-1] in text


class OracleContainmentGrader:
    """Check that no API response carries the declared oracle artifact."""

    check_id = "runtime.oracle_containment"
    level = Level.RUNTIME
    requires_capabilities = frozenset({"oracle"})
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
            evidence=problems or ["no API response carries the oracle artifact"],
            measured=measured,
            remediation=(
                "Keep oracle and reward artifacts out of observations, state and "
                "served files."
                if status is CheckStatus.FAIL
                else None
            ),
            duration_s=time.monotonic() - started,
        )

    @staticmethod
    def _grade(subject):
        if subject.runtime_evidence is None:
            return CheckStatus.SKIP, ["runtime evidence is unavailable"], {}
        location = subject.manifest.capabilities.oracle.location
        try:
            fingerprints = _fingerprints(subject.root, location)
        except OSError:
            fingerprints = None
        if fingerprints is None:
            return (
                CheckStatus.SKIP,
                [f"oracle artifact {location} is not readable inside the package"],
                {},
            )
        if not fingerprints:
            return (
                CheckStatus.SKIP,
                ["the oracle artifact has no distinctive content to trace"],
                {},
            )
        runs = [subject.runtime_evidence]
        if subject.isolation_evidence:
            runs += [
                subject.isolation_evidence.sequential,
                subject.isolation_evidence.fresh,
            ]
        sources = [
            (f"{x.operation} response", x.response_json)
            for run in runs
            for x in run.exchanges
        ]
        for label, schema in (
            ("observation schema", subject.runtime_evidence.observation_schema_json),
            ("reset schema", subject.runtime_evidence.reset_observation_schema_json),
        ):
            if schema:
                sources.append((label, schema))
        sources += [
            (f"GET {probe.path}", probe.body.decode("utf-8", "ignore"))
            for probe in subject.oracle_http
            if probe.status == 200
        ]
        leaks = sorted(
            {
                label
                for label, text in sources
                if any(_appears(fp, text) for fp in fingerprints)
            }
        )
        measured = {"fingerprints": len(fingerprints), "sources": len(sources)}
        if leaks:
            return (
                CheckStatus.FAIL,
                [
                    f"oracle content is readable through the API: {label}"
                    for label in leaks
                ],
                measured,
            )
        return CheckStatus.PASS, [], measured
