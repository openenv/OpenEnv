"""`runtime.oracle_containment`: trace declared artifacts through collected responses."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from ...report import CheckResult
from ...runtime.json_values import loads, typed_json
from ...types import CheckStatus, Level

MAX_ORACLE_BYTES = 1024 * 1024
MAX_ORACLE_FILES = 64
_MIN_FINGERPRINT = 16


@dataclass(frozen=True)
class OracleInventory:
    """Bounded file content and explicit coverage of the declared artifact."""

    files: tuple[tuple[str, str | None], ...]
    bytes_read: int
    omitted_files: int
    omitted_bytes: int
    errors: tuple[str, ...] = ()


def oracle_inventory(root: Path, location: str) -> OracleInventory:
    """Read bounded regular files inside the package; never trace truncated content."""
    package = root.resolve()
    target = package / location
    if not target.resolve().is_relative_to(package) or target.is_symlink():
        raise OSError("oracle is outside the package or is a symlink")
    if not target.exists():
        raise OSError("oracle does not exist")
    errors = []
    if target.is_dir():
        paths = []
        for directory, dirs, names in os.walk(
            target, onerror=lambda exc: errors.append("oracle directory is unreadable")
        ):
            parent = Path(directory)
            paths.extend(parent / name for name in names)
            paths.extend(parent / name for name in dirs if (parent / name).is_symlink())
        paths.sort()
    else:
        paths = [target]
    files, total, omitted_files, omitted_bytes = [], 0, 0, 0
    for index, path in enumerate(paths):
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("oracle entry is not a regular file")
            size = path.stat().st_size
            if index >= MAX_ORACLE_FILES or total >= MAX_ORACLE_BYTES:
                omitted_files += 1
                omitted_bytes += size
                continue
            with path.open("rb") as stream:
                data = stream.read(min(size, MAX_ORACLE_BYTES - total))
            total += len(data)
            complete = len(data) == size
            if not complete:
                omitted_files += 1
                omitted_bytes += size - len(data)
            files.append(
                (
                    path.relative_to(package).as_posix(),
                    data.decode("utf-8") if complete else None,
                )
            )
        except (OSError, UnicodeError):
            errors.append("oracle entry is unreadable or is not UTF-8 regular content")
    return OracleInventory(
        tuple(files), total, omitted_files, omitted_bytes, tuple(errors)
    )


def _json_values(text):
    """Walk JSON values iteratively, retaining raw text for non-JSON responses."""
    try:
        pending = [loads(text)]
    except (ValueError, ArithmeticError, RecursionError):
        return
    while pending:
        value = pending.pop()
        yield value
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def _appears(artifacts, strings, structured, text):
    # Whole artifacts avoid conclusive failures from ordinary source-code lines.
    if any(artifact in text for artifact in artifacts):
        return True
    parsed = False
    for value in _json_values(text):
        parsed = True
        if isinstance(value, str) and any(artifact in value for artifact in strings):
            return True
        if structured and isinstance(value, (dict, list)):
            try:
                if typed_json(value) in structured:
                    return True
            except ValueError:
                pass
    return not parsed and any(artifact in text for artifact in strings)


class OracleContainmentGrader:
    """Check for complete declared artifacts in the measured response surfaces."""

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
            evidence=problems
            or [
                "no complete oracle artifact matched collected reset/step/state responses, "
                "complete /schema responses or bounded same-origin HTTP file probes; "
                "MCP tool access and reward-artifact writes were not tested"
            ],
            measured=measured,
            remediation=(
                "Keep oracle and reward artifacts out of observations, state and served files."
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
            inventory = oracle_inventory(subject.root, location)
        except (OSError, UnicodeError):
            return (
                CheckStatus.SKIP,
                [f"oracle artifact {location} is not readable inside the package"],
                {},
            )
        artifacts = {
            text.strip()
            for _, text in inventory.files
            if text and len(text.strip()) >= _MIN_FINGERPRINT
        }
        untraceable = sum(
            text is not None and len(text.strip()) < _MIN_FINGERPRINT
            for _, text in inventory.files
        )
        # Structured artifacts are independent of indentation, key order and escaping.
        structured = set()
        strings = set(artifacts)
        for text in tuple(artifacts):
            try:
                value = loads(text)
                if isinstance(value, str) and len(value.strip()) >= _MIN_FINGERPRINT:
                    strings.add(value.strip())
                elif isinstance(value, (dict, list)):
                    structured.add(typed_json(value))
            except (ValueError, ArithmeticError, RecursionError):
                pass
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
        sources += [
            ("schema response", run.schema_response_json)
            for run in runs
            if run.schema_response_json is not None
        ]
        sources += [
            (f"GET {probe.path}", body.decode("utf-8", "replace"))
            for probe in subject.oracle_http
            for body in (*probe.redirect_bodies, probe.body)
        ]
        leaks = sorted(
            {
                label
                for label, text in sources
                if _appears(artifacts, strings, structured, text)
            }
        )
        measured = {
            "artifacts": len(inventory.files),
            "bytes_read": inventory.bytes_read,
            "omitted_files": inventory.omitted_files,
            "omitted_bytes": inventory.omitted_bytes,
            "inventory_errors": len(inventory.errors),
            "untraceable_files": untraceable,
            "sources": len(sources),
            "http_probes": len(subject.oracle_http),
            "mcp_tool_access_tested": False,
            "reward_artifact_writes_tested": False,
        }
        if leaks:
            return (
                CheckStatus.FAIL,
                [
                    f"oracle content is readable through the API: {label}"
                    for label in leaks
                ],
                measured,
            )
        incomplete = list(inventory.errors)
        if inventory.omitted_files:
            incomplete.append("oracle inventory exceeds the file or byte bound")
        if untraceable:
            incomplete.append(
                "some oracle files have no distinctive complete content to trace"
            )
        if not artifacts:
            incomplete.append(
                "the oracle artifact has no distinctive complete content to trace"
            )
        if any(
            run.failure_reason or not run.exchanges or run.schema_response_json is None
            for run in runs
        ):
            incomplete.append(
                "runtime transcript or complete /schema response is unavailable or incomplete"
            )
        required = {"/" + path for path, _ in inventory.files}
        if (
            not required
            or not required.issubset({probe.path for probe in subject.oracle_http})
            or any(
                probe.status is None
                or probe.failure
                or probe.status >= 500
                or probe.status == 206
                or 300 <= probe.status < 400
                for probe in subject.oracle_http
            )
        ):
            incomplete.append("required oracle HTTP probes are missing or incomplete")
        if incomplete:
            return CheckStatus.SKIP, incomplete, measured
        return CheckStatus.PASS, [], measured
