"""Small, bounded reproduction bundles; reports remain the public result contract."""

import hashlib
import json
import math
import platform
import re
from dataclasses import asdict
from pathlib import Path

_SECRET_KEY = re.compile(r"(?i)(password|secret|token|authorization|api[_-]?key)")
_TOKEN_METADATA_KEYS = {"max_tokens", "prompt_token_ids"}
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{8,}|(?:sk|ghp|github_pat)[-_][A-Za-z0-9_-]{8,}|(?i:bearer)\s+\S+)"
)
_MAX_ARTIFACT_DEPTH = 64


def _redact(value, *, depth=0):
    if depth >= _MAX_ARTIFACT_DEPTH and isinstance(value, (dict, list)):
        return {"omitted": "artifact nesting limit exceeded"}
    if isinstance(value, dict):
        return {
            _TOKEN.sub("[REDACTED]", key): "[REDACTED]"
            if _SECRET_KEY.search(key) and key not in _TOKEN_METADATA_KEYS
            else _redact(child, depth=depth + 1)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child, depth=depth + 1) for child in value]
    if isinstance(value, str):
        return _TOKEN.sub("[REDACTED]", value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_number": str(value)}
    return value


def write_runtime_bundle(
    directory: Path, report, *, plan=None, evidence=None, provider=None, cleanup=None
):
    """Write a credential-filtered trace, source/policy/plan provenance and coverage."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    coverage = {
        "requested_runtime_checks": [
            r.check_id for r in report.results if r.check_id.startswith("runtime.")
        ],
        "executed": [r.check_id for r in report.results if r.status.value != "skip"],
        "incomplete": [r.check_id for r in report.results if r.status.value == "skip"],
    }
    files["coverage.json"] = coverage
    files["report.json"] = report.model_dump(mode="json")
    files["cleanup.json"] = cleanup or {"required": False}
    if plan:
        files["runtime-plan.json"] = plan.model_dump(mode="json")
    files["run-manifest.json"] = {
        "bundle_schema_version": "1",
        "source_digest": report.source_digest,
        "policy_version": report.policy_version,
        "manifest_schema_version": report.manifest.manifest_schema_version
        if report.manifest
        else None,
        "plan_digest": hashlib.sha256(plan.model_dump_json().encode()).hexdigest()
        if plan
        else None,
        "plan_redacted": bool(
            plan
            and _redact(plan.model_dump(mode="json")) != plan.model_dump(mode="json")
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "provider": provider or {},
    }
    trace = []
    collector_metadata = None
    if evidence:
        omitted_fields = []
        for index, exchange in enumerate(evidence.exchanges):
            row = asdict(exchange)
            for key in ("request_json", "response_json"):
                try:
                    row[key] = json.loads(row[key])
                except (ValueError, RecursionError):
                    row[key] = "[malformed response omitted]"
                    omitted_fields.append({"exchange_index": index, "field": key})
            trace.append(row)
        schema = None
        schema_parse_failed = False
        if evidence.observation_schema_json is not None:
            try:
                schema = json.loads(evidence.observation_schema_json)
            except (ValueError, RecursionError):
                schema_parse_failed = True
        reset_schema = None
        reset_schema_parse_failed = False
        if evidence.reset_observation_schema_json is not None:
            try:
                reset_schema = json.loads(evidence.reset_observation_schema_json)
            except (ValueError, RecursionError):
                reset_schema_parse_failed = True
        collector_metadata = {
            "evidence_schema_version": "1",
            "trace_file": "collector-trace.json",
            "observation_schema": schema,
            "schema_available": evidence.observation_schema_json is not None,
            "schema_parse_failed": schema_parse_failed,
            "reset_observation_schema": reset_schema,
            "reset_schema_available": evidence.reset_observation_schema_json
            is not None,
            "reset_schema_parse_failed": reset_schema_parse_failed,
            "omitted_trace_fields": omitted_fields,
            "failure_phase": evidence.failure_phase,
            "failure_reason": evidence.failure_reason,
            "complete": evidence.failure_phase is None
            and evidence.failure_reason is None,
        }
        collector_metadata["redacted"] = (
            bool(omitted_fields)
            or schema_parse_failed
            or reset_schema_parse_failed
            or _redact(trace) != trace
            or _redact(collector_metadata) != collector_metadata
        )
    files["collector-trace.json"] = trace
    files["collector-evidence.json"] = collector_metadata
    digests = []
    for name, value in files.items():
        payload = (
            json.dumps(_redact(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        (directory / name).write_bytes(payload)
        digests.append(f"{hashlib.sha256(payload).hexdigest()}  {name}\n")
    (directory / "SHA256SUMS").write_text("".join(digests))
