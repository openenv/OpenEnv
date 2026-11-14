"""Pure scoring of bounded coding-audit evidence collected by a trusted runner.

This profile measures reference replay, known-negative rejection, explicit leakage
probes, and target difficulty. It does not establish containment or trainability.
"""

import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from statistics import pvariance
from typing import Any

PROFILE = "harbor-coding-audit-v1"
_CHECKS = (
    "references",
    "verifier_rejection",
    "reliability",
    "leakage",
    "target_difficulty",
)
_PROBES = (
    "private_paths_absent",
    "fresh_state_absent",
    "canary_not_in_transcript",
)
_RANK = {"pass": 0, "skip": 1, "fail": 2, "error": 3}
_SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _sha(value: str) -> str:
    return value.removeprefix("sha256:")


def _exact(value: int | float) -> Fraction:
    return Fraction(str(value))


def validate_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return quality thresholds, raising `ValueError` for invalid configuration.

    `required_negative_cases` must come from a trusted reviewed configuration;
    generated labels in evidence cannot supply this rejection coverage.
    """
    defaults = {
        "reference_repeats": 2,
        "target_attempts": 2,
        "reward_min": 0,
        "reward_max": 1,
        "oracle_tolerance": 0,
        "floor_margin": 0.1,
        "wrong_answer_max": 0,
        "required_negative_cases": None,
    }
    if not isinstance(config, dict) or set(config) - defaults.keys():
        raise ValueError("config must contain only documented quality thresholds")
    result = defaults | config
    for name in ("reference_repeats", "target_attempts"):
        if type(result[name]) is not int or result[name] < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    for name in (
        "reward_min",
        "reward_max",
        "oracle_tolerance",
        "floor_margin",
        "wrong_answer_max",
    ):
        if not (
            _number(result[name])
            or isinstance(result[name], Decimal)
            and result[name].is_finite()
        ):
            raise ValueError(f"{name} must be a finite nonbool number")
    low, high = (_exact(result[name]) for name in ("reward_min", "reward_max"))
    tolerance, margin = (
        _exact(result[name]) for name in ("oracle_tolerance", "floor_margin")
    )
    if not (
        low < high
        and 0 <= tolerance < high - low
        and 0 < margin <= high - low
        and low <= _exact(result["wrong_answer_max"]) < high - tolerance
        and low + tolerance < high - tolerance
    ):
        raise ValueError("reward bounds, tolerance, margin or negative limit conflict")
    cases = result["required_negative_cases"]
    if (
        not isinstance(cases, list)
        or not cases
        or any(not _text(case) for case in cases)
        or len(set(cases)) != len(cases)
    ):
        raise ValueError("required_negative_cases must be nonempty unique case IDs")
    result["required_negative_cases"] = list(cases)
    return result


def _issue(check: dict[str, Any], status: str, reason: str) -> None:
    if _RANK[status] > _RANK[check["status"]]:
        check["status"] = status
    check["reasons"].append(reason)


def _finish(checks: dict[str, Any]) -> dict[str, Any]:
    verdict = max((check["status"] for check in checks.values()), key=_RANK.get)
    return {
        "profile": PROFILE,
        "accepted": verdict == "pass",
        "verdict": verdict,
        "checks": checks,
        "rfc008_complete": False,
        "trainability_established": False,
    }


def _wilson(solved: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    rate = solved / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2))
    return [
        max(0.0, center - half / denominator),
        min(1.0, center + half / denominator),
    ]


def _target_metrics(
    raw: list[dict[str, Any]], trials: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    rewards = [trial["reward"] for trial in trials]
    count = len(rewards)
    solved = sum(
        _exact(reward)
        >= _exact(config["reward_max"]) - _exact(config["oracle_tolerance"])
        for reward in rewards
    )
    telemetry_missing = {"latency_seconds": 0, "input_tokens": 0, "output_tokens": 0}
    telemetry_invalid = dict(telemetry_missing)
    telemetry_errors = []
    latency: list[Fraction] = []
    usage: dict[str, list[int]] = {"input_tokens": [], "output_tokens": []}
    # Failed executions can still consume tokens/time. Account for all records.
    for trial in raw:
        seconds = trial.get("latency_seconds")
        if "latency_seconds" not in trial:
            telemetry_missing["latency_seconds"] += 1
        elif _number(seconds) and seconds >= 0:
            latency.append(_exact(seconds))
        else:
            telemetry_invalid["latency_seconds"] += 1
            telemetry_errors.append("invalid target latency_seconds")
        tokens = trial.get("usage")
        if "usage" in trial and not isinstance(tokens, dict):
            telemetry_errors.append("invalid target usage object")
        for name in usage:
            value = tokens.get(name) if isinstance(tokens, dict) else None
            if "usage" not in trial or isinstance(tokens, dict) and name not in tokens:
                telemetry_missing[name] += 1
            elif type(value) is int and value >= 0:
                usage[name].append(value)
            else:
                telemetry_invalid[name] += 1
                telemetry_errors.append(f"invalid target {name}")
    latency_total = None
    if latency:
        try:
            latency_total = float(sum(latency))
        except OverflowError:
            telemetry_errors.append("latency total exceeds finite report precision")
    return {
        "attempted": len(raw),
        "completed": count,
        "failed": len(raw) - count,
        "infrastructure_errors": sum(trial.get("error") is not None for trial in raw),
        "missing": max(0, config["target_attempts"] - count),
        "solve_count": solved,
        "solve_rate": solved / count if count else None,
        "solve_rate_wilson95": _wilson(solved, count),
        "reward_variance": _variance(rewards),
        "mixed_success": 0 < solved < count if count else None,
        "difficulty": (
            "unmeasured"
            if not count
            else "mixed"
            if 0 < solved < count
            else "all_solved"
            if solved == count
            else "none_solved"
        ),
        "latency_seconds_total": latency_total,
        "usage_totals": {
            name: sum(values) if values else None for name, values in usage.items()
        },
        "telemetry_missing": telemetry_missing,
        "telemetry_invalid": telemetry_invalid,
        "telemetry_errors": telemetry_errors,
    }


def _variance(rewards: list[float]) -> float | None:
    if not rewards:
        return None
    try:
        value = pvariance(rewards)
    except OverflowError:
        return None
    return value if _number(value) else None


def assess_coding_audit(
    config: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    """Score a trusted runner's evidence without executing tasks or making IO calls.

    Trials bind to the common task, verifier, images and pinned target identity.
    Reference inventories bind replay groups by `(case_id, submission_digest)`.
    Missing coverage is `skip`, contradictions are `fail`, and malformed or
    infrastructure evidence is `error`; only five passing checks admit an audit.
    """
    checks = {name: {"status": "pass", "reasons": []} for name in _CHECKS}
    try:
        config = validate_audit_config(config)
    except ValueError as exc:
        for check in checks.values():
            _issue(check, "error", str(exc))
        return _finish(checks)
    if not isinstance(evidence, dict):
        for check in checks.values():
            _issue(check, "error", "evidence must be an object")
        return _finish(checks)

    reliability = checks["reliability"]
    references = checks["references"]
    rejection = checks["verifier_rejection"]
    target = checks["target_difficulty"]
    low, high, tolerance, margin, negative_max = (
        _exact(config[name])
        for name in (
            "reward_min",
            "reward_max",
            "oracle_tolerance",
            "floor_margin",
            "wrong_answer_max",
        )
    )
    common_ok = True
    for name, pattern in (
        ("task_digest", _SHA256),
        ("verifier_digest", _SHA256),
        ("agent_image", _IMAGE),
        ("verifier_image", _IMAGE),
    ):
        if not _matches(pattern, evidence.get(name)):
            _issue(reliability, "error", f"invalid or missing common {name}")
            common_ok = False
    identity = evidence.get("target")
    if not (
        isinstance(identity, dict)
        and all(_text(identity.get(name)) for name in ("model", "provider"))
        and _matches(_REVISION, identity.get("revision"))
    ):
        _issue(
            target, "error", "target requires model, provider and immutable revision"
        )
        identity = None

    inventory = evidence.get("references", [])
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(inventory, list):
        _issue(references, "error", "references must be a list")
        inventory = []
    for candidate in inventory:
        if not (
            isinstance(candidate, dict)
            and _text(candidate.get("case_id"))
            and _matches(_SHA256, candidate.get("submission_digest"))
            and candidate.get("origin") in ("provided", "generated")
            and (
                candidate["origin"] != "generated"
                or _matches(_SHA256, candidate.get("candidate_script_digest"))
            )
        ):
            _issue(references, "error", "invalid reference candidate metadata")
            continue
        key = (candidate["case_id"], _sha(candidate["submission_digest"]))
        if key in candidates:
            _issue(references, "error", f"duplicate reference candidate {key[0]}")
        candidates[key] = candidate
    if not candidates:
        _issue(references, "skip", "no declared reference candidates")

    raw_trials = evidence.get("trials", [])
    if not isinstance(raw_trials, list):
        _issue(reliability, "error", "trials must be a list")
        raw_trials = []
    sandbox_counts = Counter(
        trial[name]
        for trial in raw_trials
        if isinstance(trial, dict)
        for name in ("sandbox_id", "grade_sandbox_id")
        if _text(trial.get(name))
    )
    target_cases = Counter(
        trial["case_id"]
        for trial in raw_trials
        if isinstance(trial, dict)
        and trial.get("role") == "target"
        and _text(trial.get("case_id"))
    )
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_target = []
    for index, trial in enumerate(raw_trials):
        if not isinstance(trial, dict) or trial.get("role") not in (
            "nop",
            "reference",
            "negative",
            "target",
        ):
            _issue(reliability, "error", f"trial {index}: invalid role or record")
            continue
        role = trial["role"]
        role_check = checks[
            {
                "nop": "verifier_rejection",
                "negative": "verifier_rejection",
                "reference": "references",
                "target": "target_difficulty",
            }[role]
        ]
        if role == "target":
            raw_target.append(trial)
        errors = []
        if not common_ok:
            errors.append("common binding unavailable")
        for name, expected in (
            ("task_digest", evidence.get("task_digest")),
            ("verifier_digest", evidence.get("verifier_digest")),
            ("image_ref", evidence.get("agent_image")),
            ("verifier_image_ref", evidence.get("verifier_image")),
        ):
            actual = trial.get(name)
            if name in ("task_digest", "verifier_digest"):
                matches = (
                    _matches(_SHA256, actual)
                    and _matches(_SHA256, expected)
                    and _sha(actual) == _sha(expected)
                )
            else:
                matches = actual == expected
            if not matches:
                errors.append(f"{name} binding mismatch")
        submission_ok = _text(trial.get("case_id")) and _matches(
            _SHA256, trial.get("submission_digest")
        )
        if not submission_ok:
            errors.append("missing case ID or submission digest")
        for name in ("sandbox_id", "grade_sandbox_id"):
            value = trial.get(name)
            if not _text(value) or sandbox_counts[value] != 1:
                errors.append(f"{name} must be fresh and distinct")
        if trial.get("cleanup_verified") is not True:
            errors.append("cleanup not verified")
        if "error" not in trial or trial["error"] is not None:
            errors.append("execution error or missing error status")
        reward = trial.get("reward")
        if not _number(reward) or not low <= _exact(reward) <= high:
            errors.append("reward must be finite nonbool and in bounds")
        if role == "target":
            if identity is None or any(
                trial.get(name) != identity[name]
                for name in ("model", "revision", "provider")
            ):
                errors.append("target identity mismatch")
            if _text(trial.get("case_id")) and target_cases[trial["case_id"]] != 1:
                errors.append("duplicate target attempt case ID")
        if (
            role == "reference"
            and submission_ok
            and (trial["case_id"], _sha(trial["submission_digest"])) not in candidates
        ):
            errors.append("undeclared reference submission")
        if errors:
            message = f"trial {index}: " + "; ".join(errors)
            _issue(role_check, "error", message)
            _issue(reliability, "error", message)
        else:
            by_role[role].append(trial)
    if not raw_trials:
        _issue(reliability, "skip", "no trial evidence")

    floor_rewards = [_exact(trial["reward"]) for trial in by_role["nop"]]
    if not floor_rewards:
        _issue(rejection, "skip", "NOP floor evidence missing")
        _issue(references, "skip", "NOP floor evidence missing")
    elif max(floor_rewards) > low + tolerance:
        _issue(rejection, "fail", "NOP reward exceeds configured floor")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in by_role["reference"]:
        groups[(trial["case_id"], _sha(trial["submission_digest"]))].append(trial)
    validated = []
    excluded = []
    for key, candidate in candidates.items():
        trials = groups[key]
        rewards = [_exact(trial["reward"]) for trial in trials]
        if not rewards:
            _issue(references, "skip", f"candidate {key[0]} execution missing")
            continue
        stable = max(rewards) - min(rewards) <= tolerance
        if not stable:
            _issue(reliability, "fail", f"reference {key[0]} replay reward drift")
        correct = min(rewards) >= high - tolerance
        if not correct:
            if candidate["origin"] == "generated":
                excluded.append({"case_id": key[0], "submission_digest": key[1]})
            else:
                _issue(
                    references, "fail", f"provided reference {key[0]} reward too low"
                )
            continue
        if len(trials) < config["reference_repeats"]:
            _issue(references, "skip", f"reference {key[0]} replay coverage incomplete")
            _issue(
                reliability, "skip", f"reference {key[0]} replay coverage incomplete"
            )
            continue
        if floor_rewards and min(rewards) - max(floor_rewards) < margin:
            _issue(references, "fail", f"reference {key[0]} lacks floor separation")
            continue
        if stable and floor_rewards:
            validated.append({"case_id": key[0], "submission_digest": key[1]})
    references.update(validated=validated, excluded=excluded)
    if not validated:
        _issue(references, "skip", "no validated reference")
        _issue(reliability, "skip", "no validated reference replay group")

    negatives = defaultdict(list)
    for trial in by_role["negative"]:
        negatives[trial["case_id"]].append(_exact(trial["reward"]))
    covered = []
    for case in config["required_negative_cases"]:
        if not negatives[case]:
            _issue(rejection, "skip", f"trusted negative {case} execution missing")
        elif max(negatives[case]) > negative_max:
            _issue(rejection, "fail", f"trusted negative {case} was rewarded")
        else:
            covered.append(case)
    rejection["covered_negative_cases"] = covered

    leakage = checks["leakage"]
    leakage["scope"] = "bounded probes; universal containment is not established"
    probes = evidence.get("leakage_probes", [])
    if not isinstance(probes, list):
        _issue(leakage, "error", "leakage_probes must be a list")
        probes = []
    seen = set()
    for probe in probes:
        if not isinstance(probe, dict) or not _text(probe.get("name")):
            _issue(leakage, "error", "invalid leakage probe record")
            continue
        name = probe["name"]
        if name in seen:
            _issue(leakage, "error", f"duplicate leakage probe {name}")
        seen.add(name)
        if type(probe.get("passed")) is not bool or "error" not in probe:
            _issue(leakage, "error", f"invalid leakage probe result {name}")
        elif probe["error"] is not None:
            _issue(leakage, "error", f"leakage probe {name} execution error")
        elif not probe["passed"]:
            _issue(leakage, "fail", f"leakage probe {name} failed")
    for name in _PROBES:
        if name not in seen:
            _issue(leakage, "skip", f"required leakage probe {name} missing")
    leakage["observed_probes"] = sorted(seen)

    target.update(_target_metrics(raw_target, by_role["target"], config))
    if target["completed"] and target["reward_variance"] is None:
        _issue(target, "error", "reward variance exceeds finite report precision")
    for message in target["telemetry_errors"]:
        _issue(target, "error", message)
    if len(by_role["target"]) < config["target_attempts"]:
        _issue(target, "skip", "target attempt coverage incomplete")
    return _finish(checks)
