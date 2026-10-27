"""Seed forwarding, independent records and bounded fresh replay comparisons."""

import json
import math
import statistics
import time

from ...report import CheckResult
from ...types import CheckStatus
from .basic import _RuntimeGrader

JUDGED_REPLAYS = 20


def _trace(evidence, *, operation=None):
    trace = [
        {
            "operation": row.operation,
            "request": json.loads(row.request_json),
            "response": json.loads(row.response_json),
        }
        for row in evidence.exchanges
        if operation is None or row.operation == operation
    ]
    for row in trace:
        json.dumps(row, allow_nan=False)
        operation, request, response = (
            row["operation"],
            row["request"],
            row["response"],
        )
        if (
            operation not in {"reset", "step", "state"}
            or not isinstance(request, dict)
            or request.get("type") != operation
            or not isinstance(response, dict)
            or response.get("type")
            != ("state" if operation == "state" else "observation")
            or not isinstance(response.get("data"), dict)
        ):
            raise ValueError("malformed trace envelope")
    return trace


def _difference(expected, observed, path="$"):
    """First differing location, exposing indices and key ordinals only."""
    if type(expected) is not type(observed):
        return path
    if isinstance(expected, dict):
        for index, key in enumerate(sorted(expected.keys() | observed.keys())):
            child = f"{path}.<field:{index}>"
            if key not in expected or key not in observed:
                return child
            mismatch = _difference(expected[key], observed[key], child)
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        for index, (left, right) in enumerate(zip(expected, observed)):
            mismatch = _difference(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        if len(expected) != len(observed):
            return f"{path}[{min(len(expected), len(observed))}]"
    elif expected != observed:
        return path
    return None


def _telemetry(evidence):
    value = json.loads(evidence.telemetry_json)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError("malformed telemetry")
    if value["schema_version"] != 1:
        raise ValueError("unsupported telemetry version")
    return value


def _missing_telemetry(grader, evidence):
    if evidence is not None and evidence.failure_reason:
        # The shared runtime dependency reports incomplete primary collection.
        return None
    if evidence is not None and evidence.telemetry_json is None:
        failure = evidence.telemetry_error
        return CheckResult(
            check_id=grader.check_id,
            status=CheckStatus.FAIL if failure else CheckStatus.SKIP,
            evidence=[
                failure
                or evidence.telemetry_unavailable
                or "session telemetry is unavailable"
            ],
            duration_s=0,
        )
    return None


class SeedControlGrader(_RuntimeGrader):
    """Require observed seed forwarding for the original and scheduled resets."""

    check_id = "runtime.seed_control"

    def run(self, subject):
        evidence = subject.runtime_evidence
        result = _missing_telemetry(self, evidence) or super().run(subject)
        if result.status is CheckStatus.PASS:
            seed_replays = [
                replay.evidence for replay in evidence.replays if replay.scope == "seed"
            ]
            if not seed_replays:
                return result.model_copy(
                    update={
                        "status": CheckStatus.SKIP,
                        "evidence": [
                            evidence.replay_failure_reason
                            or "different-seed reset experiment is unavailable"
                        ],
                    }
                )
            unavailable = [
                sample.telemetry_unavailable
                for sample in seed_replays
                if sample.telemetry_unavailable
            ]
            if unavailable:
                return result.model_copy(
                    update={"status": CheckStatus.SKIP, "evidence": unavailable}
                )
            if any(
                sample.failure_reason and sample.telemetry_json is None
                for sample in seed_replays
            ):
                return result.model_copy(
                    update={
                        "status": CheckStatus.SKIP,
                        "evidence": [
                            "different-seed reset experiment did not produce seed telemetry"
                        ],
                    }
                )
        if result.status is CheckStatus.PASS:
            return result.model_copy(
                update={
                    "evidence": ["requested seeds were observed as forwarded to reset"]
                }
            )
        return result

    def check(self, subject, evidence):
        problems = []
        original_seed = None
        for index, sample in enumerate(
            [evidence]
            + [replay.evidence for replay in evidence.replays if replay.scope == "seed"]
        ):
            if sample.telemetry_error:
                problems.append(f"replay {index}: telemetry collection failed")
                continue
            if sample.telemetry_unavailable:
                continue
            incomplete = sample.failure_reason and sample.telemetry_json is None
            if incomplete:
                # Failed later actions leave seed forwarding unproven; an
                # observed reset rejection remains a seed experiment failure.
                _trace(sample, operation="reset")
            resets = [row for row in sample.exchanges if row.operation == "reset"]
            if incomplete and not resets:
                continue
            if len(resets) != 1:
                problems.append(f"replay {index}: expected one measured reset")
                continue
            seed = json.loads(resets[0].request_json)["data"]["seed"]
            if type(seed) is not int:
                problems.append(f"replay {index}: requested seed is not an integer")
                continue
            if index == 0:
                original_seed = seed
            elif seed == original_seed:
                problems.append(f"replay {index}: scheduled seed was not changed")
            if incomplete:
                continue
            if sample.telemetry_json is None:
                problems.append(f"replay {index}: seed telemetry unavailable")
                continue
            observed = _telemetry(sample).get("seed")
            if (
                not isinstance(observed, dict)
                or observed.get("requested") is not True
                or observed.get("accepted") is not True
                or type(observed.get("value")) is not int
                or observed["value"] != seed
            ):
                problems.append(f"replay {index}: seed was not observed as forwarded")
        return problems


class TrajectoryRecordGrader(_RuntimeGrader):
    """Compare a subject-emitted record with the independent collector transcript."""

    check_id = "runtime.trajectory_record"

    def run(self, subject):
        return _missing_telemetry(self, subject.runtime_evidence) or super().run(
            subject
        )

    def check(self, subject, evidence):
        if evidence.telemetry_error:
            return [evidence.telemetry_error]
        record = _telemetry(evidence).get("trajectory")
        if not isinstance(record, dict):
            return ["subject-emitted trajectory record is missing"]
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
            or record.get("source") != "openenv-server"
        ):
            return ["subject-emitted trajectory record metadata is invalid"]
        if record.get("complete") is not True:
            return ["subject-emitted trajectory record is incomplete"]
        if not isinstance(record.get("records"), list):
            return ["subject-emitted trajectory records are malformed"]
        mismatch = _difference(_trace(evidence), record["records"])
        return [f"subject record differs at {mismatch}"] if mismatch else []


class EpisodeDeterminismGrader(_RuntimeGrader):
    """Compare full traces, or per-step reward population variance for a judge."""

    check_id = "runtime.episode_determinism"

    def run(self, subject):
        started = time.monotonic()
        evidence = subject.runtime_evidence
        if evidence is None:
            return super().run(subject)
        measured = {}
        try:
            status, messages, measured = self._grade(subject, evidence)
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            status, messages = CheckStatus.FAIL, ["malformed replay evidence"]
        return CheckResult(
            check_id=self.check_id,
            status=status,
            evidence=messages,
            measured=measured,
            duration_s=time.monotonic() - started,
        )

    def _grade(self, subject, evidence):
        if any(
            row.scope not in {"session", "container", "seed"}
            for row in evidence.replays
        ):
            return CheckStatus.FAIL, ["invalid replay scope"], {}
        replays = [row for row in evidence.replays if row.scope != "seed"]
        samples = [evidence] + [row.evidence for row in replays]
        completed = [
            not row.failure_reason
            and not row.failure_phase
            and any(exchange.operation == "step" for exchange in row.exchanges)
            for row in samples
        ]
        measured = {"completed_replays": sum(completed)}
        judged = subject.manifest.capabilities.llm_judged
        required = JUDGED_REPLAYS if judged else 3
        # Inspect retained content before reporting an incomplete schedule: a
        # timeout cannot conceal malformed wire data or an observed divergence.
        traces = [_trace(sample) for sample in samples]
        rewards = []
        for trace in traces:
            sample_rewards = []
            for row in trace:
                if row["operation"] != "step":
                    continue
                if judged:
                    data = row["response"]["data"]
                    reward = data["reward"]
                    if reward is None and data.get("done") is False:
                        sample_rewards.append(None)
                        continue
                    low, high = subject.manifest.reward.range
                    if (
                        type(reward) not in (int, float)
                        or not math.isfinite(reward)
                        or not low <= reward <= high
                    ):
                        return (
                            CheckStatus.FAIL,
                            ["invalid judged replay reward"],
                            measured,
                        )
                    sample_rewards.append(reward)
                    # Keep null versus numeric positions visible to trace comparison.
                    data["reward"] = 0.0
            rewards.append(sample_rewards)
        # Policy-owned volatile exclusions are deliberately empty. All replays
        # use the same episode identity; no author field can suppress a difference.
        for index, trace in enumerate(traces[1:], 1):
            if completed[0] and completed[index]:
                mismatch = _difference(traces[0], trace)
            else:
                shared = min(len(traces[0]), len(trace))
                mismatch = _difference(traces[0][:shared], trace[:shared])
            if mismatch:
                return (
                    CheckStatus.FAIL,
                    [f"replay {index}: first divergence at {mismatch}"],
                    measured,
                )
        if (
            not all(completed)
            or len(samples) < required
            or {row.scope for row in replays} != {"session", "container"}
        ):
            return (
                CheckStatus.SKIP,
                [
                    evidence.replay_failure_reason
                    or f"requires {required} completed replays across fresh sessions and containers"
                ],
                measured,
            )
        if judged and len(samples) != JUDGED_REPLAYS:
            return (
                CheckStatus.FAIL,
                ["judged procedure requires exactly 20 samples"],
                measured,
            )
        if judged:
            variances = [
                None if values[0] is None else statistics.pvariance(values)
                for values in zip(*rewards)
            ]
            measured.update(
                reward_population_variance=variances,
                variance_units="reward_squared",
            )
            bound = subject.manifest.reward.variance_tolerance
            if type(bound) not in (int, float) or not math.isfinite(bound) or bound < 0:
                return CheckStatus.FAIL, ["invalid declared variance bound"], measured
            for index, variance in enumerate(variances):
                if variance is not None and variance > bound:
                    return (
                        CheckStatus.FAIL,
                        [
                            f"step {index}: reward population variance exceeds declared bound"
                        ],
                        measured,
                    )
            if not any(variance is not None for variance in variances):
                return (
                    CheckStatus.SKIP,
                    ["missing prerequisite: an observed numeric step reward"],
                    measured,
                )
        if any(
            row.scope == "container" and row.cleanup_complete is not True
            for row in replays
        ):
            return (
                CheckStatus.SKIP,
                ["fresh container cleanup was not confirmed"],
                measured,
            )
        return (
            CheckStatus.PASS,
            ["fresh-session and fresh-container replays agree"],
            measured,
        )
