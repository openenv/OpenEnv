import json
from dataclasses import replace

import pytest
from openenv.validation.graders.runtime.repeatability import (
    EpisodeDeterminismGrader,
    SeedControlGrader,
    TrajectoryRecordGrader,
)
from openenv.validation.manifest import JudgePin
from openenv.validation.runtime.contracts import ReplayEvidence
from openenv.validation.types import CheckStatus
from test_runtime_grading import good_rows, mutate_response, subject_with

GRADERS = [SeedControlGrader, EpisodeDeterminismGrader, TrajectoryRecordGrader]


def sample(subject, *, seed=42, reward=1.0):
    rows = good_rows()
    request = json.loads(rows[0].request_json)
    request["data"]["seed"] = seed
    rows[0] = replace(rows[0], request_json=json.dumps(request))
    mutate_response(rows, 2, lambda response: response["data"].update(reward=reward))
    telemetry = {
        "schema_version": 1,
        "seed": {"requested": True, "accepted": True, "value": seed},
        "trajectory": {
            "schema_version": 1,
            "source": "openenv-server",
            "complete": True,
            "reason": None,
            "records": [
                {
                    "operation": row.operation,
                    "request": json.loads(row.request_json),
                    "response": json.loads(row.response_json),
                }
                for row in rows
            ],
        },
    }
    return replace(
        subject.runtime_evidence,
        exchanges=tuple(rows),
        telemetry_json=json.dumps(telemetry),
        replays=(),
    )


def subject_with_replays(tmp_path, *, judged=False):
    subject = subject_with(tmp_path)
    count = 20 if judged else 3
    replays = tuple(
        ReplayEvidence(
            "container" if index == 1 else "session",
            sample(subject),
            cleanup_complete=True if index == 1 else None,
        )
        for index in range(1, count)
    ) + (ReplayEvidence("seed", sample(subject, seed=43)),)
    if judged:
        subject.manifest.capabilities.llm_judged = True
        subject.manifest.judge = JudgePin(model="test-judge", version="1")
        subject.manifest.reward.variance_tolerance = 0.2
    return replace(subject, runtime_evidence=replace(sample(subject), replays=replays))


def alter_telemetry(evidence, mutate):
    value = json.loads(evidence.telemetry_json)
    mutate(value)
    return replace(evidence, telemetry_json=json.dumps(value))


def step_token(evidence, token, location):
    """Insert a literal JSON token without rounding it in the test fixture."""
    rows = list(evidence.exchanges)
    attribute = "request_json" if location == "request" else "response_json"
    value = json.loads(
        rows[2].request_json if location == "request" else rows[2].response_json
    )
    if location == "reward":
        value["data"]["reward"] = "EXACT_TOKEN"
    else:
        value["data"]["precision_probe"] = "EXACT_TOKEN"
    payload = json.dumps(value).replace('"EXACT_TOKEN"', token)
    rows[2] = replace(rows[2], **{attribute: payload})
    return replace(evidence, exchanges=tuple(rows))


def record_exchanges(evidence):
    value = json.loads(evidence.telemetry_json)
    value["trajectory"]["records"] = "EXACT_RECORDS"
    records = (
        "["
        + ",".join(
            '{"operation":'
            + json.dumps(row.operation)
            + ',"request":'
            + row.request_json
            + ',"response":'
            + row.response_json
            + "}"
            for row in evidence.exchanges
        )
        + "]"
    )
    return json.dumps(value).replace('"EXACT_RECORDS"', records)


@pytest.mark.parametrize("grader", [EpisodeDeterminismGrader, TrajectoryRecordGrader])
@pytest.mark.parametrize("location", ["request", "response"])
@pytest.mark.parametrize(
    "left,right,status",
    [
        ("1.00000000000000001", "1.00000000000000002", CheckStatus.FAIL),
        ("1e-999", "0.0", CheckStatus.FAIL),
        ("1", "1.0", CheckStatus.FAIL),
        ("true", "1", CheckStatus.FAIL),
        ('"1.0"', "1.0", CheckStatus.FAIL),
        ("null", "0", CheckStatus.FAIL),
        ("[]", "{}", CheckStatus.FAIL),
        ("1.00000000000000001", "1.00000000000000001", CheckStatus.PASS),
        ("1.00", "1e0", CheckStatus.PASS),
        ("1e-999", "10e-1000", CheckStatus.PASS),
        ("-0.0", "0.0", CheckStatus.PASS),
        ('{"a":[1.00],"b":true}', '{"b":true,"a":[1e0]}', CheckStatus.PASS),
    ],
)
def test_exact_json_values_in_replays_and_subject_records(
    tmp_path, grader, location, left, right, status
):
    subject = subject_with_replays(tmp_path)
    original = subject.runtime_evidence
    evidence = step_token(original, left, location)
    other = step_token(original, right, location)
    if grader is EpisodeDeterminismGrader:
        evidence = replace(
            evidence,
            replays=tuple(
                replace(row, evidence=other) if row.scope != "seed" else row
                for row in original.replays
            ),
        )
    else:
        evidence = replace(evidence, telemetry_json=record_exchanges(other))
    result = grader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is status
    if status is CheckStatus.FAIL:
        assert "divergence" in result.evidence[0] or "differs at" in result.evidence[0]


@pytest.mark.parametrize(
    "left,right,limits,bound,status",
    [
        ("1", "1.0", (0.0, 1.0), 0.0, CheckStatus.PASS),
        ("0.1", "0.7", (0.1, 0.7), 0.09, CheckStatus.PASS),
        ("1e-999", "10e-1000", (0.0, 1.0), 0.0, CheckStatus.PASS),
        ("1.00000000000000001", "1.0", (0.0, 1.0), 0.2, CheckStatus.FAIL),
        ("-1e-999", "0.0", (0.0, 1.0), 0.2, CheckStatus.FAIL),
        (
            "0.50000000000000001",
            "0.50000000000000002",
            (0.0, 1.0),
            0.0,
            CheckStatus.FAIL,
        ),
        ("1e-999", "0.0", (0.0, 1.0), 0.0, CheckStatus.FAIL),
        (
            "0.1",
            "0.70000000000000000000000000000001",
            (0.0, 1.0),
            0.09,
            CheckStatus.FAIL,
        ),
    ],
)
def test_exact_judged_reward_range_and_variance(
    tmp_path, left, right, limits, bound, status
):
    subject = subject_with_replays(tmp_path, judged=True)
    subject.manifest.reward.range = limits
    subject.manifest.reward.variance_tolerance = bound
    original = subject.runtime_evidence
    evidence = step_token(original, left, "reward")
    evidence = replace(
        evidence,
        replays=tuple(
            replace(
                row,
                evidence=step_token(
                    row.evidence, left if index < 9 else right, "reward"
                ),
            )
            if row.scope != "seed"
            else row
            for index, row in enumerate(original.replays)
        ),
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is status
    if status is CheckStatus.PASS:
        assert result.measured["completed_replays"] == 20
        # Report measurements retain their numeric JSON contract.
        assert (
            type(
                json.loads(result.model_dump_json())["measured"][
                    "reward_population_variance"
                ][0]
            )
            is float
        )
    elif left in {"1.00000000000000001", "-1e-999"}:
        assert result.evidence == ["invalid judged replay reward"]
    else:
        assert result.evidence == [
            "step 0: reward population variance exceeds declared bound"
        ]


@pytest.mark.parametrize(
    "token", ["true", '"0.5"', "[]", "{}", "NaN", "Infinity", "-Infinity"]
)
def test_judged_rewards_remain_finite_numeric_scalars(tmp_path, token):
    subject = subject_with_replays(tmp_path, judged=True)
    evidence = step_token(subject.runtime_evidence, token, "reward")
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL


@pytest.mark.parametrize("bound", [None, True, "0.2", -0.1, float("nan"), float("inf")])
def test_judged_variance_bound_remains_validated(tmp_path, bound):
    subject = subject_with_replays(tmp_path, judged=True)
    subject.manifest.reward.variance_tolerance = bound
    result = EpisodeDeterminismGrader().run(subject)
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["invalid declared variance bound"]


@pytest.mark.parametrize("grader", [EpisodeDeterminismGrader, TrajectoryRecordGrader])
@pytest.mark.parametrize(
    "token", ["NaN", "Infinity", "-Infinity", "1e-9999999999999999999999"]
)
def test_nonfinite_or_unsupported_trace_numbers_are_findings(tmp_path, grader, token):
    subject = subject_with_replays(tmp_path)
    evidence = step_token(subject.runtime_evidence, token, "response")
    result = grader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL


def test_unsupported_subject_record_number_is_a_finding(tmp_path):
    subject = subject_with_replays(tmp_path)
    other = step_token(subject.runtime_evidence, "1e9999999999999999999999", "response")
    evidence = replace(subject.runtime_evidence, telemetry_json=record_exchanges(other))
    result = TrajectoryRecordGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL


@pytest.mark.parametrize("fault", [None, "diverged", "invalid_reward", "invalid_bound"])
def test_judged_arithmetic_is_bounded_without_hiding_failures(tmp_path, fault):
    subject = subject_with_replays(tmp_path, judged=True)
    original = subject.runtime_evidence
    evidence = step_token(original, "1e-1000000000", "reward")
    evidence = replace(
        evidence,
        replays=tuple(
            replace(row, evidence=step_token(row.evidence, "1e-1000000000", "reward"))
            for row in original.replays
        ),
    )
    if fault in {"diverged", "invalid_reward"}:
        replay = evidence.replays[0]
        changed = step_token(
            replay.evidence,
            "true" if fault == "invalid_reward" else "2",
            "reward" if fault == "invalid_reward" else "response",
        )
        evidence = replace(
            evidence,
            replays=(replace(replay, evidence=changed),) + evidence.replays[1:],
        )
    elif fault == "invalid_bound":
        subject.manifest.reward.variance_tolerance = -0.1
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is (CheckStatus.FAIL if fault else CheckStatus.SKIP)
    if fault is None:
        assert result.evidence == [
            "judged reward precision exceeds supported arithmetic bounds"
        ]
        assert result.measured == {"completed_replays": 20}


def with_reward_columns(subject, columns):
    """Supply ten samples of each literal token at each numeric step position."""

    def changed(evidence, index):
        rows = list(evidence.exchanges[:2])
        for position, tokens in enumerate(columns, 1):
            step_rows = mutate_response(
                list(evidence.exchanges),
                2,
                lambda value: value["data"]["observation"].update(counter=position),
            )
            mutate_response(
                step_rows, 3, lambda value: value["data"].update(step_count=position)
            )
            step = step_token(
                replace(evidence, exchanges=tuple(step_rows)),
                tokens[index >= 10],
                "reward",
            )
            rows.extend(step.exchanges[2:])
        return replace(evidence, exchanges=tuple(rows))

    original = subject.runtime_evidence
    evidence = replace(
        changed(original, 0),
        replays=tuple(
            replace(row, evidence=changed(row.evidence, index + 1))
            if row.scope != "seed"
            else row
            for index, row in enumerate(original.replays)
        ),
    )
    return replace(subject, runtime_evidence=evidence)


@pytest.mark.parametrize("unsupported_first", [False, True])
@pytest.mark.parametrize(
    "right,constant,status",
    [
        ("1.0", "1e-1000000000", CheckStatus.FAIL),
        ("0.0", "1e-1000000000", CheckStatus.SKIP),
        ("1.0", "1e-9999", CheckStatus.FAIL),
        ("0.0", "1e-9999", CheckStatus.PASS),
    ],
)
def test_supported_step_variance_is_checked_before_arithmetic_skip(
    tmp_path, unsupported_first, right, constant, status
):
    subject = subject_with_replays(tmp_path, judged=True)
    subject.manifest.reward.variance_tolerance = 0.0
    columns = [("0.0", right), (constant, constant)]
    if unsupported_first:
        columns.reverse()
    result = EpisodeDeterminismGrader().run(with_reward_columns(subject, columns))
    assert result.status is status
    assert result.measured["completed_replays"] == 20
    if status is CheckStatus.FAIL:
        index = int(unsupported_first)
        assert result.evidence == [
            f"step {index}: reward population variance exceeds declared bound"
        ]
        assert result.measured["reward_population_variance"][index] == 0.25
    elif status is CheckStatus.PASS:
        assert result.measured["reward_population_variance"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "left,right,status",
    [
        (str(-(10**308)), str(10**308), CheckStatus.FAIL),
        ("-1e308", "1e308", CheckStatus.FAIL),
        ("1e308", "1e308", CheckStatus.PASS),
    ],
    ids=["integer_tokens", "decimal_tokens", "constant_decimal_tokens"],
)
def test_extreme_variance_finding_survives_unrepresentable_display(
    tmp_path, left, right, status
):
    subject = subject_with_replays(tmp_path, judged=True)
    subject.manifest.reward.range = (-1e308, 1e308)
    subject.manifest.reward.variance_tolerance = 1e308
    subject = with_reward_columns(subject, [(left, right)])
    result = EpisodeDeterminismGrader().run(subject)
    assert result.status is status
    assert result.measured["completed_replays"] == 20
    if status is CheckStatus.FAIL:
        assert result.evidence == [
            "step 0: reward population variance exceeds declared bound"
        ]
        assert result.measured["reward_population_variance"] == [None]
        assert result.measured["reward_population_variance_unavailable"] == {
            "0": "exceeds finite float display range"
        }
        # Missing display values are explicit JSON nulls, never fabricated numbers.
        saved = json.loads(result.model_dump_json())
        assert saved["measured"] == result.measured
    else:
        assert result.measured["reward_population_variance"] == [0.0]
        assert "reward_population_variance_unavailable" not in result.measured


@pytest.mark.parametrize("grader", GRADERS)
def test_seed_invariant_fresh_replays_and_independent_record_pass(tmp_path, grader):
    assert grader().run(subject_with_replays(tmp_path)).status is CheckStatus.PASS


@pytest.mark.parametrize("grader", GRADERS)
def test_missing_runtime_evidence_is_incomplete(tmp_path, grader):
    subject = replace(subject_with_replays(tmp_path), runtime_evidence=None)
    assert grader().run(subject).status is CheckStatus.SKIP


@pytest.mark.parametrize("grader", [SeedControlGrader, TrajectoryRecordGrader])
def test_unavailable_telemetry_differs_from_failed_telemetry(tmp_path, grader):
    subject = subject_with_replays(tmp_path)
    evidence = replace(subject.runtime_evidence, telemetry_json=None)
    assert (
        grader().run(replace(subject, runtime_evidence=evidence)).status
        is CheckStatus.SKIP
    )
    evidence = replace(evidence, telemetry_error="telemetry failed (PermissionError)")
    result = grader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL
    assert "PermissionError" in result.evidence[0]


@pytest.mark.parametrize(
    "field,value",
    [("accepted", False), ("value", 99), ("requested", False), ("value", True)],
)
def test_rejected_dropped_or_wrong_seed_fails(tmp_path, field, value):
    subject = subject_with_replays(tmp_path)
    evidence = alter_telemetry(
        subject.runtime_evidence, lambda data: data["seed"].update({field: value})
    )
    result = SeedControlGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL
    assert "seed was not observed as forwarded" in result.evidence[0]


def test_different_seed_schedule_is_required_but_output_need_not_differ(tmp_path):
    subject = subject_with_replays(tmp_path)
    evidence = replace(
        subject.runtime_evidence, replays=subject.runtime_evidence.replays[:-1]
    )
    assert (
        SeedControlGrader().run(replace(subject, runtime_evidence=evidence)).status
        is CheckStatus.SKIP
    )
    evidence = replace(
        evidence, replays=evidence.replays + (ReplayEvidence("seed", sample(subject)),)
    )
    assert (
        SeedControlGrader().run(replace(subject, runtime_evidence=evidence)).status
        is CheckStatus.FAIL
    )


@pytest.mark.parametrize(
    "field,value", [("trajectory", None), ("schema_version", True)]
)
def test_missing_or_malformed_record_fails(tmp_path, field, value):
    subject = subject_with_replays(tmp_path)
    evidence = alter_telemetry(
        subject.runtime_evidence, lambda data: data.update({field: value})
    )
    assert (
        TrajectoryRecordGrader().run(replace(subject, runtime_evidence=evidence)).status
        is CheckStatus.FAIL
    )


@pytest.mark.parametrize("mutation", ["truncated", "changed", "incomplete"])
def test_subject_record_cannot_pass_by_using_validator_transcript_alone(
    tmp_path, mutation
):
    subject = subject_with_replays(tmp_path)

    def mutate(data):
        record = data["trajectory"]
        if mutation == "truncated":
            record["records"].pop()
        elif mutation == "changed":
            record["records"][2]["response"]["data"]["reward"] = 0.0
        else:
            record["complete"] = False

    evidence = alter_telemetry(subject.runtime_evidence, mutate)
    result = TrajectoryRecordGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL


@pytest.mark.parametrize(
    "field,value",
    [("reward", 0.0), ("done", True), ("observation", {"counter": "private-value"})],
)
def test_divergence_reports_first_path_without_values(tmp_path, field, value):
    subject = subject_with_replays(tmp_path)
    evidence = subject.runtime_evidence
    replay = evidence.replays[0]
    rows = mutate_response(
        list(replay.evidence.exchanges),
        2,
        lambda data: data["data"].update({field: value}),
    )
    replay = replace(replay, evidence=replace(replay.evidence, exchanges=tuple(rows)))
    evidence = replace(evidence, replays=(replay,) + evidence.replays[1:])
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL
    field_index = {"done": 0, "observation": 1, "reward": 2}[field]
    assert f"$[2].<field:2>.<field:0>.<field:{field_index}>" in result.evidence[0]
    assert "private-value" not in result.model_dump_json()


def test_fresh_sessions_without_fresh_container_are_incomplete(tmp_path):
    subject = subject_with_replays(tmp_path)
    evidence = replace(
        subject.runtime_evidence,
        replays=tuple(
            replace(row, scope="session") if row.scope != "seed" else row
            for row in subject.runtime_evidence.replays
        ),
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.SKIP


@pytest.mark.parametrize("cleanup_complete", [False, None])
@pytest.mark.parametrize("judged", [False, True])
@pytest.mark.parametrize("fault", [None, "malformed", "diverged"])
def test_unconfirmed_container_cleanup_prevents_pass_without_hiding_failures(
    tmp_path, cleanup_complete, judged, fault
):
    subject = subject_with_replays(tmp_path, judged=judged)
    evidence = subject.runtime_evidence
    replay = evidence.replays[0]
    rows = list(replay.evidence.exchanges)
    if fault == "malformed":
        rows[2] = replace(rows[2], response_json="{not-json")
    elif fault == "diverged":
        rows = mutate_response(
            rows, 2, lambda value: value["data"]["observation"].update(counter=99)
        )
    replay = replace(
        replay,
        evidence=replace(replay.evidence, exchanges=tuple(rows)),
        cleanup_complete=cleanup_complete,
    )
    evidence = replace(
        evidence,
        replays=(replay,) + evidence.replays[1:],
        replay_failure_reason="fresh container cleanup failed"
        if cleanup_complete is False
        else None,
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is (CheckStatus.FAIL if fault else CheckStatus.SKIP)
    if fault is None:
        assert result.measured["completed_replays"] == (20 if judged else 3)
        assert result.evidence == ["fresh container cleanup was not confirmed"]
    elif fault == "diverged":
        assert "first divergence" in result.evidence[0]
    else:
        assert result.evidence == ["malformed replay evidence"]


def test_partial_judged_sample_is_incomplete(tmp_path):
    subject = subject_with_replays(tmp_path, judged=True)
    evidence = replace(
        subject.runtime_evidence, replays=subject.runtime_evidence.replays[:18]
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.SKIP
    assert result.measured["completed_replays"] == 19


@pytest.mark.parametrize(
    "low,bound,status,cleanup_complete",
    [
        (0.5, 0.1, CheckStatus.PASS, True),
        (0.5, 0.1, CheckStatus.SKIP, False),
        (0.0, 0.2, CheckStatus.FAIL, True),
        (0.0, 0.2, CheckStatus.FAIL, False),
    ],
)
def test_judged_population_variance_uses_reward_squared_units(
    tmp_path, low, bound, status, cleanup_complete
):
    subject = subject_with_replays(tmp_path, judged=True)
    subject.manifest.reward.variance_tolerance = bound
    evidence = subject.runtime_evidence
    replays = tuple(
        replace(
            row,
            evidence=sample(subject, reward=low if index < 10 else 1.0),
            cleanup_complete=cleanup_complete if row.scope == "container" else None,
        )
        for index, row in enumerate(evidence.replays[:-1])
    ) + (evidence.replays[-1],)
    result = EpisodeDeterminismGrader().run(
        replace(subject, runtime_evidence=replace(evidence, replays=replays))
    )
    assert result.status is status
    assert result.measured["variance_units"] == "reward_squared"
    assert result.measured["reward_population_variance"] == [
        pytest.approx((1.0 - low) ** 2 / 4)
    ]


def test_judged_mode_still_checks_observations_and_state(tmp_path):
    subject = subject_with_replays(tmp_path, judged=True)
    evidence = subject.runtime_evidence
    replay = evidence.replays[0]
    rows = mutate_response(
        list(replay.evidence.exchanges),
        3,
        lambda data: data["data"].update(step_count=2),
    )
    replay = replace(replay, evidence=replace(replay.evidence, exchanges=tuple(rows)))
    evidence = replace(evidence, replays=(replay,) + evidence.replays[1:])
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL
    assert "$[3].<field:2>.<field:0>.<field:1>" in result.evidence[0]


def test_judged_variance_does_not_pool_different_steps(tmp_path):
    subject = subject_with_replays(tmp_path, judged=True)
    subject.manifest.reward.variance_tolerance = 0.0

    def two_steps(evidence):
        rows = list(evidence.exchanges)
        second = mutate_response(
            [rows[2]], 0, lambda data: data["data"].update(reward=0.0)
        )[0]
        state = mutate_response(
            [rows[3]], 0, lambda data: data["data"].update(step_count=2)
        )[0]
        return replace(evidence, exchanges=tuple(rows + [second, state]))

    evidence = two_steps(subject.runtime_evidence)
    evidence = replace(
        evidence,
        replays=tuple(
            replace(row, evidence=two_steps(row.evidence)) for row in evidence.replays
        ),
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.PASS
    assert result.measured["reward_population_variance"] == [0.0, 0.0]


@pytest.mark.parametrize("scored", [True, False])
def test_judged_replays_allow_consistent_unscored_steps(tmp_path, scored):
    subject = subject_with_replays(tmp_path, judged=True)

    def unscored(evidence):
        rows = list(evidence.exchanges)
        prefix = mutate_response(
            [rows[2]], 0, lambda response: response["data"].update(reward=None)
        )[0]
        return replace(
            evidence,
            exchanges=tuple(
                rows[:2] + [prefix, rows[3]] + (rows[2:] if scored else [])
            ),
        )

    evidence = unscored(subject.runtime_evidence)
    evidence = replace(
        evidence,
        replays=tuple(
            replace(row, evidence=unscored(row.evidence)) for row in evidence.replays
        ),
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is (CheckStatus.PASS if scored else CheckStatus.SKIP)
    assert result.measured["reward_population_variance"] == (
        [None, 0.0] if scored else [None]
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_judged_null_reward_cannot_hide_changed_scoring_or_terminal_failure(
    tmp_path, terminal
):
    subject = subject_with_replays(tmp_path, judged=True)
    evidence = subject.runtime_evidence
    replay = evidence.replays[0]
    rows = mutate_response(
        list(replay.evidence.exchanges),
        2,
        lambda response: response["data"].update(reward=None, done=terminal),
    )
    replay = replace(replay, evidence=replace(replay.evidence, exchanges=tuple(rows)))
    evidence = replace(evidence, replays=(replay,) + evidence.replays[1:])
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL


@pytest.mark.parametrize("grader", [SeedControlGrader, TrajectoryRecordGrader])
def test_malformed_telemetry_is_a_finding(tmp_path, grader):
    subject = subject_with_replays(tmp_path)
    evidence = replace(subject.runtime_evidence, telemetry_json="{not-json")
    assert (
        grader().run(replace(subject, runtime_evidence=evidence)).status
        is CheckStatus.FAIL
    )


@pytest.mark.parametrize("grader", GRADERS)
@pytest.mark.parametrize("telemetry_available", [True, False])
def test_truncated_primary_evidence_never_passes(tmp_path, grader, telemetry_available):
    subject = subject_with_replays(tmp_path)
    evidence = replace(
        subject.runtime_evidence,
        failure_reason="step failed (TimeoutError)",
        telemetry_json=subject.runtime_evidence.telemetry_json
        if telemetry_available
        else None,
    )
    result = grader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.SKIP
    assert result.evidence


@pytest.mark.parametrize("scope", ["session", "container"])
def test_seed_proof_is_independent_of_other_replay_collection_failures(tmp_path, scope):
    subject = subject_with_replays(tmp_path)
    evidence = subject.runtime_evidence
    replays = tuple(
        replace(
            replay,
            evidence=replace(
                replay.evidence,
                failure_reason="step failed (TimeoutError)",
                telemetry_json=None,
                telemetry_error="unrelated replay telemetry unavailable",
            ),
        )
        if replay.scope == scope
        else replay
        for replay in evidence.replays
    )
    result = SeedControlGrader().run(
        replace(subject, runtime_evidence=replace(evidence, replays=replays))
    )
    assert result.status is CheckStatus.PASS


def test_changed_seed_rejection_still_fails_seed_control(tmp_path):
    subject = subject_with_replays(tmp_path)
    evidence = subject.runtime_evidence
    changed = evidence.replays[-1]
    changed = replace(
        changed,
        evidence=alter_telemetry(
            changed.evidence, lambda value: value["seed"].update(accepted=False)
        ),
    )
    evidence = replace(evidence, replays=evidence.replays[:-1] + (changed,))
    assert (
        SeedControlGrader().run(replace(subject, runtime_evidence=evidence)).status
        is CheckStatus.FAIL
    )


@pytest.mark.parametrize(
    "original_accepted,telemetry_error",
    [(True, None), (False, None), (True, "session telemetry failed (ValueError)")],
)
@pytest.mark.parametrize("step_error", [False, True])
def test_incomplete_changed_seed_probe_does_not_claim_a_seed_violation(
    tmp_path, original_accepted, telemetry_error, step_error
):
    subject = subject_with_replays(tmp_path)
    evidence = alter_telemetry(
        subject.runtime_evidence,
        lambda value: value["seed"].update(accepted=original_accepted),
    )
    changed = evidence.replays[-1]
    rows = changed.evidence.exchanges[:2]
    if step_error:
        rows += (
            replace(
                changed.evidence.exchanges[2],
                response_json=json.dumps(
                    {"type": "error", "data": {"code": "STEP_ERROR"}}
                ),
            ),
        )
    changed = replace(
        changed,
        evidence=replace(
            changed.evidence,
            exchanges=rows,
            failure_phase="step",
            failure_reason="step failed (STEP_ERROR)"
            if step_error
            else "step failed (TimeoutError)",
            telemetry_json=None,
            telemetry_error=telemetry_error,
        ),
    )
    evidence = replace(evidence, replays=evidence.replays[:-1] + (changed,))
    result = SeedControlGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is (
        CheckStatus.SKIP
        if original_accepted and telemetry_error is None
        else CheckStatus.FAIL
    )
    if telemetry_error is not None:
        assert "telemetry collection failed" in result.evidence[0]
    elif original_accepted:
        assert result.evidence == [
            "different-seed reset experiment did not produce seed telemetry"
        ]
    else:
        assert "seed was not observed as forwarded" in result.evidence[0]


def test_rejected_reset_is_still_a_seed_failure_without_telemetry(tmp_path):
    subject = subject_with_replays(tmp_path)
    evidence = subject.runtime_evidence
    changed = evidence.replays[-1]
    rejected = replace(
        changed.evidence.exchanges[0],
        response_json=json.dumps({"type": "error", "data": {"code": "RESET_ERROR"}}),
    )
    changed = replace(
        changed,
        evidence=replace(
            changed.evidence,
            exchanges=(rejected,),
            failure_phase="reset",
            failure_reason="reset failed (RESET_ERROR)",
            telemetry_json=None,
        ),
    )
    evidence = replace(evidence, replays=evidence.replays[:-1] + (changed,))
    result = SeedControlGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["malformed runtime evidence"]


@pytest.mark.parametrize("judged", [False, True])
@pytest.mark.parametrize("fault", ["timeout", "malformed", "diverged"])
def test_incomplete_replay_preserves_observed_failures(tmp_path, judged, fault):
    subject = subject_with_replays(tmp_path, judged=judged)
    evidence = subject.runtime_evidence
    replay = evidence.replays[-2]
    rows = list(replay.evidence.exchanges[:3])
    if fault == "malformed":
        rows[2] = replace(rows[2], response_json="{not-json")
    elif fault == "diverged":
        rows = mutate_response(
            rows, 2, lambda value: value["data"]["observation"].update(counter=99)
        )
    partial = replace(
        replay.evidence,
        exchanges=tuple(rows),
        failure_phase="step",
        failure_reason="step failed (TimeoutError)",
    )
    replay = replace(replay, evidence=partial)
    evidence = replace(
        evidence,
        replays=evidence.replays[:-2] + (replay, evidence.replays[-1]),
        replay_failure_reason="total replay deadline exceeded",
    )
    result = EpisodeDeterminismGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is (
        CheckStatus.SKIP if fault == "timeout" else CheckStatus.FAIL
    )
    if fault == "timeout":
        assert result.measured["completed_replays"] == (19 if judged else 2)
        assert "reward_population_variance" not in result.measured
    elif fault == "diverged":
        assert "$[2].<field:2>.<field:0>.<field:1>.<field:0>" in result.evidence[0]


@pytest.mark.parametrize("grader", [EpisodeDeterminismGrader, TrajectoryRecordGrader])
@pytest.mark.parametrize("mismatch", ["extra_key", "changed_value"])
def test_public_divergence_paths_never_disclose_subject_keys(
    tmp_path, grader, mismatch
):
    subject = subject_with_replays(tmp_path)
    evidence = subject.runtime_evidence
    private_key = "hf_privateSubjectSecret123\nUNTRUSTED\r\tkey"
    if mismatch == "changed_value":
        rows = mutate_response(
            list(evidence.exchanges),
            2,
            lambda response: response["data"]["observation"].update({private_key: 0}),
        )
        evidence = replace(evidence, exchanges=tuple(rows))
    if grader is EpisodeDeterminismGrader:
        replay = evidence.replays[0]
        rows = mutate_response(
            list(replay.evidence.exchanges),
            2,
            lambda response: response["data"]["observation"].update({private_key: 1}),
        )
        replay = replace(
            replay, evidence=replace(replay.evidence, exchanges=tuple(rows))
        )
        evidence = replace(evidence, replays=(replay,) + evidence.replays[1:])
    else:
        evidence = alter_telemetry(
            evidence,
            lambda value: value["trajectory"]["records"][2]["response"]["data"][
                "observation"
            ].update({private_key: 1}),
        )
    result = grader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.FAIL
    assert "$[2].<field:2>.<field:0>.<field:1>.<field:1>" in result.evidence[0]
    assert private_key not in "".join(result.evidence)
    assert "privateSubjectSecret" not in result.model_dump_json()
    assert "UNTRUSTED" not in result.model_dump_json()
    assert "\n" not in result.evidence[0]
    assert "\r" not in result.evidence[0]
    assert "\t" not in result.evidence[0]


@pytest.mark.parametrize("grader", [SeedControlGrader, TrajectoryRecordGrader])
def test_explicit_unsupported_telemetry_skips_with_capability_named(tmp_path, grader):
    subject = subject_with_replays(tmp_path)
    evidence = replace(
        subject.runtime_evidence,
        telemetry_json=None,
        telemetry_unavailable="unsupported capability: session_telemetry",
    )
    result = grader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.SKIP
    assert result.evidence == ["unsupported capability: session_telemetry"]


def test_unsupported_changed_seed_telemetry_does_not_become_seed_failure(tmp_path):
    subject = subject_with_replays(tmp_path)
    seed = subject.runtime_evidence.replays[-1]
    seed = replace(
        seed,
        evidence=replace(
            seed.evidence,
            telemetry_json=None,
            telemetry_unavailable="unsupported capability: session_telemetry",
        ),
    )
    evidence = replace(
        subject.runtime_evidence,
        replays=subject.runtime_evidence.replays[:-1] + (seed,),
    )
    result = SeedControlGrader().run(replace(subject, runtime_evidence=evidence))
    assert result.status is CheckStatus.SKIP
    assert "session_telemetry" in result.evidence[0]


def test_seed_success_describes_forwarding(tmp_path):
    result = SeedControlGrader().run(subject_with_replays(tmp_path))
    assert result.status is CheckStatus.PASS
    assert result.evidence == ["requested seeds were observed as forwarded to reset"]
