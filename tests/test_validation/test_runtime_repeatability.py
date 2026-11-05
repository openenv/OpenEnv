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
