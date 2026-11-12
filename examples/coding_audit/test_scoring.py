"""Admission tests for pure coding-audit evidence scoring; no execution backends."""

import copy
import json

import pytest
from examples.coding_audit.scoring import assess_coding_audit, validate_audit_config

TASK = "a" * 64
VERIFIER = "b" * 64
IMAGE = "registry/agent@sha256:" + "c" * 64
VERIFIER_IMAGE = "registry/verifier@sha256:" + "d" * 64
TARGET = {"model": "Qwen/Qwen3.8-27B", "revision": "e" * 40, "provider": "hf-inference"}
CONFIG = {"required_negative_cases": ["wrong_answer", "empty_answer"]}


def trial(role, case_id, reward, number, submission_digest=None):
    result = {
        "role": role,
        "case_id": case_id,
        "task_digest": TASK,
        "verifier_digest": VERIFIER,
        "image_ref": IMAGE,
        "verifier_image_ref": VERIFIER_IMAGE,
        "submission_digest": submission_digest or "f" * 64,
        "sandbox_id": f"agent-{number}",
        "grade_sandbox_id": f"grade-{number}",
        "reward": reward,
        "cleanup_verified": True,
        "error": None,
    }
    if role == "target":
        result.update(TARGET)
        result.update(
            latency_seconds=2.5, usage={"input_tokens": 100, "output_tokens": 20}
        )
    return result


@pytest.fixture
def evidence():
    return {
        "task_digest": TASK,
        "verifier_digest": VERIFIER,
        "agent_image": IMAGE,
        "verifier_image": VERIFIER_IMAGE,
        "target": dict(TARGET),
        "references": [
            {
                "case_id": "oracle",
                "submission_digest": "f" * 64,
                "origin": "provided",
            }
        ],
        "trials": [
            trial("nop", "nop", 0, 0),
            trial("reference", "oracle", 1, 1),
            trial("reference", "oracle", 1, 2),
            trial("negative", "wrong_answer", 0, 3),
            trial("negative", "empty_answer", 0, 4),
            trial("target", "attempt-1", 0, 5),
            trial("target", "attempt-2", 1, 6),
        ],
        "leakage_probes": [
            {"name": name, "passed": True, "error": None}
            for name in (
                "private_paths_absent",
                "fresh_state_absent",
                "canary_not_in_transcript",
            )
        ],
    }


def test_complete_audit_reports_bounded_measurements_without_mutating_inputs(evidence):
    original = copy.deepcopy(evidence)
    report = assess_coding_audit(CONFIG, evidence)
    assert report["profile"] == "harbor-coding-audit-v1"
    assert report["accepted"] is True
    assert report["verdict"] == "pass"
    assert not report["rfc008_complete"]
    assert not report["trainability_established"]
    assert {check["status"] for check in report["checks"].values()} == {"pass"}
    target = report["checks"]["target_difficulty"]
    assert target["completed"] == 2
    assert target["failed"] == target["missing"] == 0
    assert target["solve_rate"] == 0.5
    assert target["solve_rate_wilson95"] == pytest.approx([0.0945312057, 0.9054687943])
    assert target["reward_variance"] == 0.25
    assert target["mixed_success"] is True
    assert target["difficulty"] == "mixed"
    assert target["latency_seconds_total"] == 5
    assert target["usage_totals"] == {"input_tokens": 200, "output_tokens": 40}
    assert all(value == 0 for value in target["telemetry_missing"].values())
    assert "bounded probes" in report["checks"]["leakage"]["scope"]
    assert evidence == original
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("reward,difficulty", [(0, "none_solved"), (1, "all_solved")])
def test_uniform_target_rewards_are_valid_difficulty_measurements(
    evidence, reward, difficulty
):
    for record in evidence["trials"][-2:]:
        record["reward"] = reward
    report = assess_coding_audit(CONFIG, evidence)
    assert report["accepted"]
    target = report["checks"]["target_difficulty"]
    assert target["difficulty"] == difficulty
    assert target["mixed_success"] is False
    assert target["reward_variance"] == 0
    assert target["solve_rate"] == reward


def test_target_infrastructure_error_is_missing_evidence_not_a_zero_reward(evidence):
    failed = evidence["trials"][-2]
    failed.update(error="provider timeout", reward=None)
    report = assess_coding_audit(CONFIG, evidence)
    assert not report["accepted"]
    target = report["checks"]["target_difficulty"]
    assert target["status"] == "error"
    assert target["attempted"] == 2
    assert target["completed"] == target["failed"] == target["missing"] == 1
    assert target["infrastructure_errors"] == 1
    assert target["solve_rate"] == 1
    assert target["reward_variance"] == 0
    assert target["usage_totals"]["input_tokens"] == 200


def test_missing_target_attempt_is_skip(evidence):
    evidence["trials"].pop()
    report = assess_coding_audit(CONFIG, evidence)
    assert report["verdict"] == "skip"
    assert report["checks"]["target_difficulty"]["missing"] == 1


def test_configured_repeat_and_attempt_counts_are_honored(evidence):
    config = CONFIG | {"reference_repeats": 3, "target_attempts": 3}
    report = assess_coding_audit(config, evidence)
    assert report["checks"]["references"]["status"] == "skip"
    assert report["checks"]["target_difficulty"]["missing"] == 1
    evidence["trials"].extend(
        [trial("reference", "oracle", 1, 10), trial("target", "attempt-3", 0, 11)]
    )
    assert assess_coding_audit(config, evidence)["accepted"]


@pytest.mark.parametrize(
    "duplicate", ["record", "case_id", "sandbox_id", "grade_sandbox_id", "cross_role"]
)
def test_duplicate_target_attempts_never_count_as_coverage(evidence, duplicate):
    first, second = evidence["trials"][-2:]
    if duplicate == "record":
        evidence["trials"][-1] = dict(first)
    elif duplicate == "cross_role":
        second["sandbox_id"] = evidence["trials"][1]["grade_sandbox_id"]
    else:
        second[duplicate] = first[duplicate]
    report = assess_coding_audit(CONFIG, evidence)
    assert report["verdict"] == "error"
    assert report["checks"]["target_difficulty"]["completed"] < 2


def test_no_target_data_reports_null_metrics(evidence):
    evidence["trials"] = [
        record for record in evidence["trials"] if record["role"] != "target"
    ]
    report = assess_coding_audit(CONFIG, evidence)
    target = report["checks"]["target_difficulty"]
    assert target["status"] == "skip"
    for name in (
        "solve_rate",
        "solve_rate_wilson95",
        "reward_variance",
        "mixed_success",
    ):
        assert target[name] is None
    assert target["usage_totals"] == {"input_tokens": None, "output_tokens": None}
    assert target["difficulty"] == "unmeasured"


def test_missing_telemetry_is_explicit_and_does_not_invalidate_reward(evidence):
    for record in evidence["trials"][-2:]:
        del record["usage"]
        del record["latency_seconds"]
    report = assess_coding_audit(CONFIG, evidence)
    assert report["accepted"]
    target = report["checks"]["target_difficulty"]
    assert target["telemetry_missing"] == {
        "latency_seconds": 2,
        "input_tokens": 2,
        "output_tokens": 2,
    }
    assert target["latency_seconds_total"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("latency_seconds", -1),
        ("latency_seconds", None),
        ("latency_seconds", True),
        ("latency_seconds", float("inf")),
        ("usage", None),
        ("usage", {"input_tokens": True, "output_tokens": -5}),
    ],
)
def test_supplied_malformed_telemetry_is_error_not_missing(evidence, field, value):
    evidence["trials"][-1][field] = value
    report = assess_coding_audit(CONFIG, evidence)
    assert report["verdict"] == "error"
    target = report["checks"]["target_difficulty"]
    assert target["status"] == "error"
    assert target["telemetry_errors"]
    assert all(value == 0 for value in target["telemetry_missing"].values())


def test_latency_total_overflow_returns_error_report(evidence):
    evidence["trials"].append(trial("target", "attempt-3", 1, 10))
    for record, seconds in zip(evidence["trials"][-3:], [10**308, 10**308, 0.0]):
        record["latency_seconds"] = seconds
    report = assess_coding_audit(CONFIG, evidence)
    assert report["verdict"] == "error"
    assert report["checks"]["target_difficulty"]["latency_seconds_total"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "reward",
    [True, False, float("nan"), float("inf"), -float("inf"), None, "1", -0.01, 1.01],
)
@pytest.mark.parametrize("index", [0, 1, 3, 5])
def test_invalid_rewards_block_all_roles(evidence, reward, index):
    evidence["trials"][index]["reward"] = reward
    report = assess_coding_audit(CONFIG, evidence)
    assert not report["accepted"]
    assert report["checks"]["reliability"]["status"] == "error"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("cleanup", [False, 1, "true", None])
def test_cleanup_requires_literal_true(evidence, cleanup):
    evidence["trials"][1]["cleanup_verified"] = cleanup
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


@pytest.mark.parametrize(
    "field",
    [
        "error",
        "cleanup_verified",
        "sandbox_id",
        "grade_sandbox_id",
        "submission_digest",
    ],
)
def test_missing_trial_provenance_never_passes(evidence, field):
    del evidence["trials"][1][field]
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


@pytest.mark.parametrize("field", ["case_id", "submission_digest"])
@pytest.mark.parametrize("value", [[], {}, True])
def test_malformed_reference_group_keys_return_error_without_crashing(
    evidence, field, value
):
    evidence["trials"][1][field] = value
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


def test_sha256_prefix_does_not_split_identical_reference_replays(evidence):
    evidence["references"][0]["submission_digest"] = "sha256:" + "f" * 64
    evidence["trials"][1]["submission_digest"] = "sha256:" + "f" * 64
    evidence["trials"][0]["task_digest"] = "sha256:" + TASK
    evidence["trials"][3]["verifier_digest"] = "sha256:" + VERIFIER
    assert assess_coding_audit(CONFIG, evidence)["accepted"]


def test_unrepresentable_reward_variance_blocks_acceptance_without_nonfinite_json(
    evidence,
):
    config = CONFIG | {"reward_max": 1e308}
    for record in evidence["trials"]:
        if record["reward"] == 1:
            record["reward"] = 1e308
    report = assess_coding_audit(config, evidence)
    assert report["verdict"] == "error"
    assert report["checks"]["target_difficulty"]["reward_variance"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("index", [0, 1, 3, 5])
@pytest.mark.parametrize(
    "field,value",
    [
        ("task_digest", "0" * 64),
        ("verifier_digest", "1" * 64),
        ("image_ref", "other@sha256:" + "2" * 64),
        ("verifier_image_ref", "other@sha256:" + "3" * 64),
    ],
)
def test_every_trial_binds_to_expected_task_verifier_and_images(
    evidence, index, field, value
):
    evidence["trials"][index][field] = value
    report = assess_coding_audit(CONFIG, evidence)
    assert report["verdict"] == "error"
    assert any(
        "binding mismatch" in reason
        for reason in report["checks"]["reliability"]["reasons"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_digest", "a" * 63),
        ("verifier_digest", "b" * 65),
        ("agent_image", "agent:latest"),
        ("verifier_image", "sha256:abc"),
    ],
)
def test_common_binding_requires_full_digests_and_immutable_images(
    evidence, field, value
):
    evidence[field] = value
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


@pytest.mark.parametrize(
    "field,value",
    [("model", "other/model"), ("revision", "a" * 40), ("provider", "other-provider")],
)
def test_target_identity_is_bound(evidence, field, value):
    evidence["trials"][-1][field] = value
    report = assess_coding_audit(CONFIG, evidence)
    assert report["checks"]["target_difficulty"]["status"] == "error"
    assert report["checks"]["target_difficulty"]["completed"] == 1


@pytest.mark.parametrize("revision", ["main", "abc", "a" * 39, None])
def test_common_target_revision_must_be_immutable(evidence, revision):
    evidence["target"]["revision"] = revision
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


def test_reference_replay_drift_blocks_acceptance(evidence):
    evidence["trials"][2]["reward"] = 0
    report = assess_coding_audit(CONFIG, evidence)
    assert report["checks"]["references"]["status"] == "fail"
    assert report["checks"]["reliability"]["status"] == "fail"


def test_separate_generated_submissions_are_not_deterministic_replays(evidence):
    evidence["references"] = [
        {
            "case_id": "oracle",
            "submission_digest": digest * 64,
            "origin": "generated",
            "candidate_script_digest": digest * 64,
        }
        for digest in ("0", "1")
    ]
    evidence["trials"][1]["submission_digest"] = "0" * 64
    evidence["trials"][2]["submission_digest"] = "1" * 64
    report = assess_coding_audit(CONFIG, evidence)
    assert report["verdict"] == "skip"
    assert report["checks"]["references"]["validated"] == []
    assert report["checks"]["reliability"]["status"] == "skip"


def test_clean_failed_generated_candidate_is_excluded_but_must_execute(evidence):
    evidence["references"].append(
        {
            "case_id": "generated",
            "submission_digest": "0" * 64,
            "origin": "generated",
            "candidate_script_digest": "1" * 64,
        }
    )
    report = assess_coding_audit(CONFIG, evidence)
    assert report["checks"]["references"]["status"] == "skip"
    evidence["trials"].append(trial("reference", "generated", 0, 10, "0" * 64))
    report = assess_coding_audit(CONFIG, evidence)
    assert report["accepted"]
    assert report["checks"]["references"]["excluded"] == [
        {"case_id": "generated", "submission_digest": "0" * 64}
    ]
    evidence["trials"][-1]["error"] = "execution failed"
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


def test_generated_reference_can_be_validated_without_provided_oracle(evidence):
    evidence["references"][0].update(
        origin="generated", candidate_script_digest="1" * 64
    )
    assert assess_coding_audit(CONFIG, evidence)["accepted"]
    del evidence["references"][0]["candidate_script_digest"]
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"


@pytest.mark.parametrize("kind", ["nop", "wrong_answer", "contradictory_negative"])
def test_verifier_must_reject_nop_and_every_observed_known_negative(evidence, kind):
    if kind == "nop":
        evidence["trials"][0]["reward"] = 1
    elif kind == "wrong_answer":
        evidence["trials"][3]["reward"] = 1
    else:
        evidence["trials"].append(trial("negative", "wrong_answer", 1, 10))
    report = assess_coding_audit(CONFIG, evidence)
    assert report["checks"]["verifier_rejection"]["status"] == "fail"


def test_generated_negative_labels_do_not_establish_trusted_coverage(evidence):
    evidence["trials"][3]["case_id"] = "model_says_wrong"
    evidence["required_negative_cases"] = ["model_says_wrong", "empty_answer"]
    evidence["trials"][3]["reviewed"] = True
    report = assess_coding_audit(CONFIG, evidence)
    assert report["checks"]["verifier_rejection"]["status"] == "skip"
    assert report["checks"]["verifier_rejection"]["covered_negative_cases"] == [
        "empty_answer"
    ]


@pytest.mark.parametrize("name", ["nop", "reference", "negative"])
def test_missing_role_is_incomplete(evidence, name):
    evidence["trials"] = [
        record for record in evidence["trials"] if record["role"] != name
    ]
    report = assess_coding_audit(CONFIG, evidence)
    assert not report["accepted"]
    assert report["verdict"] == "skip"


def test_floor_margin_is_required(evidence):
    config = CONFIG | {"floor_margin": 0.95, "oracle_tolerance": 0.1}
    evidence["trials"][0]["reward"] = 0.1
    report = assess_coding_audit(config, evidence)
    assert report["checks"]["references"]["status"] == "fail"


def test_floor_gap_does_not_round_away_tiny_nop_reward(evidence):
    config = CONFIG | {"oracle_tolerance": 1e-300, "floor_margin": 1}
    evidence["trials"][0]["reward"] = 1e-300
    report = assess_coding_audit(config, evidence)
    assert report["checks"]["verifier_rejection"]["status"] == "pass"
    assert report["checks"]["references"]["status"] == "fail"
    assert not report["accepted"]


def test_margin_must_fit_exact_reward_range():
    config = CONFIG | {
        "reward_min": 1e-300,
        "wrong_answer_max": 1e-300,
        "floor_margin": 1,
    }
    with pytest.raises(ValueError):
        validate_audit_config(config)


def test_zero_float_tolerance_preserves_large_integer_reference_and_solve_threshold(
    evidence,
):
    config = CONFIG | {"reward_max": 2**53 + 1, "oracle_tolerance": 0.0}
    for record in evidence["trials"]:
        if record["role"] in ("reference", "target"):
            record["reward"] = 2**53
    report = assess_coding_audit(config, evidence)
    assert report["checks"]["references"]["status"] == "fail"
    target = report["checks"]["target_difficulty"]
    assert target["solve_count"] == 0
    assert target["difficulty"] == "none_solved"
    assert not report["accepted"]


def test_floor_separation_uses_decimal_values_without_float_subtraction_error(evidence):
    config = CONFIG | {
        "reward_min": 0.7,
        "reward_max": 1,
        "oracle_tolerance": 0.1,
        "wrong_answer_max": 0.8,
        "floor_margin": 0.1,
    }
    for record in evidence["trials"]:
        record["reward"] = 0.9 if record["role"] == "reference" else 0.8
    assert assess_coding_audit(config, evidence)["accepted"]


@pytest.mark.parametrize(
    "change,status",
    [
        ("missing", "skip"),
        ("unknown", "skip"),
        ("duplicate", "error"),
        ("false", "fail"),
        ("truthy", "error"),
        ("error", "error"),
        ("missing_error", "error"),
    ],
)
def test_leakage_requires_explicit_named_successful_probes(evidence, change, status):
    if change == "missing":
        evidence["leakage_probes"].pop()
    elif change == "unknown":
        evidence["leakage_probes"][-1]["name"] = "unknown_probe"
    elif change == "duplicate":
        evidence["leakage_probes"].append(dict(evidence["leakage_probes"][0]))
    elif change == "false":
        evidence["leakage_probes"][0]["passed"] = False
    elif change == "truthy":
        evidence["leakage_probes"][0]["passed"] = 1
    elif change == "error":
        evidence["leakage_probes"][0]["error"] = "probe failed to execute"
    else:
        del evidence["leakage_probes"][0]["error"]
    report = assess_coding_audit(CONFIG, evidence)
    assert report["checks"]["leakage"]["status"] == status
    assert not report["accepted"]


def test_unknown_probe_cannot_replace_required_names_but_can_be_diagnostic(evidence):
    evidence["leakage_probes"].append({"name": "extra", "passed": True, "error": None})
    report = assess_coding_audit(CONFIG, evidence)
    assert report["accepted"]
    assert "extra" in report["checks"]["leakage"]["observed_probes"]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"reference_repeats": 1},
        {"reference_repeats": True},
        {"target_attempts": 1},
        {"reward_min": True},
        {"reward_max": float("nan")},
        {"oracle_tolerance": 1},
        {"floor_margin": 0},
        {"floor_margin": float("inf")},
        {"wrong_answer_max": 1},
        {"required_negative_cases": []},
        {"required_negative_cases": ["a", "a"]},
        {"required_negative_cases": [1]},
        {"unknown": True},
    ],
)
def test_invalid_config_raises_and_assessment_returns_error(evidence, config):
    config = config if not config else CONFIG | config
    with pytest.raises(ValueError):
        validate_audit_config(config)
    report = assess_coding_audit(config, evidence)
    assert report["verdict"] == "error"
    assert not report["accepted"]


def test_empty_evidence_cannot_be_accepted():
    report = assess_coding_audit(CONFIG, {})
    assert not report["accepted"]
    assert report["checks"]["leakage"]["status"] == "skip"
    assert report["checks"]["references"]["status"] == "skip"


@pytest.mark.parametrize(
    "field,value",
    [
        ("trials", {}),
        ("references", {}),
        ("leakage_probes", {}),
        ("trials", [None]),
        ("references", [None]),
    ],
)
def test_malformed_containers_return_error(evidence, field, value):
    evidence[field] = value
    assert assess_coding_audit(CONFIG, evidence)["verdict"] == "error"
