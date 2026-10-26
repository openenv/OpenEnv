import copy
import json
import subprocess
import time
from dataclasses import replace

import pytest
from conftest import load_fixture_manifest
from openenv.core.env_server.types import Observation
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime import (
    basic,
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.runtime.contracts import RuntimeEvidence, WireExchange
from openenv.validation.types import CheckStatus
from support.runtime import evidence, exchange

GRADERS = [RewardWellFormedGrader, ObservationSchemaGrader, StateContractGrader]
OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "counter": {"type": "integer"},
        "reward": {"type": ["number", "null"]},
        "done": {"type": "boolean"},
    },
    "required": ["counter", "reward", "done"],
}


def good_rows():
    return [
        exchange(
            "reset",
            {"type": "reset", "data": {"episode_id": "measured-episode", "seed": 42}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 0}, "reward": None, "done": False},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {
                "type": "state",
                "data": {"episode_id": "measured-episode", "step_count": 0},
            },
        ),
        exchange(
            "step",
            {"type": "step", "data": {"increment": 1}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 1}, "reward": 1.0, "done": False},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {
                "type": "state",
                "data": {"episode_id": "measured-episode", "step_count": 1},
            },
        ),
    ]


def subject_with(tmp_path, rows=None, schema=OBSERVATION_SCHEMA):
    return Subject(
        root=tmp_path,
        manifest=NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        ),
        image_ref=None,
        running=None,
        outputs_dir=tmp_path,
        runtime_evidence=evidence(
            *(good_rows() if rows is None else rows), observation_schema=schema
        ),
    )


def mutate_response(rows, index, mutate):
    payload = json.loads(rows[index].response_json)
    mutate(payload)
    rows[index] = replace(rows[index], response_json=json.dumps(payload))
    return rows


@pytest.mark.parametrize("grader", GRADERS)
def test_good_measured_session_passes_each_basic_contract(tmp_path, grader):
    assert grader().run(subject_with(tmp_path)).status is CheckStatus.PASS


@pytest.mark.parametrize(
    "reward",
    [
        True,
        False,
        "0.5",
        [],
        {},
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
        -0.1,
        1.1,
    ],
)
@pytest.mark.parametrize("done", [False, True])
def test_step_rewards_are_checked_without_coercion(tmp_path, reward, done):
    rows = mutate_response(
        good_rows(),
        2,
        lambda response: response["data"].update(reward=reward, done=done),
    )
    result = RewardWellFormedGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert any("reward" in message for message in result.evidence)


@pytest.mark.parametrize("reward", [None, True, False])
@pytest.mark.parametrize("done", [False, True])
def test_core_reward_types_do_not_imply_level_two_certification(tmp_path, reward, done):
    # Core compatibility does not weaken the stricter certification contract.
    assert Observation(reward=reward, done=done).reward is reward
    rows = mutate_response(
        good_rows(),
        2,
        lambda response: response["data"].update(reward=reward, done=done),
    )
    result = RewardWellFormedGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert any("exchange 2: reward" in message for message in result.evidence)
    assert json.loads(rows[2].response_json)["data"]["reward"] is reward


def test_null_reset_reward_remains_unscored(tmp_path):
    rows = good_rows()
    assert json.loads(rows[0].response_json)["data"]["reward"] is None
    assert (
        RewardWellFormedGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.PASS
    )
    assert json.loads(rows[0].response_json)["data"]["reward"] is None


@pytest.mark.parametrize(
    "invalid_index,invalid_observation",
    [(0, {"ready": "yes"}), (2, {"counter": "not-an-integer"})],
)
def test_explicit_reset_schema_is_independent_of_step_schema(
    tmp_path, invalid_index, invalid_observation
):
    rows = mutate_response(
        good_rows(),
        0,
        lambda response: response["data"].update(observation={"ready": True}),
    )
    subject = subject_with(tmp_path, rows)
    # A legacy single-schema server remains strict about reset fields.
    assert ObservationSchemaGrader().run(subject).status is CheckStatus.FAIL
    subject = replace(
        subject,
        runtime_evidence=replace(
            subject.runtime_evidence,
            reset_observation_schema_json=json.dumps(
                {
                    "type": "object",
                    "properties": {"ready": {"type": "boolean"}},
                    "required": ["ready"],
                }
            ),
        ),
    )
    assert ObservationSchemaGrader().run(subject).status is CheckStatus.PASS
    malformed = mutate_response(
        rows,
        invalid_index,
        lambda response: response["data"].update(observation=invalid_observation),
    )
    subject = replace(
        subject,
        runtime_evidence=replace(subject.runtime_evidence, exchanges=tuple(malformed)),
    )
    result = ObservationSchemaGrader().run(subject)
    assert result.status is CheckStatus.FAIL
    assert all(f"exchange {invalid_index}:" in message for message in result.evidence)


@pytest.mark.parametrize(
    "schema",
    [
        None,
        [],
        {"type": "invalid"},
        False,
        {"$ref": "https://example.invalid/reset.json"},
    ],
)
def test_explicit_invalid_or_rejecting_reset_schema_cannot_fall_back(tmp_path, schema):
    subject = subject_with(tmp_path)
    subject = replace(
        subject,
        runtime_evidence=replace(
            subject.runtime_evidence, reset_observation_schema_json=json.dumps(schema)
        ),
    )
    assert ObservationSchemaGrader().run(subject).status is CheckStatus.FAIL


def test_missing_step_reward_is_an_explicit_failure(tmp_path):
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].pop("reward")
    )
    result = RewardWellFormedGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2: missing reward" in result.evidence


@pytest.mark.parametrize("reward", [0, 1, 0.5])
def test_numeric_reward_endpoints_and_fraction_are_valid(tmp_path, reward):
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].update(reward=reward)
    )
    assert (
        RewardWellFormedGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.PASS
    )


@pytest.mark.parametrize("grader", GRADERS)
def test_reset_only_is_incomplete_not_a_passing_step_contract(tmp_path, grader):
    rows = mutate_response(
        good_rows()[:2], 0, lambda response: response["data"].update(done=True)
    )
    result = grader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.SKIP
    assert any("no step" in message for message in result.evidence)


@pytest.mark.parametrize("grader", GRADERS)
def test_missing_evidence_is_a_named_skip(tmp_path, grader):
    subject = replace(subject_with(tmp_path), runtime_evidence=None)
    result = grader().run(subject)
    assert result.status is CheckStatus.SKIP
    assert result.evidence == ["runtime evidence is unavailable"]


@pytest.mark.parametrize("grader", GRADERS)
def test_truncated_collection_cannot_pass_from_a_valid_prefix(tmp_path, grader):
    subject = subject_with(tmp_path)
    subject = replace(
        subject,
        runtime_evidence=replace(
            subject.runtime_evidence,
            failure_phase="step",
            failure_reason="step failed (TimeoutError)",
        ),
    )
    assert grader().run(subject).status is CheckStatus.SKIP


@pytest.mark.parametrize(
    "schema", [{"type": "not-a-json-type"}, {"required": "counter"}, None]
)
def test_invalid_or_missing_advertised_schema_fails(tmp_path, schema):
    result = ObservationSchemaGrader().run(subject_with(tmp_path, schema=schema))
    assert result.status is CheckStatus.FAIL
    assert any("schema" in message for message in result.evidence)


@pytest.mark.parametrize(
    "reference", ["https://example.invalid/schema", "file:///etc/passwd", "other.json"]
)
@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
def test_nonlocal_schema_references_are_rejected_before_starting_worker(
    tmp_path, monkeypatch, keyword, reference
):
    def no_worker(*args, **kwargs):
        pytest.fail("untrusted external references must not reach schema evaluation")

    monkeypatch.setattr(basic.subprocess, "run", no_worker)
    result = ObservationSchemaGrader().run(
        subject_with(tmp_path, schema={keyword: reference})
    )
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["observation schema has a non-local reference"]


def test_internal_schema_references_are_supported(tmp_path):
    schema = copy.deepcopy(OBSERVATION_SCHEMA)
    schema["$defs"] = {"counter": {"type": "integer"}}
    schema["properties"]["counter"] = {"$ref": "#/$defs/counter"}
    assert (
        ObservationSchemaGrader().run(subject_with(tmp_path, schema=schema)).status
        is CheckStatus.PASS
    )


def test_schema_reconstructs_reward_and_done_from_envelope(tmp_path):
    # These are deliberately absent from the nested observation, as on the wire.
    rows = good_rows()
    assert "reward" not in json.loads(rows[2].response_json)["data"]["observation"]
    assert (
        ObservationSchemaGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.PASS
    )


@pytest.mark.parametrize(
    "field,value", [("done", None), ("done", "false"), ("done", 0), ("observation", [])]
)
def test_invalid_raw_envelope_is_rejected_before_defaults(tmp_path, field, value):
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].update({field: value})
    )
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2: malformed observation envelope" in result.evidence


@pytest.mark.parametrize("data", [None, [], "private-payload", 42, 0.5, False])
@pytest.mark.parametrize(
    "grader,index,envelope",
    [
        (RewardWellFormedGrader, 0, "observation"),
        (RewardWellFormedGrader, 2, "observation"),
        (ObservationSchemaGrader, 0, "observation"),
        (ObservationSchemaGrader, 2, "observation"),
        (StateContractGrader, 1, "state"),
        (StateContractGrader, 3, "state"),
    ],
)
def test_non_object_data_is_a_finding_not_a_validator_crash(
    tmp_path, grader, index, envelope, data
):
    rows = mutate_response(
        good_rows(), index, lambda response: response.update(data=data)
    )
    result = grader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert f"exchange {index}: malformed {envelope} envelope" in result.evidence
    assert "private-payload" not in result.model_dump_json()


def test_missing_done_is_not_filled_by_a_model_default(tmp_path):
    rows = mutate_response(
        good_rows(), 2, lambda response: response["data"].pop("done")
    )
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2: malformed observation envelope" in result.evidence


def test_schema_mismatch_evidence_does_not_echo_private_subject_values(tmp_path):
    private_value = "not-a-number-secret-observation"
    rows = mutate_response(
        good_rows(),
        2,
        lambda response: response["data"]["observation"].update(counter=private_value),
    )
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert any("counter" in message for message in result.evidence)
    assert private_value not in result.model_dump_json()


def test_schema_worker_deadline_becomes_a_validation_finding(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        assert command[-1].endswith("schema_worker.py")
        assert kwargs["encoding"] == "utf-8"
        assert 0 < kwargs["timeout"] <= 5
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(basic.subprocess, "run", timeout)
    result = ObservationSchemaGrader().run(subject_with(tmp_path))
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["observation schema evaluation exceeded its time budget"]


def test_pathological_regex_runs_in_killable_worker(tmp_path, monkeypatch):
    run = subprocess.run

    def short_deadline(command, **kwargs):
        kwargs["timeout"] = 0.5
        return run(command, **kwargs)

    monkeypatch.setattr(basic.subprocess, "run", short_deadline)
    schema = copy.deepcopy(OBSERVATION_SCHEMA)
    schema["properties"]["counter"] = {"type": "string", "pattern": "^(a+)+$"}
    rows = good_rows()
    for index in (0, 2):
        mutate_response(
            rows,
            index,
            lambda response: response["data"]["observation"].update(
                counter="a" * 100 + "!"
            ),
        )
    started = time.monotonic()
    result = ObservationSchemaGrader().run(subject_with(tmp_path, rows, schema=schema))
    assert time.monotonic() - started < 3
    assert result.status is CheckStatus.FAIL
    assert any("time budget" in message for message in result.evidence)


@pytest.mark.parametrize("episode_id", ["other-episode", None, 42])
def test_state_identity_must_match_the_requested_episode(tmp_path, episode_id):
    rows = mutate_response(
        good_rows(), 3, lambda response: response["data"].update(episode_id=episode_id)
    )
    result = StateContractGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 3: episode_id differs from reset" in result.evidence


@pytest.mark.parametrize("count", [True, 1.0, "1", None, -1, 0, 2])
def test_state_count_is_strict_and_matches_successful_steps(tmp_path, count):
    rows = mutate_response(
        good_rows(), 3, lambda response: response["data"].update(step_count=count)
    )
    result = StateContractGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "exchange 3: incorrect step_count" in result.evidence


def test_reset_state_count_must_start_at_zero(tmp_path):
    rows = mutate_response(
        good_rows(), 1, lambda response: response["data"].update(step_count=5)
    )
    assert (
        StateContractGrader().run(subject_with(tmp_path, rows)).status
        is CheckStatus.FAIL
    )


def test_missing_state_snapshots_do_not_pass(tmp_path):
    rows = [row for row in good_rows() if row.operation != "state"]
    result = StateContractGrader().run(subject_with(tmp_path, rows))
    assert result.status is CheckStatus.FAIL
    assert "no state snapshots were measured" in result.evidence


@pytest.mark.parametrize("grader", GRADERS)
def test_malformed_wire_is_a_finding_not_a_validator_crash(tmp_path, grader):
    rows = good_rows()
    rows[2] = WireExchange("step", rows[2].request_json, "{not-json")
    # Collector failure is part of the evidence even for graders that do not read
    # step response bodies (the state check only needs the step request count).
    subject = subject_with(tmp_path, rows)
    subject = replace(
        subject,
        runtime_evidence=RuntimeEvidence(
            exchanges=subject.runtime_evidence.exchanges,
            observation_schema_json=subject.runtime_evidence.observation_schema_json,
            failure_phase="step",
            failure_reason="step failed (JSONDecodeError)",
        ),
    )
    assert grader().run(subject).status is CheckStatus.SKIP


def test_schema_required_keyword_identifies_missing_property(tmp_path):
    result = ObservationSchemaGrader().run(
        subject_with(tmp_path, schema={"required": ["tool_name"]})
    )
    assert result.status is CheckStatus.FAIL
    assert all("(required)" in message for message in result.evidence)
    assert all(
        'missing properties: ["tool_name"]' in message for message in result.evidence
    )


def test_schema_diagnostics_do_not_echo_private_property_names(tmp_path):
    secret = "hf_abcdefghijk123456789"
    result = ObservationSchemaGrader().run(
        subject_with(tmp_path, schema={"required": [secret]})
    )
    assert result.status is CheckStatus.FAIL
    assert secret not in result.model_dump_json()
