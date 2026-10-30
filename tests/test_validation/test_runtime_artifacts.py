import hashlib
import json
from dataclasses import replace

import pytest
from conftest import load_fixture_manifest
from openenv.validation.graders import Subject
from openenv.validation.graders.runtime import (
    ObservationSchemaGrader,
    RewardWellFormedGrader,
    StateContractGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.report import CheckResult, ValidationReportV2
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.contracts import (
    ReplayEvidence,
    RuntimeEvidence,
    WireExchange,
)
from openenv.validation.types import CheckStatus, Lane, Level, SignatureKind, Verdict
from support.runtime import evidence, exchange


def report():
    return ValidationReportV2(
        report_schema_version="2",
        target="subject",
        source_digest="0" * 64,
        signature=SignatureKind.OPENENV_SERVED,
        manifest=NormalizedManifest.model_validate(
            load_fixture_manifest("served_min_pass")
        ),
        policy_version="v2",
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC, Level.RUNTIME],
        results=[],
        verdict=Verdict.WARN,
    )


def measured():
    return evidence(
        exchange(
            "reset",
            {"type": "reset", "data": {"episode_id": "recorded", "seed": 42}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 0}, "done": False, "reward": None},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {"type": "state", "data": {"episode_id": "recorded", "step_count": 0}},
        ),
        exchange(
            "step",
            {"type": "step", "data": {"increment": 1}},
            {
                "type": "observation",
                "data": {"observation": {"counter": 1}, "done": False, "reward": 0.5},
            },
        ),
        exchange(
            "state",
            {"type": "state"},
            {"type": "state", "data": {"episode_id": "recorded", "step_count": 1}},
        ),
        observation_schema={
            "type": "object",
            "properties": {"counter": {"type": "integer"}},
            "required": ["counter"],
        },
    )


def rebuild(directory):
    metadata = json.loads((directory / "collector-evidence.json").read_text())
    trace = json.loads((directory / metadata["trace_file"]).read_text())
    return RuntimeEvidence(
        exchanges=tuple(
            WireExchange(
                row["operation"],
                json.dumps(row["request_json"]),
                json.dumps(row["response_json"]),
            )
            for row in trace
        ),
        observation_schema_json=(
            json.dumps(metadata["observation_schema"])
            if metadata["schema_available"]
            else None
        ),
        reset_observation_schema_json=(
            json.dumps(metadata["reset_observation_schema"])
            if metadata["reset_schema_available"]
            else None
        ),
        failure_phase=metadata["failure_phase"],
        failure_reason=metadata["failure_reason"],
    )


def test_saved_collector_evidence_replays_all_basic_graders(tmp_path):
    original = measured()
    validation_report = report()
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    replay = rebuild(tmp_path)
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, original
    )
    for grader in (
        RewardWellFormedGrader(),
        ObservationSchemaGrader(),
        StateContractGrader(),
    ):
        assert (
            grader.run(subject).status
            == grader.run(replace(subject, runtime_evidence=replay)).status
        )
    assert json.loads(replay.observation_schema_json) == json.loads(
        original.observation_schema_json
    )
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["evidence_schema_version"] == "1"
    assert metadata["complete"] is True
    assert metadata["redacted"] is False
    sums = (tmp_path / "SHA256SUMS").read_text()
    assert (
        hashlib.sha256((tmp_path / "collector-evidence.json").read_bytes()).hexdigest()
        in sums
    )


def test_saved_schema_reproduces_observation_schema_failure(tmp_path):
    original = replace(
        measured(), observation_schema_json='{"required":["missing-field"]}'
    )
    validation_report = report()
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, original
    )
    original_result = ObservationSchemaGrader().run(subject)
    replayed_result = ObservationSchemaGrader().run(
        replace(subject, runtime_evidence=rebuild(tmp_path))
    )
    assert original_result.status.value == "fail"
    assert original_result.status == replayed_result.status
    assert original_result.evidence == replayed_result.evidence


def test_truncated_trace_metadata_preserves_the_collection_failure(tmp_path):
    original = replace(
        measured(), failure_phase="step", failure_reason="step failed (TimeoutError)"
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["complete"] is False
    replayed = rebuild(tmp_path)
    assert replayed.failure_phase == original.failure_phase
    assert replayed.failure_reason == original.failure_reason


def test_schema_and_trace_redaction_marks_evidence_as_modified(tmp_path):
    original = replace(
        measured(),
        observation_schema_json=json.dumps({"description": "hf_notarealtoken12345"}),
        failure_reason="Authorization: Bearer not-a-real-secret",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert metadata["observation_schema"]["description"] == "[REDACTED]"
    assert "not-a-real-secret" not in (tmp_path / "collector-evidence.json").read_text()


def test_deep_subject_json_is_bounded_in_artifacts(tmp_path):
    nested = '"leaf"'
    for _ in range(600):
        nested = '{"child":' + nested + "}"
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", "{}", nested),),
        observation_schema_json=nested,
        failure_phase="step",
        failure_reason="step failed (ValueError)",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert (
        "artifact nesting limit exceeded"
        in (tmp_path / "collector-trace.json").read_text()
    )
    assert (tmp_path / "collector-trace.json").stat().st_size < 20_000


def test_nonfinite_wire_values_remain_valid_artifact_json(tmp_path):
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", "{}", '{"reward":NaN}'),),
        observation_schema_json="{}",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)

    def reject_constant(value):
        raise ValueError(value)

    trace = json.loads(
        (tmp_path / "collector-trace.json").read_text(), parse_constant=reject_constant
    )
    assert trace[0]["response_json"]["reward"] == {"invalid_number": "nan"}


def test_missing_schema_is_distinct_from_an_advertised_null_schema(tmp_path):
    for schema in (None, "null"):
        directory = tmp_path / ("missing" if schema is None else "null")
        original = replace(measured(), observation_schema_json=schema)
        write_runtime_bundle(directory, report(), evidence=original)
        assert rebuild(directory).observation_schema_json == schema


def test_malformed_wire_omission_is_visible_in_metadata(tmp_path):
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", "{}", "not JSON"),),
        failure_phase="step",
        failure_reason="step failed (JSONDecodeError)",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert metadata["omitted_trace_fields"] == [
        {"exchange_index": 0, "field": "response_json"}
    ]


@pytest.mark.parametrize(
    "telemetry", ['{"rubric":[{"config":{"api_key":"private-value"}}]}', "{broken"]
)
def test_telemetry_redaction_or_omission_marks_bundle_modified(tmp_path, telemetry):
    original = replace(measured(), telemetry_json=telemetry)
    write_runtime_bundle(tmp_path, report(), evidence=original)
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert "private-value" not in "".join(
        path.read_text() for path in tmp_path.iterdir()
    )


def test_replay_artifact_retains_transcript_telemetry_identity_and_cleanup(tmp_path):
    sample = replace(
        measured(),
        telemetry_json='{"schema_version":1,"seed":{"value":42}}',
        reset_observation_schema_json='{"type":"object"}',
    )
    original = replace(
        measured(),
        replays=(
            ReplayEvidence("session", sample),
            ReplayEvidence(
                "container",
                sample,
                '{"container_id":"second","image_id":"sha256:abc"}',
                True,
            ),
        ),
        replay_failure_reason="judged replay sampling exceeded total budget",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    artifact = json.loads((tmp_path / "replays.json").read_text())
    assert artifact["failure_reason"] == original.replay_failure_reason
    assert [row["scope"] for row in artifact["samples"]] == ["session", "container"]
    for row in artifact["samples"]:
        assert row["trace"] == [
            {
                "operation": exchange.operation,
                "request_json": json.loads(exchange.request_json),
                "response_json": json.loads(exchange.response_json),
            }
            for exchange in sample.exchanges
        ]
        assert row["schema"] == json.loads(sample.observation_schema_json)
        assert row["reset_schema"] == json.loads(sample.reset_observation_schema_json)
        assert row["telemetry"] == json.loads(sample.telemetry_json)
        assert row["redacted"] is False
    container = artifact["samples"][1]
    assert container["cleanup_complete"] is True
    assert container["provider"] == {"container_id": "second", "image_id": "sha256:abc"}
    sums = (tmp_path / "SHA256SUMS").read_text()
    digest = hashlib.sha256((tmp_path / "replays.json").read_bytes()).hexdigest()
    assert f"{digest}  replays.json" in sums


def test_replay_artifact_redacts_each_independent_evidence_source(tmp_path):
    sample = RuntimeEvidence(
        exchanges=(
            exchange(
                "step", {"token": "private-request"}, {"value": "hf_notarealtoken12345"}
            ),
        ),
        observation_schema_json='{"description":"Bearer private-schema"}',
        reset_observation_schema_json='{"description":"Bearer private-reset-schema"}',
        telemetry_json='{"rubric":{"api_key":"private-telemetry"}}',
        failure_reason="Authorization: Bearer private-failure",
    )
    original = replace(
        measured(),
        replays=(
            ReplayEvidence("container", sample, '{"secret":"private-provider"}', False),
        ),
        replay_failure_reason="Bearer private-schedule",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    text = (tmp_path / "replays.json").read_text()
    assert "private-" not in text
    assert "hf_notarealtoken" not in text
    row = json.loads(text)["samples"][0]
    assert row["redacted"] is True
    assert row["cleanup_complete"] is False
    assert row["provider"]["secret"] == "[REDACTED]"


def test_malformed_replay_omissions_are_explicit_and_do_not_leak_raw_data(tmp_path):
    malformed = "not-json-private-value"
    sample = RuntimeEvidence(
        exchanges=(WireExchange("step", malformed, malformed),),
        observation_schema_json=malformed,
        reset_observation_schema_json=malformed,
        telemetry_json=malformed,
    )
    original = replace(
        measured(), replays=(ReplayEvidence("container", sample, malformed, True),)
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    text = (tmp_path / "replays.json").read_text()
    assert malformed not in text
    row = json.loads(text)["samples"][0]
    assert row["redacted"] is True
    assert row["omitted_trace_fields"] == [
        {"exchange_index": 0, "field": "request_json"},
        {"exchange_index": 0, "field": "response_json"},
    ]
    assert row["omitted_evidence_fields"] == [
        "telemetry",
        "schema",
        "reset_schema",
        "provider",
    ]


def test_replay_artifact_distinguishes_absent_fields_from_json_null(tmp_path):
    samples = tuple(
        ReplayEvidence(
            "session",
            RuntimeEvidence(
                observation_schema_json=value,
                reset_observation_schema_json=value,
                telemetry_json=value,
            ),
            value,
        )
        for value in (None, "null")
    )
    write_runtime_bundle(
        tmp_path, report(), evidence=replace(measured(), replays=samples)
    )
    rows = json.loads((tmp_path / "replays.json").read_text())["samples"]
    for key in ("schema", "reset_schema", "telemetry", "provider"):
        assert rows[0][key] is rows[1][key] is None
        assert rows[0][f"{key}_available"] is False
        assert rows[1][f"{key}_available"] is True


def test_rewriting_bundle_removes_stale_optional_evidence(tmp_path):
    original = replace(
        measured(),
        telemetry_json="{}",
        replays=(ReplayEvidence("session", measured()),),
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    assert (tmp_path / "replays.json").exists()
    assert (tmp_path / "session-telemetry.json").exists()
    write_runtime_bundle(tmp_path, report(), evidence=measured())
    assert not (tmp_path / "replays.json").exists()
    assert not (tmp_path / "session-telemetry.json").exists()
    assert "replays.json" not in (tmp_path / "SHA256SUMS").read_text()


def test_discovery_artifact_retains_true_counts_bounded_previews_and_digest(tmp_path):
    tools = {"tools": []}
    tasks = {
        "splits": [{"name": "train"}],
        "counts": {"train": 100},
        "previews": {"train": ["task-0", "task-1"]},
    }
    original = replace(
        measured(), tools_json=json.dumps(tools), tasks_json=json.dumps(tasks)
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    path = tmp_path / "discovery.json"
    artifact = json.loads(path.read_text())
    assert artifact["tools"] == tools and artifact["tasks"] == tasks
    assert artifact["tools_available"] is artifact["tasks_available"] is True
    assert artifact["tools_error"] is artifact["tasks_error"] is None
    assert artifact["redacted"] is False
    assert (
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  discovery.json"
        in (tmp_path / "SHA256SUMS").read_text()
    )


def test_discovery_artifact_redacts_tool_task_and_error_credentials(tmp_path):
    original = replace(
        measured(),
        tools_json='{"tools":[{"api_key":"private-tool"}]}',
        tasks_json='{"previews":{"train":[{"password":"private-task"}]}}',
        tools_error="Authorization: Bearer private-error",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    text = (tmp_path / "discovery.json").read_text()
    assert "private-" not in text
    assert json.loads(text)["redacted"] is True


def test_token_redaction_preserves_task_check_identifiers(tmp_path):
    validation_report = report()
    validation_report.results = [
        CheckResult(
            check_id="runtime.task_declaration_accuracy",
            status=CheckStatus.PASS,
            duration_s=0,
        )
    ]
    original = replace(
        measured(),
        tools_json='{"tools":[{"name":"task_declaration_accuracy","description":"sk_thisisafaketoken123"}]}',
    )
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    assert json.loads(
        (tmp_path / "report.json").read_text()
    ) == validation_report.model_dump(mode="json")
    artifact = json.loads((tmp_path / "discovery.json").read_text())
    assert artifact["tools"]["tools"][0] == {
        "name": "task_declaration_accuracy",
        "description": "[REDACTED]",
    }


def test_malformed_discovery_omissions_preserve_collection_failures(tmp_path):
    original = replace(
        measured(),
        tools_json="invalid-private-tool",
        tasks_json="invalid-private-task",
        tools_error="tool discovery failed (ValueError)",
        tasks_error="task discovery failed (ValueError)",
    )
    write_runtime_bundle(tmp_path, report(), evidence=original)
    text = (tmp_path / "discovery.json").read_text()
    assert "invalid-private" not in text
    artifact = json.loads(text)
    assert artifact["redacted"] is True
    assert artifact["omitted_fields"] == ["tools", "tasks"]
    assert artifact["tools_error"] == original.tools_error
    assert artifact["tasks_error"] == original.tasks_error


def test_discovery_distinguishes_missing_from_null_and_removes_stale_artifact(tmp_path):
    original = replace(measured(), tasks_json="null")
    write_runtime_bundle(tmp_path, report(), evidence=original)
    artifact = json.loads((tmp_path / "discovery.json").read_text())
    assert artifact["tools"] is artifact["tasks"] is None
    assert artifact["tools_available"] is False and artifact["tasks_available"] is True
    write_runtime_bundle(tmp_path, report())
    assert not (tmp_path / "discovery.json").exists()
    assert "discovery.json" not in (tmp_path / "SHA256SUMS").read_text()


def test_redaction_preserves_token_metadata_and_filters_secret_keys():
    from openenv.validation.runtime.artifacts import _redact

    secret = "hf_abcdefghijk123456789"
    source = {
        "task_distribution": "task_distribution",
        "max_tokens": 100,
        "prompt_token_ids": [1, 2, 3],
        "access_token": "opaque-credential",
        secret: {"description": "public"},
        "nested": {"key": secret},
    }
    filtered = _redact(source)
    assert filtered["task_distribution"] == "task_distribution"
    assert filtered["max_tokens"] == 100
    assert filtered["prompt_token_ids"] == [1, 2, 3]
    assert filtered["access_token"] == "[REDACTED]"
    assert secret not in json.dumps(filtered)


@pytest.mark.parametrize(
    "reset_schema", [None, "null", '{"required":["missing-reset-field"]}']
)
def test_reset_schema_bundle_preserves_selection_and_grader_result(
    tmp_path, reset_schema
):
    original = replace(measured(), reset_observation_schema_json=reset_schema)
    validation_report = report()
    write_runtime_bundle(tmp_path, validation_report, evidence=original)
    replay = rebuild(tmp_path)
    if reset_schema is None:
        assert replay.reset_observation_schema_json is None
    else:
        assert json.loads(replay.reset_observation_schema_json) == json.loads(
            reset_schema
        )
    subject = Subject(
        tmp_path, validation_report.manifest, None, None, tmp_path, original
    )
    expected = ObservationSchemaGrader().run(subject)
    actual = ObservationSchemaGrader().run(replace(subject, runtime_evidence=replay))
    assert actual.status == expected.status
    assert actual.evidence == expected.evidence


@pytest.mark.parametrize(
    "schema,parse_failed",
    [("not JSON", True), ('{"description":"hf_abcdefghijk123456789"}', False)],
)
def test_reset_schema_omission_or_redaction_is_visible(tmp_path, schema, parse_failed):
    write_runtime_bundle(
        tmp_path,
        report(),
        evidence=replace(measured(), reset_observation_schema_json=schema),
    )
    text = (tmp_path / "collector-evidence.json").read_text()
    metadata = json.loads(text)
    assert metadata["reset_schema_available"] is True
    assert metadata["reset_schema_parse_failed"] is parse_failed
    assert metadata["redacted"] is True
    assert "hf_abcdefghijk123456789" not in text
