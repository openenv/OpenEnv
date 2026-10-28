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
from openenv.validation.report import ValidationReportV2
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.contracts import (
    HttpProbe,
    IsolationEvidence,
    RuntimeEvidence,
    WireExchange,
)
from openenv.validation.runtime.json_values import dumps, loads
from openenv.validation.types import Lane, Level, SignatureKind, Verdict
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


def rebuild(directory, session="collector"):
    metadata = loads((directory / f"{session}-evidence.json").read_text())
    trace = loads((directory / metadata["trace_file"]).read_text())
    return RuntimeEvidence(
        exchanges=tuple(
            WireExchange(
                row["operation"],
                dumps(row["request_json"]),
                dumps(row["response_json"]),
            )
            for row in trace
        ),
        observation_schema_json=(
            dumps(metadata["observation_schema"])
            if metadata["schema_available"]
            else None
        ),
        reset_observation_schema_json=(
            dumps(metadata["reset_observation_schema"])
            if metadata["reset_schema_available"]
            else None
        ),
        schema_response_json=(
            dumps(metadata["schema_response"])
            if metadata["schema_response_available"]
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


def test_bundle_keeps_each_isolation_session_and_its_completion(tmp_path):
    original = measured()
    interrupted = replace(
        original,
        exchanges=original.exchanges[:2],
        failure_phase="step",
        failure_reason="step failed (KeyboardInterrupt)",
    )
    write_runtime_bundle(
        tmp_path,
        report(),
        evidence=original,
        isolation=IsolationEvidence(sequential=interrupted, fresh=original),
    )
    sequential = json.loads(
        (tmp_path / "isolation-sequential-evidence.json").read_text()
    )
    fresh = json.loads((tmp_path / "isolation-fresh-evidence.json").read_text())
    assert sequential["session"] != fresh["session"]
    assert sequential["complete"] is False
    assert sequential["failure_reason"] == interrupted.failure_reason
    assert fresh["complete"] is True
    assert len(json.loads((tmp_path / sequential["trace_file"]).read_text())) == 2
    assert len(json.loads((tmp_path / fresh["trace_file"]).read_text())) == 4
    sums = (tmp_path / "SHA256SUMS").read_text()
    assert "isolation-sequential-trace.json" in sums
    assert "isolation-fresh-trace.json" in sums


def test_bundle_redacts_oracle_body_and_records_original_digest(tmp_path):
    body = b'{"api_key":"opaque-credential","answer":"canary"}'
    write_runtime_bundle(
        tmp_path,
        report(),
        oracle_http=(HttpProbe(path="oracle/solve.sh", status=403, body=body),),
    )
    probe = json.loads((tmp_path / "oracle-http.json").read_text())[0]
    assert probe["status"] == 403
    assert probe["body"] == {"api_key": "[REDACTED]", "answer": "canary"}
    assert probe["body_sha256"] == hashlib.sha256(body).hexdigest()
    assert probe["retained_bytes"] == len(body)
    assert probe["redacted"] is True
    assert "opaque-credential" not in (tmp_path / "oracle-http.json").read_text()
    assert "oracle-http.json" in (tmp_path / "SHA256SUMS").read_text()


@pytest.mark.parametrize(
    "body",
    [
        b'{"api_key":"opaque-credential", BROKEN}',
        b'{"api_\\u006bey":"opaque-credential", BROKEN}',
        b"#!/bin/sh\nexport API_KEY=opaque-credential\n",
        b"export PASSWORD=opaque-credential\n",
        b'{"stdout":"export API_KEY=opaque-credential","answer":"canary"}',
        b'["canary", {"stdout":"export PASSWORD=opaque-credential"}]',
        b'\xef\xbb\xbf{"api_\\u006bey":"opaque-credential", BROKEN}',
    ],
)
def test_unsafe_unstructured_http_bodies_are_omitted_in_every_hop(tmp_path, body):
    write_runtime_bundle(
        tmp_path,
        report(),
        oracle_http=(HttpProbe("/oracle", 200, body, redirect_bodies=(body,)),),
    )
    text = (tmp_path / "oracle-http.json").read_text()
    assert "opaque-credential" not in text
    probe = json.loads(text)[0]
    for retained in (probe, *probe["redirect_bodies"]):
        assert retained["body_omitted"] is True
        assert retained["redacted"] is True
        assert retained["body_sha256"] == hashlib.sha256(body).hexdigest()
        assert retained["retained_bytes"] == len(body)


def test_bundle_keeps_separate_redacted_redirect_documents_and_failure(tmp_path):
    redirect = b'{"api_key":"opaque-credential","answer":"redirect-canary"}'
    write_runtime_bundle(
        tmp_path,
        report(),
        oracle_http=(
            HttpProbe(
                path="oracle/solve.sh",
                status=302,
                body=b"final prefix",
                failure="TimeoutError",
                redirect_bodies=(redirect,),
            ),
        ),
    )
    probe = json.loads((tmp_path / "oracle-http.json").read_text())[0]
    assert probe["failure"] == "TimeoutError"
    assert probe["body"] == "final prefix"
    hop = probe["redirect_bodies"][0]
    assert hop["body"] == {"api_key": "[REDACTED]", "answer": "redirect-canary"}
    assert hop["body_sha256"] == hashlib.sha256(redirect).hexdigest()
    assert hop["redacted"] is probe["redacted"] is True


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_nested_harmless_http_strings_survive_every_hop(tmp_path, prefix):
    body = prefix + b'{"output":["completed-probe-prefix", {"answer":"canary"}]}'
    write_runtime_bundle(
        tmp_path,
        report(),
        oracle_http=(HttpProbe("/oracle", 200, body, redirect_bodies=(body,)),),
    )
    probe = json.loads((tmp_path / "oracle-http.json").read_text())[0]
    for hop in (probe, *probe["redirect_bodies"]):
        assert hop["body"] == json.loads(body)
        assert hop["body_omitted"] is hop["redacted"] is False


def test_bundle_preserves_exact_numbers_and_types_in_all_json_surfaces(tmp_path):
    from decimal import Decimal

    raw = '{"number":1.0000000000000001,"float":1e0,"huge":1e400,"integer":1,"text":"1.0000000000000001","api_key":"opaque-credential"}'
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", raw, raw),),
        observation_schema_json=raw,
        reset_observation_schema_json=raw,
        schema_response_json=raw,
    )
    body = raw.encode()
    write_runtime_bundle(
        tmp_path,
        report(),
        evidence=original,
        oracle_http=(HttpProbe("/oracle", 200, body, redirect_bodies=(body,)),),
    )
    trace = json.loads(
        (tmp_path / "collector-trace.json").read_text(), parse_float=Decimal
    )
    metadata = json.loads(
        (tmp_path / "collector-evidence.json").read_text(), parse_float=Decimal
    )
    probe = json.loads(
        (tmp_path / "oracle-http.json").read_text(), parse_float=Decimal
    )[0]
    for value in (
        trace[0]["request_json"],
        trace[0]["response_json"],
        metadata["observation_schema"],
        metadata["reset_observation_schema"],
        metadata["schema_response"],
        probe["body"],
        probe["redirect_bodies"][0]["body"],
    ):
        assert value["number"] == Decimal("1.0000000000000001")
        assert value["huge"] == Decimal("1e400")
        assert type(value["float"]) is Decimal
        assert type(value["integer"]) is int
        assert value["text"] == "1.0000000000000001"
        assert value["api_key"] == "[REDACTED]"
    assert metadata["redacted"] is probe["redacted"] is True
    manifest = json.loads((tmp_path / "run-manifest.json").read_text())
    assert (
        manifest["json_numbers"]
        == "exact integer and decimal tokens; decode with Decimal"
    )
    assert "opaque-credential" not in "".join(
        path.read_text() for path in tmp_path.iterdir()
    )
    replay = rebuild(tmp_path)
    assert loads(replay.exchanges[0].response_json)["number"] == Decimal(
        "1.0000000000000001"
    )
    for schema in (
        replay.observation_schema_json,
        replay.reset_observation_schema_json,
        replay.schema_response_json,
    ):
        assert loads(schema)["number"] == Decimal("1.0000000000000001")


def test_unsupported_decimal_range_is_visibly_omitted(tmp_path):
    raw = '{"answer":1e9999999999999999999}'
    original = RuntimeEvidence(
        exchanges=(WireExchange("step", raw, raw),),
        observation_schema_json=raw,
        reset_observation_schema_json=raw,
        schema_response_json=raw,
    )
    write_runtime_bundle(
        tmp_path,
        report(),
        evidence=original,
        oracle_http=(HttpProbe("/oracle", 200, raw.encode()),),
    )
    metadata = json.loads((tmp_path / "collector-evidence.json").read_text())
    assert metadata["redacted"] is True
    assert metadata["schema_parse_failed"] is True
    assert metadata["reset_schema_parse_failed"] is True
    assert metadata["schema_response_parse_failed"] is True
    assert len(metadata["omitted_trace_fields"]) == 2
    probe = json.loads((tmp_path / "oracle-http.json").read_text())[0]
    assert probe["redacted"] is probe["body_omitted"] is True


@pytest.mark.parametrize("schema", [None, '{"action":{"default":"canary"}}'])
def test_bundle_replays_the_complete_schema_response(tmp_path, schema):
    original = replace(measured(), schema_response_json=schema)
    write_runtime_bundle(tmp_path, report(), evidence=original)
    replay = rebuild(tmp_path)
    assert json.loads(replay.schema_response_json or "null") == json.loads(
        schema or "null"
    )


def test_full_schema_redaction_and_parse_failure_are_visible(tmp_path):
    for schema in ('{"api_key":"opaque-credential"}', "not JSON"):
        directory = tmp_path / ("malformed" if schema == "not JSON" else "secret")
        write_runtime_bundle(
            directory,
            report(),
            evidence=replace(measured(), schema_response_json=schema),
        )
        text = (directory / "collector-evidence.json").read_text()
        metadata = json.loads(text)
        assert metadata["schema_response_available"] is True
        assert metadata["schema_response_parse_failed"] is (schema == "not JSON")
        assert metadata["redacted"] is True
        assert "opaque-credential" not in text


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
