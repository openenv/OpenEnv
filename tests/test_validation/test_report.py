import pytest
from conftest import load_fixture_manifest
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.report import CheckResult, ValidationReport
from openenv.validation.types import CheckStatus, Lane, Level, SignatureKind, Verdict
from pydantic import ValidationError


def test_check_result_ids_are_namespaced():
    with pytest.raises(ValidationError):
        CheckResult(check_id="not-namespaced", status=CheckStatus.PASS, duration_s=0.0)


def test_check_result_duration_is_non_negative():
    with pytest.raises(ValidationError):
        CheckResult(
            check_id="static.manifest", status=CheckStatus.PASS, duration_s=-1.0
        )


def test_report_round_trips_with_embedded_manifest():
    manifest = NormalizedManifest.model_validate(
        load_fixture_manifest("served_min_pass")
    )
    report = ValidationReport(
        report_schema_version="1",
        target="tests/fixtures/validation/served_min_pass",
        source_digest="0" * 64,
        signature=SignatureKind.OPENENV_SERVED,
        manifest=manifest,
        policy_version="v1",
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC],
        results=[
            CheckResult(
                check_id="static.manifest",
                status=CheckStatus.PASS,
                measured={"fields": 9},
                evidence=["manifest validates against schema version 1"],
                duration_s=0.01,
            )
        ],
        verdict=Verdict.PASS,
    )
    round_tripped = ValidationReport.model_validate_json(report.model_dump_json())
    assert round_tripped == report
    assert round_tripped.manifest == manifest


def test_report_allows_null_manifest_when_schema_failed():
    report = ValidationReport(
        report_schema_version="1",
        target="tests/fixtures/validation/broken_manifest",
        source_digest="0" * 64,
        signature=SignatureKind.OPENENV_SERVED,
        manifest=None,
        policy_version="v1",
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC],
        results=[
            CheckResult(
                check_id="static.manifest",
                status=CheckStatus.FAIL,
                evidence=["reward.range: reward range must be strictly increasing"],
                duration_s=0.01,
            )
        ],
        verdict=Verdict.FAIL,
    )
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report
    assert report.manifest is None


def test_report_schema_version_is_pinned():
    manifest = NormalizedManifest.model_validate(
        load_fixture_manifest("served_min_pass")
    )
    with pytest.raises(ValidationError):
        ValidationReport(
            report_schema_version="999",
            target="x",
            source_digest="0" * 64,
            signature=SignatureKind.OPENENV_SERVED,
            manifest=manifest,
            policy_version="v1",
            lane=Lane.LOCAL,
            levels_run=[Level.STATIC],
            results=[],
            verdict=Verdict.PASS,
        )
