import json
from dataclasses import FrozenInstanceError

import pytest
import yaml
from conftest import FIXTURES, load_fixture_manifest
from openenv.validation.manifest import (
    ExecutionDeclaration,
    ManifestError,
    NetworkPolicy,
    NormalizedManifest,
    NormalizedManifestV2,
    ResourceDeclaration,
)
from openenv.validation.parsers.openenv_yaml import OpenEnvYamlParser
from openenv.validation.policy import load_policy
from openenv.validation.report import ValidationReport, ValidationReportV2
from openenv.validation.runner import run_validation
from openenv.validation.runtime.contracts import (
    LaunchSpec,
    load_runtime_plan,
    MAX_PLAN_BYTES,
    RuntimeEvidence,
    RuntimePlan,
    RuntimePlanError,
    WireExchange,
)
from openenv.validation.types import Lane, Level, SignatureKind, Verdict
from pydantic import ValidationError


def plan_data():
    return {
        "plan_schema_version": "1",
        "reset": {"episode_id": "probe-episode", "seed": 42},
        "actions": [{"increment": 1}],
    }


def write_plan(root, payload):
    directory = root / "validation"
    directory.mkdir(exist_ok=True)
    (directory / "runtime.json").write_text(payload)


def test_runtime_plan_roundtrip_and_pure_read(tmp_path):
    data = plan_data()
    write_plan(tmp_path, json.dumps(data))
    (tmp_path / "server.py").write_text("raise RuntimeError('must not import')")
    plan = load_runtime_plan(tmp_path, ExecutionDeclaration())
    assert RuntimePlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.reset.episode_id == "probe-episode"


@pytest.mark.parametrize(
    "path", ["../secret", "/secret", "a/../../b", "a\\b", "C:/x", ""]
)
@pytest.mark.parametrize("field", ["probe_path", "dockerfile", "context"])
def test_execution_paths_reject_nonportable_or_escaped_locations(field, path):
    with pytest.raises(ValidationError, match="package-relative"):
        ExecutionDeclaration(**{field: path})


def test_execution_defaults_to_no_credential_requirement():
    assert ExecutionDeclaration().requires_credentials is False


@pytest.mark.parametrize("required", [False, True])
def test_execution_accepts_explicit_credential_requirement(required):
    assert (
        ExecutionDeclaration(requires_credentials=required).requires_credentials
        is required
    )


@pytest.mark.parametrize("required", ["true", "false", 0, 1, None, [], {}])
def test_execution_credential_requirement_requires_boolean(required):
    with pytest.raises(ValidationError, match="valid boolean"):
        ExecutionDeclaration(requires_credentials=required)


@pytest.mark.parametrize("model", [NormalizedManifest, ValidationReport])
def test_credential_declaration_does_not_change_schema_one(model):
    assert "requires_credentials" not in json.dumps(model.model_json_schema())


def test_plan_symlink_cannot_escape_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "validation").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "runtime.json").write_text(json.dumps(plan_data()))
    with pytest.raises(RuntimePlanError, match="inside the package"):
        load_runtime_plan(package, ExecutionDeclaration())


@pytest.mark.parametrize(
    "payload", ["x" * (MAX_PLAN_BYTES + 1), " " * MAX_PLAN_BYTES + "{}"]
)
def test_plan_reader_enforces_bytes_before_json_parsing(tmp_path, payload):
    write_plan(tmp_path, payload)
    with pytest.raises(RuntimePlanError, match="exceeds"):
        load_runtime_plan(tmp_path, ExecutionDeclaration())


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e10000"])
def test_nonfinite_json_is_rejected_before_pydantic_coercion(tmp_path, number):
    payload = json.dumps(plan_data()).replace(
        '"increment": 1', f'"increment": {number}'
    )
    write_plan(tmp_path, payload)
    with pytest.raises(RuntimePlanError, match="finite"):
        load_runtime_plan(tmp_path, ExecutionDeclaration())


def test_duplicate_json_fields_are_not_ambiguous(tmp_path):
    write_plan(tmp_path, '{"plan_schema_version":"1","plan_schema_version":"2"}')
    with pytest.raises(RuntimePlanError, match="duplicate"):
        load_runtime_plan(tmp_path, ExecutionDeclaration())


def test_plan_schema_errors_omit_inputs_and_unknown_field_names(tmp_path):
    data = plan_data()
    data["reset"]["seed"] = "private-input-value"
    data["private-extra-field"] = "private-extra-value"
    write_plan(tmp_path, json.dumps(data))

    with pytest.raises(RuntimePlanError) as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())

    diagnostic = str(error.value)
    assert "reset.seed: int_type" in diagnostic
    assert "<field>: extra_forbidden" in diagnostic
    assert "private-" not in diagnostic


def test_plan_schema_error_diagnostics_are_bounded(tmp_path):
    data = plan_data()
    data["actions"] = ["private-input-value"] * 100
    write_plan(tmp_path, json.dumps(data))

    with pytest.raises(RuntimePlanError) as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())

    diagnostic = str(error.value)
    assert diagnostic.count("dict_type") == 5
    assert "100 errors total" in diagnostic
    assert "private-input-value" not in diagnostic
    assert len(diagnostic) < 1024


def test_plan_schema_error_preserves_safe_validator_reason(tmp_path):
    data = plan_data()
    data["reset"]["options"] = {"seed": "private-input-value"}
    write_plan(tmp_path, json.dumps(data))
    with pytest.raises(RuntimePlanError) as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())
    assert "reset options cannot override episode_id or seed" in str(error.value)
    assert "private-input-value" not in str(error.value)


def test_duplicate_json_diagnostic_omits_key(tmp_path):
    write_plan(tmp_path, '{"private-duplicate-key": 1, "private-duplicate-key": 2}')
    with pytest.raises(RuntimePlanError, match="duplicate runtime JSON key") as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())
    assert "private-duplicate-key" not in str(error.value)


def test_missing_plan_diagnostic_omits_filesystem_paths(tmp_path):
    with pytest.raises(RuntimePlanError) as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())
    assert str(error.value) == "runtime plan could not be read (FileNotFoundError)"


def test_invalid_plan_json_reports_location_without_input(tmp_path):
    write_plan(tmp_path, '{"private-key": private-value}')
    with pytest.raises(RuntimePlanError) as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())
    assert str(error.value) == (
        "invalid runtime plan JSON at line 1, column 17: Expecting value"
    )


def test_invalid_plan_encoding_does_not_echo_bytes(tmp_path):
    write_plan(tmp_path, "")
    (tmp_path / "validation/runtime.json").write_bytes(b"\xffprivate-value")
    with pytest.raises(RuntimePlanError) as error:
        load_runtime_plan(tmp_path, ExecutionDeclaration())
    assert str(error.value) == "runtime plan contains invalid text encoding"


@pytest.mark.parametrize("override", ["episode_id", "seed"])
def test_reset_options_cannot_replace_reproducibility_inputs(override):
    data = plan_data()
    data["reset"]["options"] = {override: "other"}
    with pytest.raises(ValidationError, match="cannot override"):
        RuntimePlan.model_validate(data)


@pytest.mark.parametrize("seed", [True, "42", -1, 2**32])
def test_reset_seed_is_explicit_bounded_integer(seed):
    data = plan_data()
    data["reset"]["seed"] = seed
    with pytest.raises(ValidationError):
        RuntimePlan.model_validate(data)


@pytest.mark.parametrize("actions", [[], [{}] * 101, [{"callback": object()}]])
def test_actions_are_bounded_nonempty_json_data(actions):
    data = plan_data()
    data["actions"] = actions
    with pytest.raises(ValidationError):
        RuntimePlan.model_validate(data)


def test_input_cannot_override_manifest_capabilities():
    data = plan_data()
    data["capabilities"] = {"set_state": True}
    with pytest.raises(ValidationError, match="Extra inputs"):
        RuntimePlan.model_validate(data)


def test_nested_inputs_are_bounded():
    data = plan_data()
    nested = {}
    for _ in range(40):
        nested = {"child": nested}
    data["actions"] = [nested]
    with pytest.raises(ValidationError, match="nesting"):
        RuntimePlan.model_validate(data)


def test_direct_plan_construction_also_has_size_bound():
    data = plan_data()
    data["actions"] = [{"text": "x" * MAX_PLAN_BYTES}]
    with pytest.raises(ValidationError, match="exceeds"):
        RuntimePlan.model_validate(data)


def test_parser_opts_into_v2_only_when_execution_is_present(tmp_path):
    source = yaml.safe_load((FIXTURES / "served_min_pass" / "openenv.yaml").read_text())
    path = tmp_path / "openenv.yaml"
    path.write_text(yaml.safe_dump(source))
    v1 = OpenEnvYamlParser().parse(tmp_path)
    assert type(v1) is NormalizedManifest
    source["validation"]["execution"] = {"kind": "openenv_ws"}
    path.write_text(yaml.safe_dump(source))
    v2 = OpenEnvYamlParser().parse(tmp_path)
    assert type(v2) is NormalizedManifestV2
    assert v2.manifest_schema_version == "2"
    assert NormalizedManifestV2.model_validate_json(v2.model_dump_json()) == v2
    assert v2.model_dump(
        exclude={"execution", "manifest_schema_version"}
    ) == v1.model_dump(exclude={"manifest_schema_version"})


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("reward", "range", [0.0, float("inf")]),
        ("reward", "range", [float("-inf"), 1.0]),
        ("reward", "floor_margin", float("inf")),
        ("reward", "oracle_tolerance", float("inf")),
        ("reward", "variance_tolerance", float("inf")),
        ("resources", "cpu", float("inf")),
        ("resources", "episode_timeout_s", float("inf")),
    ],
)
def test_v2_rejects_nonfinite_declarations_without_changing_v1(section, field, value):
    data = load_fixture_manifest("served_min_pass")
    data[section][field] = value
    assert NormalizedManifest.model_validate(data).manifest_schema_version == "1"
    data.update(manifest_schema_version="2", execution={})
    with pytest.raises(ValidationError, match="numeric declarations must be finite"):
        NormalizedManifestV2.model_validate(data)


def test_v2_rejects_nonfinite_judge_parameters():
    data = load_fixture_manifest("served_min_pass")
    data.update(
        manifest_schema_version="2",
        execution={},
        judge={
            "model": "judge",
            "version": "1",
            "params": {"temperature": float("nan")},
        },
    )
    data["capabilities"]["llm_judged"] = True
    data["reward"]["variance_tolerance"] = 0.1
    with pytest.raises(ValidationError, match="numeric declarations must be finite"):
        NormalizedManifestV2.model_validate(data)


def test_null_execution_is_not_silently_downgraded(tmp_path):
    source = yaml.safe_load((FIXTURES / "served_min_pass" / "openenv.yaml").read_text())
    source["validation"]["execution"] = None
    (tmp_path / "openenv.yaml").write_text(yaml.safe_dump(source))
    with pytest.raises(ManifestError, match="execution"):
        OpenEnvYamlParser().parse(tmp_path)


def test_static_validation_preserves_v2_execution_in_versioned_report(tmp_path):
    source = yaml.safe_load((FIXTURES / "served_min_pass" / "openenv.yaml").read_text())
    source["validation"]["execution"] = {"kind": "openenv_ws"}
    (tmp_path / "openenv.yaml").write_text(yaml.safe_dump(source))
    report = run_validation(tmp_path, max_level=Level.STATIC)
    payload = json.loads(report.model_dump_json())
    assert payload["report_schema_version"] == "2"
    assert payload["manifest"]["execution"]["probe_path"] == "validation/runtime.json"
    assert ValidationReportV2.model_validate_json(report.model_dump_json()) == report
    from jsonschema import validate

    schema_path = (
        FIXTURES.parents[2] / "src/openenv/validation/schemas/report-v2.schema.json"
    )
    validate(payload, json.loads(schema_path.read_text()))


@pytest.mark.parametrize("version", [None, "1", "2"])
def test_v2_report_roundtrips_null_and_both_manifest_versions(version):
    manifest = None
    if version is not None:
        data = load_fixture_manifest("served_min_pass")
        data["manifest_schema_version"] = version
        model = NormalizedManifest
        if version == "2":
            data["execution"] = {}
            model = NormalizedManifestV2
        manifest = model.model_validate(data)
    report = ValidationReportV2(
        report_schema_version="2",
        target="subject",
        source_digest="0" * 64,
        signature=SignatureKind.OPENENV_SERVED,
        manifest=manifest,
        policy_version="v2",
        lane=Lane.LOCAL,
        levels_run=[Level.STATIC],
        results=[],
        verdict=Verdict.WARN,
    )
    roundtrip = ValidationReportV2.model_validate_json(report.model_dump_json())
    assert roundtrip == report
    assert type(roundtrip.manifest) is type(manifest)
    with pytest.raises(ValidationError):
        ValidationReport.model_validate_json(report.model_dump_json())


def launch_data():
    return {
        "image_ref": "sha256:" + "a" * 64,
        "run_id": "validation-contract-test",
        "resources": ResourceDeclaration(
            cpu=1, memory_mb=256, disk_mb=64, episode_timeout_s=30
        ),
        "network": NetworkPolicy(),
    }


@pytest.mark.parametrize(
    "image", ["probe:latest", "sha256:abc", "repo@sha256:" + "z" * 64]
)
def test_launch_rejects_mutable_or_invalid_image_identity(image):
    data = launch_data()
    data["image_ref"] = image
    with pytest.raises(ValidationError):
        LaunchSpec(**data)


def test_launch_defaults_do_not_inherit_host_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    spec = LaunchSpec(**launch_data())
    assert spec.env_vars == {}
    assert spec.startup_timeout_s == 30
    with pytest.raises(ValidationError, match="frozen"):
        spec.image_ref = "sha256:" + "b" * 64


@pytest.mark.parametrize("env", [{"BAD=NAME": "x"}, {"NAME": "a\0b"}])
def test_launch_rejects_unsafe_explicit_environment(env):
    with pytest.raises(ValidationError):
        LaunchSpec(**launch_data(), env_vars=env)


def test_launch_rejects_infinite_budget():
    data = launch_data()
    data["resources"].cpu = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        LaunchSpec(**data)


def test_evidence_records_preserve_raw_wire_and_are_immutable():
    exchange = WireExchange("step", '{"increment":1}', '{"reward":true}')
    evidence = RuntimeEvidence(exchanges=(exchange,))
    assert evidence.exchanges[0].response_json == '{"reward":true}'
    with pytest.raises(FrozenInstanceError):
        exchange.response_json = '{"reward":1}'
    with pytest.raises(FrozenInstanceError):
        evidence.failure_reason = "altered"


def test_policy_v2_adds_only_startup_and_preserves_every_v1_rule():
    v1 = load_policy("v1")
    v2 = load_policy("v2")
    assert v2.bounds == v1.bounds
    assert [
        entry for entry in v2.entries if entry.check_id != "runtime.startup"
    ] == v1.entries
    startup = v2.entries_for_lane(Lane.LOCAL)["runtime.startup"]
    assert startup.level is Level.RUNTIME
    assert startup.severity.value == "fail"


def test_acceptance_catalog_covers_exact_policy_runtime_inventory():
    catalog = json.loads((FIXTURES / "runtime" / "cases.json").read_text())
    assert catalog["catalog_schema_version"] == "1"
    checks = catalog["checks"]
    assert len({check["check_id"] for check in checks}) == len(checks)
    expected = {
        check_id
        for check_id, entry in load_policy("v2").entries_for_lane(Lane.LOCAL).items()
        if entry.level is Level.RUNTIME
    }
    assert {check["check_id"] for check in checks} == expected
    cases = {case["case_id"]: case for case in catalog["cases"]}
    for check in checks:
        for case_id, status in (
            (check["positive_case"], "pass"),
            (check["negative_case"], "fail"),
        ):
            assert cases[case_id]["expected"][check["check_id"]] == status
            assert cases[case_id]["evidence_predicate"]
