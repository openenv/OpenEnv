import pytest
from conftest import FIXTURES, load_fixture_manifest
from openenv.validation.manifest import ManifestError, NormalizedManifest
from openenv.validation.parsers.openenv_yaml import OpenEnvYamlParser
from openenv.validation.runner import default_parser_registry
from openenv.validation.signature import detect_signature
from openenv.validation.types import SignatureKind


def test_parse_matches_the_committed_golden_manifest():
    parsed = OpenEnvYamlParser().parse(FIXTURES / "served_min_pass")
    golden = NormalizedManifest.model_validate(load_fixture_manifest("served_min_pass"))
    assert parsed == golden


def test_broken_manifest_raises_manifest_error_with_field_evidence():
    with pytest.raises(ManifestError) as exc_info:
        OpenEnvYamlParser().parse(FIXTURES / "broken_manifest")
    evidence = "\n".join(exc_info.value.errors)
    assert "reward.range" in evidence or "strictly increasing" in evidence
    assert "resources" in evidence


def test_unpinned_judge_raises_manifest_error_on_the_judge_pin():
    with pytest.raises(ManifestError) as exc_info:
        OpenEnvYamlParser().parse(FIXTURES / "unpinned_judge")
    assert any("judge pin" in error for error in exc_info.value.errors)


def test_missing_validation_block_has_remediation(tmp_path):
    (tmp_path / "openenv.yaml").write_text("spec_version: 1\nname: bare\n")
    with pytest.raises(ManifestError) as exc_info:
        OpenEnvYamlParser().parse(tmp_path)
    assert "validation" in exc_info.value.errors[0]
    assert exc_info.value.remediation is not None


def test_parse_is_a_pure_read(tmp_path):
    src = (FIXTURES / "served_min_pass" / "openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(src)
    (tmp_path / "server.py").write_text(
        "raise SystemExit('parser imported package code')\n"
    )
    parsed = OpenEnvYamlParser().parse(tmp_path)
    assert parsed.name == "served-min-pass"


def test_default_registry_dispatches_openenv_yaml_end_to_end():
    signature = detect_signature(FIXTURES / "served_min_pass")
    assert signature is SignatureKind.OPENENV_SERVED
    manifest = (
        default_parser_registry()
        .parser_for(signature)
        .parse(FIXTURES / "served_min_pass")
    )
    assert manifest.name == "served-min-pass"


def test_network_policy_parses_and_defaults_to_public(tmp_path):
    src = (FIXTURES / "served_min_pass" / "openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(
        src
        + "  network:\n    mode: allowlist\n    allowed_hosts: [api.example.com, 10.0.0.0/8]\n"
    )
    parsed = OpenEnvYamlParser().parse(tmp_path)
    assert parsed.network.mode == "allowlist"
    assert parsed.network.allowed_hosts == ["api.example.com", "10.0.0.0/8"]

    default = OpenEnvYamlParser().parse(FIXTURES / "served_min_pass")
    assert default.network.mode == "public"
    assert default.network.allowed_hosts == []


def test_allowed_hosts_without_allowlist_mode_is_a_manifest_error(tmp_path):
    src = (FIXTURES / "served_min_pass" / "openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(
        src + "  network:\n    allowed_hosts: [api.example.com]\n"
    )
    with pytest.raises(ManifestError) as exc_info:
        OpenEnvYamlParser().parse(tmp_path)
    assert any("allowlist" in error for error in exc_info.value.errors)
