"""The trusted Harbor contract accepts explicit numeric JSON declarations."""

import json
import re
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from openenv.cli.__main__ import app
from openenv.validation.manifest import ManifestError
from openenv.validation.parsers import harbor_toml
from openenv.validation.parsers.harbor_toml import read_harbor_contract
from typer.testing import CliRunner


@pytest.fixture
def contract_package(tmp_path):
    (tmp_path / "validation").mkdir()
    contract = {
        "profile": "harbor-semantic-v1",
        "reward_key": "reward",
        "reward": {"range": [0, 1], "oracle_tolerance": 0, "floor_margin": 0.1},
        "resources": {
            "cpu": 1,
            "memory_mb": 512,
            "disk_mb": 1024,
            "episode_timeout_s": 10,
        },
        "types": {"tags": ["swe"]},
    }
    (tmp_path / "validation/openenv.json").write_text(json.dumps(contract))
    return tmp_path


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("reward", "range", ["0", "1"]),
        ("reward", "oracle_tolerance", "0.1"),
        ("reward", "floor_margin", "0.100000000000000001"),
        ("reward", "variance_tolerance", "0.01"),
        ("resources", "cpu", "1"),
        ("resources", "memory_mb", "512"),
        ("resources", "disk_mb", "1024"),
        ("resources", "episode_timeout_s", "10"),
        ("resources", "gpus", "1"),
    ],
)
def test_numeric_strings_are_rejected(contract_package, section, field, value):
    path = contract_package / "validation/openenv.json"
    contract = json.loads(path.read_text())
    contract[section][field] = value
    path.write_text(json.dumps(contract))

    with pytest.raises(ManifestError):
        read_harbor_contract(contract_package)


def test_gpu_type_strings_are_preserved(contract_package):
    path = contract_package / "validation/openenv.json"
    contract = json.loads(path.read_text())
    contract["resources"].update(gpus=1, gpu_types=["A100", "H100"])
    path.write_text(json.dumps(contract))

    loaded = read_harbor_contract(contract_package)
    assert loaded.resources.gpu_types == ["A100", "H100"]
    assert loaded.resources.gpus == 1


@pytest.mark.parametrize("hostile_input", ["huge_integer", "deep_nesting"])
def test_hostile_json_is_a_manifest_failure(contract_package, hostile_input):
    path = contract_package / "validation/openenv.json"
    if hostile_input == "huge_integer":
        payload = path.read_text().replace('"cpu": 1', '"cpu": ' + "9" * 400)
    else:
        payload = "[" * 100_000 + "0" + "]" * 100_000
    path.write_text(payload)

    with pytest.raises(ManifestError):
        read_harbor_contract(contract_package)


@pytest.mark.parametrize("installed_version", ["0.22.0", "0.26.0"])
def test_harbor_version_mismatch_is_a_validator_error(
    contract_package, monkeypatch, installed_version
):
    (contract_package / "task.toml").write_text('schema_version = "1.4"\n')
    monkeypatch.setattr(harbor_toml, "version", lambda name: installed_version)

    result = CliRunner().invoke(
        app, ["validate", str(contract_package), "--level", "static", "--skip-build"]
    )

    assert result.exit_code == 3, result.output
    assert "harbor==0.25.0" in result.output
    assert installed_version in result.output
    assert "static.manifest" not in result.output


def test_missing_harbor_is_a_validator_error(contract_package, monkeypatch):
    (contract_package / "task.toml").write_text('schema_version = "1.4"\n')

    def missing_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(harbor_toml, "version", missing_version)
    result = CliRunner().invoke(
        app, ["validate", str(contract_package), "--level", "static", "--skip-build"]
    )

    assert result.exit_code == 3, result.output
    assert "harbor==0.25.0" in result.output
    assert "static.manifest" not in result.output


def test_cli_reference_contract_loads(tmp_path):
    reference = Path(__file__).resolve().parents[2] / "docs/source/reference/cli.md"
    section = reference.read_text().split("Harbor `task.toml` directories", 1)[1]
    section = section.split("\n## ", 1)[0]
    example = re.search(r"```json\n(.*?)\n```", section, re.DOTALL)
    assert example is not None, "document a complete validation/openenv.json contract"
    assert "Python 3.12 or newer" in section
    assert "harbor==0.25.0" in section
    assert "openenv[harbor]" in section
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation/openenv.json").write_text(example[1])

    contract = read_harbor_contract(tmp_path)

    assert contract.profile == "harbor-semantic-v1"
    assert contract.reward_key == "reward"
    assert contract.types.tags == ["swe"]
