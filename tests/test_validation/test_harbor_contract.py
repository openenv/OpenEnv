"""The trusted Harbor contract accepts explicit numeric JSON declarations."""

import json

import pytest
from openenv.validation.manifest import ManifestError
from openenv.validation.parsers.harbor_toml import read_harbor_contract


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
