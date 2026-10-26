"""Tests for the OpenEnv PyPI backfill helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "backfill_openenv_pypi.py"
)
SPEC = importlib.util.spec_from_file_location("backfill_openenv_pypi", SCRIPT_PATH)
assert SPEC is not None
backfill_openenv_pypi = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["backfill_openenv_pypi"] = backfill_openenv_pypi
SPEC.loader.exec_module(backfill_openenv_pypi)

numeric_version_key = backfill_openenv_pypi.numeric_version_key
rewrite_source_tree = backfill_openenv_pypi.rewrite_source_tree
select_versions = backfill_openenv_pypi.select_versions


def test_select_versions_defaults_to_backfill_0_2_releases() -> None:
    source_releases = {
        "0.1.0": [{}],
        "0.2.0": [{}],
        "0.2.1": [{}],
        "0.2.2": [{}],
        "0.2.3": [{}],
        "0.3.0": [{}],
    }

    versions = select_versions(
        source_releases=source_releases,
        target_releases={},
        explicit_versions=None,
        from_version="0.2.0",
        before_version="0.3.0",
        skip_existing=False,
    )

    assert versions == ["0.2.0", "0.2.1", "0.2.2", "0.2.3"]


def test_select_versions_refuses_existing_target_release() -> None:
    with pytest.raises(RuntimeError, match="target release already exists"):
        select_versions(
            source_releases={"0.2.0": [{}]},
            target_releases={"0.2.0": [{"filename": "openenv-0.2.0.tar.gz"}]},
            explicit_versions=["0.2.0"],
            from_version="0.2.0",
            before_version="0.3.0",
            skip_existing=False,
        )


def test_numeric_version_key_rejects_non_numeric_versions() -> None:
    with pytest.raises(ValueError, match="Only numeric release versions"):
        numeric_version_key("0.2.0rc1")


def test_rewrite_source_tree_renames_project_and_removes_egg_info(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "openenv-core"\nversion = "0.2.0"\n'
        '[project.optional-dependencies]\nall = ["openenv-core[core]"]\n',
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text("pip install openenv-core\n", encoding="utf-8")
    egg_info = tmp_path / "src" / "openenv_core.egg-info"
    egg_info.mkdir(parents=True)
    (egg_info / "PKG-INFO").write_text("Name: openenv-core\n", encoding="utf-8")

    rewrite_source_tree(tmp_path, "openenv-core", "openenv")

    assert 'name = "openenv"' in pyproject.read_text(encoding="utf-8")
    assert "openenv[core]" in pyproject.read_text(encoding="utf-8")
    assert readme.read_text(encoding="utf-8") == "pip install openenv\n"
    assert not egg_info.exists()
