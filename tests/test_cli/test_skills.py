# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv skills command."""

import os
from pathlib import Path

from openenv.cli.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


def test_skills_add_installs_local_skill(tmp_path: Path) -> None:
    """openenv skills add installs to project .agents/skills by default."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["skills", "add"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    skill_md = tmp_path / ".agents" / "skills" / "openenv-cli" / "SKILL.md"
    assert skill_md.exists()
    assert "openenv" in skill_md.read_text().lower()


def test_skills_add_rejects_dest_with_agent_flags(tmp_path: Path) -> None:
    """--dest cannot be combined with assistant/global flags."""
    result = runner.invoke(
        app,
        ["skills", "add", "--dest", str(tmp_path), "--claude"],
    )

    assert result.exit_code == 1
    assert "--dest cannot be combined" in result.output


def test_skills_add_requires_force_when_target_exists(tmp_path: Path) -> None:
    """Existing destination requires --force to overwrite."""
    existing = tmp_path / "skills" / "openenv-cli"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text("old")

    result = runner.invoke(app, ["skills", "add", "--dest", str(tmp_path / "skills")])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_skills_add_force_overwrites_existing(tmp_path: Path) -> None:
    """--force overwrites existing skill content."""
    existing = tmp_path / "skills" / "openenv-cli"
    existing.mkdir(parents=True)
    skill_md = existing / "SKILL.md"
    skill_md.write_text("old")

    result = runner.invoke(
        app,
        ["skills", "add", "--dest", str(tmp_path / "skills"), "--force"],
    )

    assert result.exit_code == 0
    assert skill_md.read_text() != "old"


def test_skills_add_creates_agent_symlink(tmp_path: Path) -> None:
    """Assistant flag creates a symlink to the central skill location."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["skills", "add", "--claude"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    link_path = tmp_path / ".claude" / "skills" / "openenv-cli"
    target_path = tmp_path / ".agents" / "skills" / "openenv-cli"
    assert link_path.is_symlink()
    assert link_path.resolve() == target_path.resolve()


def test_skills_add_creates_multiple_agent_symlinks(tmp_path: Path) -> None:
    """Multiple assistant flags create symlinks to the same central skill."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["skills", "add", "--claude", "--codex"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    target_path = tmp_path / ".agents" / "skills" / "openenv-cli"
    for agent_dir in (".claude", ".codex"):
        link_path = tmp_path / agent_dir / "skills" / "openenv-cli"
        assert link_path.is_symlink()
        assert link_path.resolve() == target_path.resolve()
