# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GitServerClient git configuration."""

from pathlib import Path

from openenv.core.tools.git_server_client import GitServerClient


def test_git_config_does_not_touch_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)

    client = GitServerClient(
        gitea_url="http://gitea:3000",
        username="openenv",
        password="secret",
        workspace_dir=str(tmp_path / "workspace"),
    )

    assert list(home.iterdir()) == []

    # Git commands run by the client use the client's own identity
    assert client.execute_git_command("init repo")[0] == 0
    code, stdout, _ = client.execute_git_command("config --get user.email", "repo")
    assert code == 0
    assert stdout.strip() == "openenv@local.env"


def test_execute_git_command_keeps_quoted_args(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)

    client = GitServerClient(
        gitea_url="http://gitea:3000",
        username="openenv",
        password="secret",
        workspace_dir=str(tmp_path / "workspace"),
    )
    client.execute_git_command("init repo")

    code, _, stderr = client.execute_git_command(
        'commit --allow-empty -m "two words"', "repo"
    )
    assert code == 0, stderr
    _, stdout, _ = client.execute_git_command("log -1 --format=%s", "repo")
    assert stdout.strip() == "two words"
