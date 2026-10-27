# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GitServerClient git configuration."""

import subprocess
from pathlib import Path

import pytest
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


@pytest.mark.parametrize(
    "gitea_url, password",
    [
        ("http://gitea:3000", "p@ss/word#1:%"),
        ("https://gitea.example.com", "secret"),
    ],
)
def test_stored_credentials_match_gitea_url(tmp_path, monkeypatch, gitea_url, password):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)

    client = GitServerClient(
        gitea_url=gitea_url,
        username="openenv",
        password=password,
        workspace_dir=str(tmp_path / "workspace"),
    )

    # Ask git for the credentials it would send when cloning from the server
    result = subprocess.run(
        ["git", "credential", "fill"],
        input=f"url={gitea_url}/openenv/repo.git\n\n",
        env={**client._git_env, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "username=openenv" in result.stdout.splitlines()
    assert f"password={password}" in result.stdout.splitlines()
