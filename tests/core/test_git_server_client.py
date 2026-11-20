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


def _client(tmp_path, monkeypatch):
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
    assert client.execute_git_command("init repo")[0] == 0
    return client


@pytest.mark.parametrize(
    "command",
    [
        "-c alias.x='!touch {marker}' x",
        "config alias.x '!touch {marker}'",
        "rebase --exec 'touch {marker}' HEAD",
        "rebase --exe='touch {marker}' HEAD",
        "rebase -ix 'touch {marker}' HEAD",
        "grep -O'touch {marker}' x",
        "submodule foreach 'touch {marker}'",
    ],
)
def test_agent_commands_cannot_run_programs(tmp_path, monkeypatch, command):
    """An agent sends git arguments, and git can run programs through aliases and options."""
    client = _client(tmp_path, monkeypatch)
    marker = tmp_path / "ran"
    code, _, stderr = client.execute_git_command(command.format(marker=marker), "repo")
    assert code != 0
    assert "not available" in stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "command",
    [
        "commit --allow-empty -F {secret}",
        "diff --no-index {secret} /dev/null",
        "add --pathspec-from-file={secret}",
        "config --file {secret} --list",
    ],
)
def test_agent_commands_cannot_read_server_files(tmp_path, monkeypatch, command):
    client = _client(tmp_path, monkeypatch)
    secret = tmp_path / "token"
    secret.write_text("server-secret")
    code, stdout, stderr = client.execute_git_command(
        command.format(secret=secret), "repo"
    )
    assert code != 0
    assert "server-secret" not in stdout + stderr


def test_agent_commands_still_cover_everyday_git(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    for command in [
        'commit --allow-empty -m "first"',
        "switch -c feature",
        "checkout -b other",
        "log --oneline -n 1",
        "config --get user.email",
    ]:
        code, _, stderr = client.execute_git_command(command, "repo")
        assert code == 0, (command, stderr)


@pytest.mark.parametrize("working_dir", ["..", "../..", "/tmp"])
def test_working_dir_stays_in_the_workspace(tmp_path, monkeypatch, working_dir):
    client = _client(tmp_path, monkeypatch)
    code, _, stderr = client.execute_git_command("status", working_dir)
    assert code != 0
    assert "outside the workspace" in stderr


@pytest.mark.parametrize("target_dir", ["..", "../outside", "."])
def test_clone_never_deletes_outside_the_workspace(tmp_path, monkeypatch, target_dir):
    """Cloning first deletes the target directory, so it must be a folder inside the workspace."""
    client = _client(tmp_path, monkeypatch)
    client.is_ready = True
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside the workspace"):
        client.clone_to_workspace("repo", target_dir)
    assert outside.exists()
    assert (tmp_path / "workspace" / "repo").exists()


def test_git_runs_without_the_server_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "server-token")
    monkeypatch.setenv("GITEA_PASSWORD", "server-password")
    client = _client(tmp_path, monkeypatch)
    assert "HF_TOKEN" not in client._git_env
    assert "GITEA_PASSWORD" not in client._git_env
