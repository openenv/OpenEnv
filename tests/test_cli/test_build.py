# SPDX-License-Identifier: BSD-3-Clause

"""Tests for openenv build helpers."""

from pathlib import Path

from openenv.cli.commands.build import (
    _detect_build_context,
    _docker_build_command,
    _parse_build_args,
)


def test_detect_build_context_uses_envs_child_as_in_repo(tmp_path: Path) -> None:
    """An environment below repo_root/envs uses the repository as build context."""
    repo_root = tmp_path / "repo"
    env_path = repo_root / "envs" / "example_env"
    env_path.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert _detect_build_context(env_path) == (
        "in-repo",
        repo_root.absolute(),
        repo_root.absolute(),
    )


def test_detect_build_context_keeps_repo_sibling_standalone(tmp_path: Path) -> None:
    """A repo-local directory outside envs/ remains a standalone environment."""
    repo_root = tmp_path / "repo"
    env_path = repo_root / "examples" / "example_env"
    env_path.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert _detect_build_context(env_path) == (
        "standalone",
        env_path.absolute(),
        None,
    )


def test_detect_build_context_keeps_non_git_path_standalone(tmp_path: Path) -> None:
    """A path outside any repository remains a standalone environment."""
    env_path = tmp_path / "example_env"
    env_path.mkdir()

    assert _detect_build_context(env_path) == (
        "standalone",
        env_path.absolute(),
        None,
    )


def test_parse_build_args_preserves_values_with_equals() -> None:
    """Build arg parsing only splits on the first equals sign."""
    assert _parse_build_args(["ENV=prod", "TOKEN=a=b=c"]) == {
        "ENV": "prod",
        "TOKEN": "a=b=c",
    }


def test_parse_build_args_warns_and_skips_invalid(capsys) -> None:
    """Malformed build args keep the existing warning-and-skip behavior."""
    assert _parse_build_args(["missing_equals", "ENV=prod"]) == {"ENV": "prod"}

    captured = capsys.readouterr()
    assert "Warning: Invalid build arg format: missing_equals" in captured.err


def test_docker_build_command_preserves_existing_order(tmp_path: Path) -> None:
    """Docker command assembly keeps the existing option ordering."""
    dockerfile = tmp_path / "Dockerfile"
    build_dir = tmp_path / "build"

    assert _docker_build_command(
        tag="openenv-example",
        dockerfile=dockerfile,
        build_dir=build_dir,
        build_args={"BUILD_MODE": "standalone", "ENV_NAME": "example"},
        no_cache=False,
    ) == [
        "docker",
        "build",
        "-t",
        "openenv-example",
        "-f",
        str(dockerfile),
        "--build-arg",
        "BUILD_MODE=standalone",
        "--build-arg",
        "ENV_NAME=example",
        str(build_dir),
    ]


def test_docker_build_command_includes_no_cache_before_build_args(
    tmp_path: Path,
) -> None:
    """No-cache builds keep the current flag placement before build args."""
    dockerfile = tmp_path / "Dockerfile"
    build_dir = tmp_path / "build"

    assert _docker_build_command(
        tag="openenv-example",
        dockerfile=dockerfile,
        build_dir=build_dir,
        build_args={"TOKEN": "a=b=c"},
        no_cache=True,
    ) == [
        "docker",
        "build",
        "-t",
        "openenv-example",
        "-f",
        str(dockerfile),
        "--no-cache",
        "--build-arg",
        "TOKEN=a=b=c",
        str(build_dir),
    ]
