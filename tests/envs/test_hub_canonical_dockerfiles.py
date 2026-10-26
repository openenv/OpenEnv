"""Deployment defaults for first-party Hub canonicals."""

from __future__ import annotations

from pathlib import Path

import pytest


CANONICAL_ENVS = (
    "browsergym_env",
    "chat_env",
    "echo_env",
    "repl_env",
    "sumo_rl_env",
    "tbench2_env",
    "textarena_env",
)


@pytest.mark.parametrize("env_name", CANONICAL_ENVS)
def test_canonical_dockerfiles_enable_web_interface(env_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "envs" / env_name / "server" / "Dockerfile"

    dockerfile_text = dockerfile.read_text()

    assert "ENV ENABLE_WEB_INTERFACE=true" in dockerfile_text
