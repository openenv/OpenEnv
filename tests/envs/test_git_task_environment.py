# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Git task environment."""

import asyncio

import pytest
from git_env.server import git_task_environment
from git_env.server.git_task_environment import GitTaskEnvironment


class FakeGitServerClient:
    """Stands in for GitServerClient so no Gitea server is needed."""

    is_ready = True
    gitea_url = "http://gitea.test"

    def __init__(self, gitea_url, username, password, workspace_dir):
        self.calls = []

    def wait_for_ready(self):
        return True

    def workspace_exists(self, repo_name):
        return True

    def reset_workspace(self, repo_name, commit):
        self.calls.append(("reset_workspace", repo_name, commit))

    def get_current_commit(self, repo_name):
        return "abc123"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(git_task_environment, "GitServerClient", FakeGitServerClient)
    return GitTaskEnvironment(
        gitea_url="http://gitea.test",
        username="user",
        password="pass",
        task_repos={"task1": ("my-repo", "abc123")},
    )


class TestResetSeed:
    """reset(seed=..., episode_id=...) must be accepted.

    Resetting to a task's base state has no randomness, so a seed changes
    nothing. The server only passes arguments that `reset` declares, so an
    environment that omits them silently drops the caller's seed and episode id.
    """

    def test_task_id_is_still_the_first_parameter(self, env):
        obs = env.reset("task1")
        assert obs.success
        assert "task1" in obs.message
        assert env._git_client.calls == [("reset_workspace", "my-repo", "abc123")]

    def test_seeded_reset_gives_same_observation(self, env):
        unseeded = env.reset(task_id="task1")
        for seed in (0, 1, 1234):
            obs = env.reset(task_id="task1", seed=seed)
            assert obs.message == unseeded.message
            assert obs.output == unseeded.output
            assert env.state.step_count == 0

    def test_episode_id_is_kept_or_generated(self, env):
        env.reset(episode_id="run-1")
        assert env.state.episode_id == "run-1"
        env.reset(task_id="task1", seed=3, episode_id="run-2")
        assert env.state.episode_id == "run-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first

    def test_reset_async_forwards_seed_task_id_and_episode_id(self, env):
        obs = asyncio.run(
            env.reset_async(seed=3, episode_id="async-1", task_id="task1")
        )
        assert obs.success
        assert env.state.episode_id == "async-1"
