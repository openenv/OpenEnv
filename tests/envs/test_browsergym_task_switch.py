"""Unit tests for BrowserGym task switching behavior."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    sys.modules["gymnasium"] = types.SimpleNamespace(
        make=lambda *_args, **_kwargs: None
    )
    _INSERTED_GYMNASIUM_STUB = True
else:
    _INSERTED_GYMNASIUM_STUB = False

from envs.browsergym_env.server import browsergym_environment
from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment

if _INSERTED_GYMNASIUM_STUB:
    sys.modules.pop("gymnasium", None)


class _FakeGymEnv:
    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.closed = False
        self.reset_calls: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any):
        self.reset_calls.append(dict(kwargs))
        return {"goal": f"goal for {self.env_id}", "url": "http://example.test"}, {}

    def close(self) -> None:
        self.closed = True


def _make_browsergym_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> BrowserGymEnvironment:
    monkeypatch.setattr(
        browsergym_environment.importlib,
        "import_module",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        browsergym_environment.gym,
        "make",
        lambda env_id, **_kwargs: _FakeGymEnv(env_id),
    )
    return BrowserGymEnvironment(benchmark="miniwob", task_name="click-test")


def test_reset_preserves_requested_episode_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_browsergym_environment(monkeypatch)

    env.reset(episode_id="requested-episode")

    assert env.state.episode_id == "requested-episode"


def test_reset_async_preserves_requested_episode_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_browsergym_environment(monkeypatch)

    asyncio.run(env.reset_async(episode_id="requested-async-episode"))

    assert env.state.episode_id == "requested-async-episode"


def test_reset_preserves_positional_task_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_browsergym_environment(monkeypatch)

    env.reset(None, "enter-text")

    assert env.state.task_name == "enter-text"


def test_reset_preserves_empty_episode_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _make_browsergym_environment(monkeypatch)

    env.reset(episode_id="")

    assert env.state.episode_id == ""


def test_reset_rebuilds_browsergym_env_when_task_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_envs: list[tuple[str, dict[str, Any], _FakeGymEnv]] = []

    monkeypatch.setattr(
        browsergym_environment.importlib,
        "import_module",
        lambda _name: object(),
    )

    def fake_make(env_id: str, **kwargs: Any) -> _FakeGymEnv:
        env = _FakeGymEnv(env_id)
        created_envs.append((env_id, dict(kwargs), env))
        return env

    monkeypatch.setattr(browsergym_environment.gym, "make", fake_make)

    env = BrowserGymEnvironment(
        benchmark="miniwob",
        task_name="click-test",
        headless=True,
    )
    original_env = env.gym_env

    observation = env.reset(task_name="enter-text", seed=7)

    assert [env_id for env_id, _, _ in created_envs] == [
        "browsergym/miniwob.click-test",
        "browsergym/miniwob.enter-text",
    ]
    assert original_env.closed is True
    assert env.gym_env is created_envs[1][2]
    assert env.task_name == "enter-text"
    assert env.state.task_name == "enter-text"
    assert created_envs[1][2].reset_calls == [{"seed": 7}]
    assert observation.goal == "goal for browsergym/miniwob.enter-text"


def test_failed_task_switch_keeps_previous_browsergym_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail_enter_text = True
    created_envs: list[_FakeGymEnv] = []

    monkeypatch.setattr(
        browsergym_environment.importlib,
        "import_module",
        lambda _name: object(),
    )

    def fake_make(env_id: str, **_kwargs: Any) -> _FakeGymEnv:
        if env_id == "browsergym/miniwob.enter-text" and fail_enter_text:
            raise RuntimeError("task is unavailable")
        env = _FakeGymEnv(env_id)
        created_envs.append(env)
        return env

    monkeypatch.setattr(browsergym_environment.gym, "make", fake_make)

    env = BrowserGymEnvironment(benchmark="miniwob", task_name="click-test")
    original_env = env.gym_env

    with pytest.raises(ValueError, match="browsergym/miniwob.enter-text"):
        env.reset(task_name="enter-text")

    assert env.gym_env is original_env
    assert original_env.closed is False
    assert env.task_name == "click-test"
    assert env.env_id == "browsergym/miniwob.click-test"
    assert env.state.task_name == "click-test"

    fail_enter_text = False
    observation = env.reset(task_name="enter-text")

    assert original_env.closed is True
    assert env.gym_env is created_envs[-1]
    assert env.task_name == "enter-text"
    assert env.env_id == "browsergym/miniwob.enter-text"
    assert observation.goal == "goal for browsergym/miniwob.enter-text"
