"""Unit tests for BrowserGym thread affinity."""

from __future__ import annotations

import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
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

from envs.browsergym_env.models import BrowserGymAction
from envs.browsergym_env.server import browsergym_environment
from envs.browsergym_env.server.browsergym_environment import BrowserGymEnvironment

if _INSERTED_GYMNASIUM_STUB:
    sys.modules.pop("gymnasium", None)


class _ThreadRecordingGymEnv:
    """Records the thread of every call, like Playwright's sync API cares about."""

    def __init__(self, threads: list[int]) -> None:
        self.threads = threads

    def reset(self, **_kwargs: Any):
        self.threads.append(threading.get_ident())
        return {"goal": "goal", "url": "http://example.test"}, {}

    def step(self, _action: str):
        self.threads.append(threading.get_ident())
        return {"goal": "goal", "url": "http://example.test"}, 0.0, False, False, {}

    def close(self) -> None:
        self.threads.append(threading.get_ident())


def test_browsergym_calls_run_on_one_thread_whatever_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP API, WebSocket sessions and /web call the env from different threads."""
    threads: list[int] = []
    monkeypatch.setattr(
        browsergym_environment.importlib, "import_module", lambda _name: object()
    )
    monkeypatch.setattr(
        browsergym_environment.gym,
        "make",
        lambda *_args, **_kwargs: _ThreadRecordingGymEnv(threads),
    )

    envs = [BrowserGymEnvironment(task_name="click-test") for _ in range(2)]

    def episode(env: BrowserGymEnvironment) -> None:
        env.reset()
        env.step(BrowserGymAction(action_str="noop()"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(episode, envs * 2))
    for env in envs:
        env.close()

    assert len(threads) == 10
    assert len(set(threads)) == 1
