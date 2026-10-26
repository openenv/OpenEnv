# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the BrowserGym drawing in the web playground."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from browsergym_env.models import BrowserGymAction
from browsergym_env.server import browsergym_environment
from browsergym_env.server.browsergym_environment import BrowserGymEnvironment


class _FakeGymEnv:
    def _obs(self, error=""):
        return {
            "goal": "Click the button.",
            "url": "http://example.test/click-test.html",
            "screenshot": np.full((40, 60, 3), 200, dtype=np.uint8),
            "last_action_error": error,
        }

    def reset(self, **_kwargs):
        return self._obs(), {}

    def step(self, _action):
        return self._obs('Could not find element with bid "99"'), 0.0, False, False, {}

    def close(self):
        pass


def test_browsergym_web_playground(monkeypatch):
    monkeypatch.setattr(
        browsergym_environment.importlib, "import_module", lambda _name: object()
    )
    monkeypatch.setattr(
        browsergym_environment.gym, "make", lambda *_args, **_kwargs: _FakeGymEnv()
    )
    env = BrowserGymEnvironment(task_name="click-test", include_screenshot=True)
    assert env.render_web({}) is None

    page = env.render_web(env.reset().model_dump())
    assert "Click the button." in page
    assert '<img alt="BrowserGym page" src="data:image/jpeg;base64,' in page
    assert "Last action error" not in page
    assert "#" not in page  # theme colours only, so it reads in dark mode

    page = env.render_web(
        env.step(BrowserGymAction(action_str="click('99')")).model_dump()
    )
    assert (
        "Last action error:</b> Could not find element with bid &quot;99&quot;" in page
    )
