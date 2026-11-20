# SPDX-License-Identifier: BSD-3-Clause

"""The actions OpenApp sends to BrowserGym for a client's action."""

import ast

import pytest
from openapp_env.models import OpenAppAction
from openapp_env.server.openapp_environment import OpenAppEnvironment


class _RecordingBrowser:
    """Stands in for BrowserGym's browser, which a unit test cannot start."""

    def __init__(self) -> None:
        self.actions: list[str] = []

    def step(self, action: str):
        self.actions.append(action)
        return {"url": "http://localhost:5001"}, 0.0, False, False, {}


@pytest.fixture
def env_and_browser():
    env = OpenAppEnvironment(openapps_url="http://localhost:5001")
    browser = _RecordingBrowser()
    env._browser_env = browser
    return env, browser


@pytest.mark.parametrize(
    "url",
    [
        "file:///home/user/.cache/huggingface/token",
        "FILE:///x",
        "view-source:file:///x",
    ],
)
def test_goto_never_opens_a_server_file(env_and_browser, url):
    env, browser = env_and_browser
    obs = env.step(OpenAppAction(action_type="goto", url=url))
    assert browser.actions == []
    assert "only http(s)" in obs.last_action_error


def test_goto_opens_web_pages(env_and_browser):
    env, browser = env_and_browser
    env.step(OpenAppAction(action_type="goto", url="http://localhost:5001/todo"))
    assert browser.actions == ["goto('http://localhost:5001/todo')"]


@pytest.mark.parametrize(
    "action",
    [
        OpenAppAction(action_type="click", bid='1")\ngoto("file:///x'),
        OpenAppAction(action_type="fill", bid="1", text='a")\ngoto("file:///x'),
        OpenAppAction(action_type="send_keys", text='a")\ngoto("file:///x'),
    ],
)
def test_arguments_cannot_add_actions(env_and_browser, action):
    """A quote in an argument must not end it: BrowserGym runs every call in the string."""
    env, browser = env_and_browser
    env.step(action)
    (sent,) = browser.actions
    (call,) = ast.parse(sent).body
    assert call.value.func.id == action.action_type


class _ResettableBrowser:
    """Records the seed BrowserGym's reset receives."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def reset(self, seed: int | None = None):
        self.seeds.append(seed)
        return {"url": "http://localhost:5001"}, {}


def test_reset_passes_seed_and_keeps_episode_id(monkeypatch):
    monkeypatch.setenv("OPENAPPS_URL", "http://localhost:5001")
    env = OpenAppEnvironment(openapps_url="http://localhost:5001")
    browser = _ResettableBrowser()
    env._browser_env = browser

    env.reset(seed=7, episode_id="ep-1")
    assert browser.seeds == [7]
    assert env.state.episode_id == "ep-1"

    env.reset()
    assert browser.seeds == [7, None]
    assert env.state.episode_id not in ("", "ep-1")
