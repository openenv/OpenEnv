# SPDX-License-Identifier: BSD-3-Clause

import pytest

pytest.importorskip("helium")

from helium_browser_env.server.helium_browser_environment import BrowserEnvironment


def test_helium_browser_web_playground(monkeypatch):
    monkeypatch.setenv("BROWSER_SANDBOX", "hf")
    env = BrowserEnvironment()
    board = env.render_web(
        {
            "screenshot": "iVBORw0KGgo=",
            "url": "https://example.com/",
            "error": "TimeoutException",
        }
    )
    assert (
        '<img alt="Helium Browser page" src="data:image/png;base64,iVBORw0KGgo="'
        in board
    )
    assert "https://example.com/" in board
    assert "TimeoutException" in board
    assert "#" not in board
    assert env.render_web({}) is None
