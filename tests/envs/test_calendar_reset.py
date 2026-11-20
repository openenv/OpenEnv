# SPDX-License-Identifier: BSD-3-Clause

"""reset(seed=..., episode_id=...) in the calendar environment.

Needs the calendar server's own dependencies (SQLAlchemy), so it skips without them.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

SERVER = Path(__file__).resolve().parents[2] / "envs" / "calendar_env" / "server"


@pytest.fixture
def env(monkeypatch):
    # The server modules import each other as top-level packages.
    monkeypatch.syspath_prepend(str(SERVER))
    from openenv_wrapper.mcp_env_environment import MCPEnvironment

    yield MCPEnvironment()
    for name in [n for n in sys.modules if n.split(".")[0] in _SERVER_PACKAGES]:
        del sys.modules[name]


_SERVER_PACKAGES = {
    p.name for p in SERVER.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
}


def test_reset_keeps_episode_id_or_generates_one(env):
    obs = env.reset(seed=3, episode_id="cal-1")
    assert env.state.episode_id == "cal-1"
    assert obs.metadata["episode_id"] == "cal-1"
    assert env.state.step_count == 0

    env.reset()
    first = env.state.episode_id
    env.reset()
    assert first and env.state.episode_id != first
