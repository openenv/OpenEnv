# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the dm_control environment's web playground drawing."""

import base64

import pytest

pytest.importorskip("dm_control", reason="dm_control is not installed")

from dm_control_env.models import DMControlAction
from dm_control_env.server.dm_control_environment import DMControlEnvironment


def test_dm_control_web_playground():
    env = DMControlEnvironment()
    env.reset()
    obs = env.step(DMControlAction(values=[0.5])).model_dump()
    try:
        env._env.physics.render(height=24, width=32)
    except Exception:
        pytest.skip("MuJoCo can't render here (no OpenGL backend)")
    time = env._env.physics.data.time
    frame = env.render_web(obs)
    assert env._env.physics.data.time == time
    assert 'alt="dm_control frame"' in frame
    jpeg = base64.b64decode(frame.split("base64,")[1].split('"')[0])
    assert jpeg.startswith(b"\xff\xd8")
    assert env.render_web({}) is None
