# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Atari environment's web playground drawing."""

import base64
import io

import pytest

pytest.importorskip("ale_py", reason="ale-py is not installed")

from atari_env.models import AtariAction
from atari_env.server.atari_environment import AtariEnvironment
from PIL import Image


def test_atari_web_playground():
    env = AtariEnvironment(game_name="pong")
    env.reset()
    obs = env.step(AtariAction(action_id=1)).model_dump()
    assert env.web_actions(obs) == [
        ("0 · NOOP", {"action_id": 0}),
        ("1 · FIRE", {"action_id": 1}),
        ("2 · RIGHT", {"action_id": 2}),
        ("3 · LEFT", {"action_id": 3}),
        ("4 · RIGHTFIRE", {"action_id": 4}),
        ("5 · LEFTFIRE", {"action_id": 5}),
    ]
    frame = env.render_web(obs)
    assert 'alt="Atari frame"' in frame
    png = base64.b64decode(frame.split("base64,")[1].split('"')[0])
    assert Image.open(io.BytesIO(png)).size == (160, 210)

    assert env.render_web({}) is None

    ram_env = AtariEnvironment(game_name="pong", obs_type="ram")
    assert ram_env.render_web(ram_env.reset().model_dump()) is None
