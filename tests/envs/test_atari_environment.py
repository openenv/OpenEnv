# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Atari environment's web playground drawing."""

import asyncio
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


def _play(env, seed=None, steps=60):
    """Reset (seeded if given), play a fixed action cycle, return what was seen."""
    obs = env.reset() if seed is None else env.reset(seed=seed)
    seen = [list(obs.screen)]
    for i in range(steps):
        obs = env.step(AtariAction(action_id=i % 3))
        seen.append((list(obs.screen), obs.reward))
    return seen


def _sticky_env():
    # Sticky actions are the only randomness in ALE, so the seed needs them
    return AtariEnvironment(
        game_name="pong", obs_type="ram", repeat_action_probability=0.5
    )


class TestResetSeed:
    """reset(seed=...) seeds ALE's RNG, which drives sticky actions."""

    def test_same_seed_same_observations_across_instances(self):
        assert _play(_sticky_env(), seed=7) == _play(_sticky_env(), seed=7)

    def test_reseeding_restarts_the_sequence(self):
        env = _sticky_env()
        first = _play(env, seed=7)
        _play(env)  # an unseeded game in between
        assert _play(env, seed=7) == first

    def test_different_seeds_diverge(self):
        env = _sticky_env()
        assert len({str(_play(env, seed=seed)) for seed in range(5)}) > 1

    def test_seed_reapplies_mode_and_difficulty(self):
        """Loading the ROM clears them, so a seeded reset must set them again."""

        class Spy:
            def __init__(self, ale):
                self._ale = ale
                self.calls = []

            def __getattr__(self, name):
                attr = getattr(self._ale, name)
                if name not in ("setMode", "setDifficulty"):
                    return attr

                def call(*args):
                    self.calls.append((name, args))
                    return attr(*args)

                return call

        env = AtariEnvironment(game_name="pong", mode=1, difficulty=1)
        env.ale = Spy(env.ale)
        env.reset()
        assert env.ale.calls == []
        env.reset(seed=3)
        assert env.ale.calls == [("setMode", (1,)), ("setDifficulty", (1,))]

    def test_unseeded_reset_still_works(self):
        env = _sticky_env()
        assert env.reset().done is False
        assert env.step(AtariAction(action_id=0)).done is False

    def test_episode_id_is_kept_or_generated(self):
        env = AtariEnvironment(game_name="pong")
        env.reset(episode_id="game-1")
        assert env.state.episode_id == "game-1"
        env.reset(seed=3, episode_id="game-2")
        assert env.state.episode_id == "game-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first
        assert env.state.step_count == 0

    def test_reset_async_forwards_seed_and_episode_id(self):
        env = _sticky_env()
        expected = _play(env, seed=3)[0]
        obs = asyncio.run(env.reset_async(seed=3, episode_id="async-1"))
        assert env.state.episode_id == "async-1"
        assert list(obs.screen) == expected
