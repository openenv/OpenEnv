# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Wildfire environment and its grid drawing in the web playground."""

from wildfire_env.models import WildfireAction
from wildfire_env.server.wildfire_environment import WildfireEnvironment


def test_wildfire_web_playground():
    env = WildfireEnvironment(width=8, height=8)
    obs = env.reset().model_dump()
    fire = obs["grid"].index(2)
    action = WildfireAction(action="water", x=fire % 8, y=fire // 8)
    obs = env.step(action).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Wildfire board"' in board
    assert board.count('title="x=') == 64
    assert f"x={fire % 8}, y={fire // 8}: water" in board
    assert f"water {obs['remaining_water']}</span>" in board
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None


def _make_env():
    return WildfireEnvironment(width=8, height=8)


class TestResetSeed:
    """reset(seed=...) must make the fire reproducible."""

    @staticmethod
    def _observations_after_seed(env, seed, steps=3):
        """One seeded reset, `steps` waits, then one unseeded reset."""
        observations = [env.reset(seed=seed)]
        observations += [env.step(WildfireAction(action="wait")) for _ in range(steps)]
        observations.append(env.reset())
        return [obs.model_dump() for obs in observations]

    def test_same_seed_same_sequence_across_instances(self):
        # Every instance starts from the constructor's seed: move one ahead.
        used = _make_env()
        used.reset()
        first = self._observations_after_seed(used, seed=1234)
        second = self._observations_after_seed(_make_env(), seed=1234)
        assert first == second

    def test_reseeding_restarts_the_sequence(self):
        env = _make_env()
        first = self._observations_after_seed(env, seed=1234)
        second = self._observations_after_seed(env, seed=1234)
        assert first == second

    def test_different_seeds_give_different_sequences(self):
        env = _make_env()
        first = self._observations_after_seed(env, seed=1)
        second = self._observations_after_seed(env, seed=2)
        assert first != second

    def test_episode_id_is_kept(self):
        env = _make_env()
        env.reset(episode_id="episode-1")
        assert env.state.episode_id == "episode-1"
        env.reset(episode_id="")
        assert env.state.episode_id == ""
        env.reset()
        assert env.state.episode_id not in (None, "", "episode-1")
