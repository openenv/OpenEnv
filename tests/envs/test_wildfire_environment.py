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


class TestStepAfterDone:
    """Once the episode ends, step() returns the final observation until reset()."""

    @staticmethod
    def _play_to_end(env):
        obs = env.reset(seed=0)
        while not obs.done:
            obs = env.step(WildfireAction(action="wait"))
        return obs

    def test_later_steps_repeat_the_final_observation(self):
        env = WildfireEnvironment(width=8, height=8, max_steps=5)
        final = self._play_to_end(env)
        steps = env.state.step_count
        fire = final.grid.index(2) if 2 in final.grid else 0

        for action in (
            WildfireAction(action="wait"),
            WildfireAction(action="water", x=fire % 8, y=fire // 8),
            WildfireAction(action="break", x=0, y=0),
        ):
            again = env.step(action)
            assert again.done is True
            assert again.reward == final.reward
            assert again.grid == final.grid
        assert env.state.step_count == steps
        assert env.state.remaining_water == final.remaining_water

    def test_reset_starts_a_new_episode(self):
        env = WildfireEnvironment(width=8, height=8, max_steps=5)
        self._play_to_end(env)
        env.reset(seed=1)
        obs = env.step(WildfireAction(action="wait"))
        assert env.state.step_count == 1
        assert obs.step == 1
