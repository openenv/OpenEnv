# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the kernrl environment."""

from kernrl.server.kernrl_environment import KernelOptEnvironment


class TestResetSeed:
    """reset(seed=...) must make the problem order reproducible.

    Uses the problems shipped with the env and no GPU, so it runs in CI.
    """

    @staticmethod
    def _problem_ids_after_seed(env, seed, n=5):
        """One seeded reset followed by n - 1 unseeded ones."""
        problem_ids = [env.reset(seed=seed).problem_id]
        problem_ids += [env.reset().problem_id for _ in range(n - 1)]
        return problem_ids

    def test_same_seed_same_sequence_across_instances(self):
        first = self._problem_ids_after_seed(KernelOptEnvironment(), seed=1234)
        second = self._problem_ids_after_seed(KernelOptEnvironment(), seed=1234)
        assert first == second

    def test_reseeding_restarts_the_sequence(self):
        env = KernelOptEnvironment()
        first = self._problem_ids_after_seed(env, seed=1234)
        second = self._problem_ids_after_seed(env, seed=1234)
        assert first == second

    def test_different_seeds_give_different_sequences(self):
        env = KernelOptEnvironment()
        first = self._problem_ids_after_seed(env, seed=1)
        second = self._problem_ids_after_seed(env, seed=2)
        assert first != second

    def test_episode_id_is_kept(self):
        env = KernelOptEnvironment()
        env.reset(episode_id="episode-1")
        assert env.state.episode_id == "episode-1"
        env.reset(episode_id="")
        assert env.state.episode_id == ""
        env.reset()
        assert env.state.episode_id not in (None, "", "episode-1")
