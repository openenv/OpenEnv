# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FinRL environment wrapper."""

import asyncio

import numpy as np
from finrl_env.server.finrl_environment import FinRLEnvironment


class FakeStockTradingEnv:
    """Stands in for FinRL's StockTradingEnv, which is not installed in CI.

    Like the real one, it is deterministic: the state is [cash, price, holdings].
    """

    def __init__(self, **config):
        self.config = config

    def reset(self, *, seed=None, options=None):
        return np.array([1000.0, 10.0, 0.0]), {}

    def _get_date(self):
        return "2020-01-01"


def _make_env():
    return FinRLEnvironment(
        finrl_env_class=FakeStockTradingEnv,
        finrl_env_config={"stock_dim": 1, "action_space": 1},
    )


class TestResetSeed:
    """reset(seed=..., episode_id=...) must be accepted.

    The wrapped trading environment is deterministic, so a seed changes nothing.
    The server only passes arguments that `reset` declares, so an environment
    that omits them silently drops the caller's seed and episode id.
    """

    def test_seeded_reset_gives_same_initial_observation(self):
        env = _make_env()
        unseeded = env.reset()
        for seed in (0, 1, 1234):
            obs = env.reset(seed=seed)
            assert obs.state == unseeded.state
            assert obs.portfolio_value == unseeded.portfolio_value == 1000.0
            assert obs.date == "2020-01-01"
            assert env.state.step_count == 0

    def test_episode_id_is_kept_or_generated(self):
        env = _make_env()
        env.reset(episode_id="run-1")
        assert env.state.episode_id == "run-1"
        env.reset(seed=3, episode_id="run-2")
        assert env.state.episode_id == "run-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first

    def test_reset_async_forwards_seed_and_episode_id(self):
        env = _make_env()
        asyncio.run(env.reset_async(seed=3, episode_id="async-1"))
        assert env.state.episode_id == "async-1"
