# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Connect4 environment."""

import asyncio

from connect4_env import Connect4Action
from connect4_env.server.connect4_environment import Connect4Environment


class TestResetSeed:
    """reset(seed=..., episode_id=...) must be accepted.

    Connect4 has no randomness, so a seed changes nothing: the board always
    starts empty. The server only passes arguments that `reset` declares, so an
    environment that omits them silently drops the caller's seed and episode id.
    """

    def test_seeded_reset_gives_same_initial_board(self):
        env = Connect4Environment()
        unseeded = env.reset()
        for seed in (0, 1, 1234):
            obs = env.reset(seed=seed)
            assert obs.board == unseeded.board
            assert env.state.next_player == 1
            assert env.state.step_count == 0

    def test_episode_id_is_kept_or_generated(self):
        env = Connect4Environment()
        env.reset(episode_id="game-1")
        assert env.state.episode_id == "game-1"
        env.reset(seed=3, episode_id="game-2")
        assert env.state.episode_id == "game-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first

    def test_seeded_reset_starts_a_playable_game(self):
        env = Connect4Environment()
        env.step(Connect4Action(column=3))
        env.reset(seed=5)
        obs = env.step(Connect4Action(column=3))
        assert obs.board[-1][3] == 1
        assert env.state.step_count == 1

    def test_reset_async_forwards_seed_and_episode_id(self):
        env = Connect4Environment()
        asyncio.run(env.reset_async(seed=3, episode_id="async-1"))
        assert env.state.episode_id == "async-1"
