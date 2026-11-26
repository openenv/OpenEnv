# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OpenSpielEnvironment.reset, run in process.

`test_openspiel_environment.py` needs gunicorn and a server process, so it is
skipped where gunicorn is missing. These tests need only OpenSpiel.
"""

import asyncio

import pytest

# Skip entire module if OpenSpiel is not installed
pytest.importorskip("open_spiel", reason="open_spiel is not installed")

from openspiel_env.models import OpenSpielAction
from openspiel_env.server.openspiel_environment import OpenSpielEnvironment


def _play(env, seed=None, steps=8):
    """Reset (seeded if given), always play the first legal action, return what was seen."""
    obs = env.reset() if seed is None else env.reset(seed=seed)
    seen = [(obs.info_state, obs.legal_actions, obs.opponent_last_action)]
    while not obs.done and len(seen) <= steps:
        obs = env.step(OpenSpielAction(action_id=obs.legal_actions[0]))
        seen.append((obs.info_state, obs.legal_actions, obs.opponent_last_action))
    return seen


def _sequence_after_seed(env, seed, n=5):
    """One seeded game followed by n - 1 unseeded ones."""
    return [_play(env, seed)] + [_play(env) for _ in range(n - 1)]


# Catch has a random ball position. Tic-tac-toe has no chance events, so with
# agent_player=1 the random opponent's first move is all reset can show of it.
GAMES = [
    pytest.param({"game_name": "catch"}, id="catch"),
    pytest.param(
        {"game_name": "tic_tac_toe", "agent_player": 1, "opponent_policy": "random"},
        id="tic_tac_toe-random-opponent",
    ),
    pytest.param(
        {"game_name": "kuhn_poker", "opponent_policy": "random"},
        id="kuhn_poker-random-opponent",
    ),
]


class TestResetSeed:
    """reset(seed=...) makes the game and the random opponent reproducible."""

    @pytest.mark.parametrize("kwargs", GAMES)
    def test_same_seed_same_observations_across_instances(self, kwargs):
        first = _sequence_after_seed(OpenSpielEnvironment(**kwargs), seed=1234)
        second = _sequence_after_seed(OpenSpielEnvironment(**kwargs), seed=1234)
        assert first == second

    @pytest.mark.parametrize("kwargs", GAMES)
    def test_reseeding_restarts_the_sequence(self, kwargs):
        env = OpenSpielEnvironment(**kwargs)
        first = _sequence_after_seed(env, seed=1234)
        second = _sequence_after_seed(env, seed=1234)
        assert first == second

    @pytest.mark.parametrize("kwargs", GAMES)
    def test_different_seeds_give_different_games(self, kwargs):
        env = OpenSpielEnvironment(**kwargs)
        games = {str(_play(env, seed)) for seed in range(10)}
        assert len(games) > 1

    @pytest.mark.parametrize("kwargs", GAMES)
    def test_unseeded_reset_still_works_and_varies(self, kwargs):
        env = OpenSpielEnvironment(**kwargs)
        games = {str(_play(env)) for _ in range(20)}
        assert len(games) > 1

    def test_seed_does_not_touch_fixed_opponents(self):
        env = OpenSpielEnvironment(
            "tic_tac_toe", agent_player=1, opponent_policy="first"
        )
        assert _play(env, seed=1) == _play(env, seed=2)

    def test_seed_is_ignored_in_single_player_games_without_chance(self):
        env = OpenSpielEnvironment("cliff_walking")
        assert _play(env, seed=1) == _play(env, seed=2)

    def test_episode_id_is_kept_or_generated(self):
        env = OpenSpielEnvironment("tic_tac_toe")
        env.reset(episode_id="game-1")
        assert env.state.episode_id == "game-1"
        env.reset(seed=3, episode_id="game-2")
        assert env.state.episode_id == "game-2"
        env.reset()
        first = env.state.episode_id
        env.reset()
        assert first and env.state.episode_id != first

    def test_reset_async_forwards_seed_and_episode_id(self):
        env = OpenSpielEnvironment("tic_tac_toe", agent_player=1)
        expected = _play(env, seed=3)
        obs = asyncio.run(env.reset_async(seed=3, episode_id="async-1"))
        assert env.state.episode_id == "async-1"
        assert (obs.info_state, obs.legal_actions, obs.opponent_last_action) == (
            expected[0]
        )


class TestSimultaneousGames:
    """Simultaneous-move games take the agent's action plus the opponent's."""

    def test_reset_and_step_matrix_rps(self):
        env = OpenSpielEnvironment("matrix_rps")
        obs = env.reset()
        assert obs.legal_actions == [0, 1, 2] and not obs.done
        obs = env.step(OpenSpielAction(action_id=0))
        assert obs.done and obs.game_phase == "terminal"
        assert env.state.step_count == 1

    def test_step_reports_the_opponent_action(self):
        env = OpenSpielEnvironment("matrix_rps", opponent_policy="last")
        env.reset()
        obs = env.step(OpenSpielAction(action_id=0))
        assert obs.opponent_last_action == 2
