# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Chess environment."""

import pytest

# Skip entire module if chess dependencies are not installed
pytest.importorskip("chess", reason="python-chess is not installed")
pytest.importorskip("moonfish", reason="moonfish is not installed")

from envs.chess_env import ChessAction, ChessObservation, ChessState
from envs.chess_env.server.chess_environment import ChessEnvironment
from moonfish.psqt import BOARD_EVALUATION_CACHE


class TestChessModels:
    """Test Chess data models."""

    def test_chess_action_creation(self):
        """Test ChessAction can be created with a move."""
        action = ChessAction(move="e2e4")
        assert action.move == "e2e4"

    def test_chess_observation_defaults(self):
        """Test ChessObservation has correct defaults."""
        obs = ChessObservation()
        assert obs.fen == ""
        assert obs.legal_moves == []
        assert obs.is_check is False
        assert obs.done is False
        assert obs.result is None

    def test_chess_state_defaults(self):
        """Test ChessState has correct defaults."""
        state = ChessState(episode_id="test-123", step_count=0)
        assert state.episode_id == "test-123"
        assert state.step_count == 0
        assert state.current_player == "white"
        assert state.move_history == []


class TestChessEnvironment:
    """Test Chess environment logic."""

    @pytest.fixture
    def env(self):
        """Create a fresh ChessEnvironment for each test."""
        return ChessEnvironment(opponent=None)  # No opponent for testing

    def test_reset_returns_observation(self, env):
        """Test reset returns a valid observation."""
        obs = env.reset()
        assert isinstance(obs, ChessObservation)
        assert obs.fen != ""
        assert len(obs.legal_moves) == 20  # 20 legal moves at start
        assert obs.is_check is False
        assert obs.done is False

    def test_reset_with_custom_fen(self, env):
        """Test reset with custom starting position."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        obs = env.reset(fen=fen)
        assert obs.fen == fen

    def test_step_valid_move(self, env):
        """Test stepping with a valid move."""
        env.reset()
        obs = env.step(ChessAction(move="e2e4"))
        assert isinstance(obs, ChessObservation)
        # After e2e4, the pawn is on e4 (shown as 4P3 in FEN's 4th rank)
        assert "4P3" in obs.fen

    def test_step_invalid_move_format(self, env):
        """Test stepping with invalid move format returns penalty."""
        env.reset()
        obs = env.step(ChessAction(move="invalid"))
        assert obs.reward == -0.1
        assert obs.done is False

    def test_step_illegal_move(self, env):
        """Test stepping with illegal move returns penalty."""
        env.reset()
        obs = env.step(ChessAction(move="e2e5"))  # Can't move pawn 3 squares
        assert obs.reward == -0.1
        assert obs.done is False

    def test_state_property(self, env):
        """Test state property returns ChessState."""
        env.reset()
        state = env.state
        assert isinstance(state, ChessState)
        assert state.episode_id != ""
        assert state.step_count == 0
        assert state.current_player == "white"

    def test_state_updates_after_move(self, env):
        """Test state updates correctly after a move."""
        env.reset()
        env.step(ChessAction(move="e2e4"))
        state = env.state
        assert state.step_count == 1
        assert "e2e4" in state.move_history
        assert state.current_player == "black"

    def test_checkmate_ends_game(self, env):
        """Test checkmate ends the game with correct reward."""
        # Fool's mate position
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        env.reset(fen=fen)
        # White is checkmated
        assert env._board.is_checkmate()

    def test_stalemate_is_draw(self, env):
        """Test stalemate ends with draw reward."""
        # Stalemate position - black king on h8, white king f7, white queen g6
        fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"
        obs = env.reset(fen=fen)
        assert env._board.is_stalemate()
        assert obs.done
        assert obs.reward == 0.0
        assert obs.legal_moves == []


class TestChessEnvironmentWithOpponent:
    """Test Chess environment with opponent configured."""

    def test_random_opponent_makes_moves(self):
        """Test random opponent makes a move after agent move."""
        env = ChessEnvironment(opponent="random", agent_color="white")
        env.reset()

        # Agent makes a move
        env.step(ChessAction(move="e2e4"))

        # After agent's move and opponent's response, should be white's turn again
        assert env.state.current_player == "white"
        assert env.state.step_count == 2  # Agent + opponent

    def test_moonfish_opponent_makes_moves(self):
        """Test moonfish opponent makes a move after agent move."""
        env = ChessEnvironment(
            opponent="moonfish", opponent_depth=1, agent_color="white"
        )
        env.reset()

        # Agent makes a move
        env.step(ChessAction(move="e2e4"))

        # After agent's move and opponent's response, should be white's turn again
        assert env.state.current_player == "white"
        assert env.state.step_count == 2

    def test_reset_clears_moonfish_evaluation_cache(self):
        """Test reset empties moonfish's position cache so it can't grow across episodes."""
        env = ChessEnvironment(
            opponent="moonfish", opponent_depth=1, agent_color="white"
        )
        env.reset()
        env.step(ChessAction(move="e2e4"))
        assert len(BOARD_EVALUATION_CACHE) > 1

        env.reset()

        # Only the starting position, evaluated for the reset observation
        assert len(BOARD_EVALUATION_CACHE) == 1

    def test_opponent_checkmate_gives_negative_reward(self):
        """Test agent gets -1.0 reward when opponent checkmates."""
        env = ChessEnvironment(
            opponent="moonfish", opponent_depth=2, agent_color="white"
        )
        # Position after 1.f3 e5 - agent plays g4, opponent plays Qh4# (fool's mate)
        fen = "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2"
        env.reset(fen=fen)

        # Agent blunders with g4, allowing Qh4#
        obs = env.step(ChessAction(move="g2g4"))

        assert obs.done is True
        assert obs.reward == -1.0
        assert obs.result == "0-1"


class TestTemporalDiscounting:
    """Test temporal discounting for credit assignment."""

    def test_discounted_rewards_in_terminal_observation(self):
        """Test that terminal observation includes discounted rewards."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)
        # Back-rank mate: black king trapped by own pawns, white rook delivers mate
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)

        obs = env.step(ChessAction(move="e1e8"))

        assert obs.done is True
        assert obs.reward == 1.0
        assert "discounted_rewards" in obs.metadata
        assert "gamma" in obs.metadata
        assert obs.metadata["gamma"] == 0.99

    def test_discounted_rewards_length_matches_agent_moves(self):
        """Test discounted rewards list length equals number of agent moves."""
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=0.99)
        # Back-rank mate position
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)

        # One move to checkmate
        obs = env.step(ChessAction(move="e1e8"))

        assert obs.done is True
        assert len(obs.metadata["discounted_rewards"]) == 1

    def test_discounting_formula(self):
        """Test the discounting formula: r_t = γ^(T-1-t) × R_final."""
        gamma = 0.5  # Use 0.5 for easy mental math
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=gamma)

        # Back-rank mate position
        fen = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
        env.reset(fen=fen)

        # One agent move to checkmate
        obs = env.step(ChessAction(move="e1e8"))

        assert obs.done is True
        rewards = obs.metadata["discounted_rewards"]
        # T=1, t=0: γ^(1-1-0) = γ^0 = 1.0
        assert len(rewards) == 1
        assert rewards[0] == 1.0  # Last move gets full reward

    def test_earlier_moves_get_less_credit(self):
        """Test that earlier moves get less credit than later moves (self-play mode)."""
        gamma = 0.9
        env = ChessEnvironment(opponent=None, agent_color="white", gamma=gamma)
        env.reset()

        # Play fool's mate - white loses
        env.step(ChessAction(move="f2f3"))  # Move 0 (white)
        env.step(ChessAction(move="e7e5"))  # Move 1 (black)
        env.step(ChessAction(move="g2g4"))  # Move 2 (white)
        obs = env.step(ChessAction(move="d8h4"))  # Move 3 (black) - Qh4# checkmate

        assert obs.done is True
        assert obs.result == "0-1"  # Black wins
        rewards = obs.metadata["discounted_rewards"]

        # Agent is white, black won, so agent lost -> reward = -1.0
        assert obs.reward == -1.0

        # Check discounting: each earlier move gets γ less credit
        # Move 3 (last): γ^0 × (-1) = -1.0
        # Move 2: γ^1 × (-1) = -0.9
        # Move 1: γ^2 × (-1) = -0.81
        # Move 0: γ^3 × (-1) = -0.729
        assert len(rewards) == 4
        assert abs(rewards[3] - (-1.0)) < 0.001
        assert abs(rewards[2] - (-0.9)) < 0.001
        assert abs(rewards[1] - (-0.81)) < 0.001
        assert abs(rewards[0] - (-0.729)) < 0.001

    def test_gamma_parameter_configurable(self):
        """Test that gamma can be configured."""
        env1 = ChessEnvironment(opponent=None, gamma=0.99)
        env2 = ChessEnvironment(opponent=None, gamma=0.5)

        assert env1._gamma == 0.99
        assert env2._gamma == 0.5


def test_chess_web_playground():
    env = ChessEnvironment(opponent=None)
    obs = env.reset().model_dump()
    actions = env.web_actions(obs)
    assert len(actions) == 20
    assert ("e2e4", {"move": "e2e4"}) in actions
    board = env.render_web(obs)
    assert 'aria-label="Chess board"' in board
    assert "♔" in board and "♚" in board
    assert "var(--color-accent)" not in board
    assert "#" not in board  # theme colours only, so it reads in dark mode

    check = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert "var(--color-accent)" in env.render_web(env.reset(fen=check).model_dump())
    assert env.render_web({}) is None
