# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chess Environment for OpenEnv using moonfish.

This module provides a chess environment for reinforcement learning,
using python-chess for game logic and moonfish for position evaluation
and opponent play.
"""

import random
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chess
from moonfish.lib import search_move
from moonfish.psqt import board_evaluation, BOARD_EVALUATION_CACHE
from openenv.core.env_server import Environment

from ..models import ChessAction, ChessObservation, ChessState
from .rubrics import ChessWinLossRubric

# moonfish keeps one evaluation cache per process, so sessions take turns with it.
_MOONFISH_LOCK = threading.Lock()


class ChessEnvironment(Environment):
    """
    Chess environment implementing the OpenEnv interface.

    Uses python-chess for game logic and moonfish for position evaluation.
    Designed for RL training where an agent plays as one color against
    an opponent (which can be random, moonfish engine, or self-play).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        opponent: str = "moonfish",
        opponent_depth: int = 2,
        max_moves: int = 500,
        agent_color: str | None = None,
        gamma: float = 0.99,
    ):
        """
        Initialize the chess environment.

        Args:
            opponent: Opponent type - "moonfish", "random", or None (self-play)
            opponent_depth: Search depth when using moonfish as opponent
            max_moves: Maximum half-moves before draw
            agent_color: Which color the agent plays - "white", "black", or None (alternate each episode)
            gamma: Discount factor for temporal credit assignment (0-1)
        """
        super().__init__(rubric=ChessWinLossRubric(gamma=gamma))
        self._opponent = opponent
        self._opponent_depth = opponent_depth
        self._max_moves = max_moves
        self._agent_color_setting = agent_color
        self._gamma = gamma
        self._board = None
        self._state = None
        self._agent_color = chess.WHITE
        self._agent_move_count = 0  # Track agent moves for discounting
        self.reset()

    def reset(self, fen: str = None):
        """
        Initialize a new chess game episode.

        Args:
            fen: Optional starting position in FEN notation.

        Returns:
            Initial observation of the board state.
        """
        self._reset_rubric()
        # moonfish's module-level evaluation cache is never trimmed.
        with _MOONFISH_LOCK:
            BOARD_EVALUATION_CACHE.clear()

        if fen:
            self._board = chess.Board(fen)
        else:
            self._board = chess.Board()

        # Generate episode ID once
        episode_id = str(uuid.uuid4())

        # Determine agent color
        if self._agent_color_setting == "white":
            self._agent_color = chess.WHITE
        elif self._agent_color_setting == "black":
            self._agent_color = chess.BLACK
        elif self._agent_color_setting is None:
            # Alternate each episode based on episode_id hash
            self._agent_color = (
                chess.WHITE if hash(episode_id) % 2 == 0 else chess.BLACK
            )
        else:
            self._agent_color = chess.WHITE

        self._state = ChessState(
            episode_id=episode_id,
            step_count=0,
            current_player="white" if self._board.turn else "black",
            fen=self._board.fen(),
            move_history=[],
        )
        self._agent_move_count = 0

        # If agent plays Black and opponent is configured, opponent moves first
        if self._opponent is not None and self._agent_color == chess.BLACK:
            self._make_opponent_move()

        # Check if starting position is already terminal (e.g., custom FEN with checkmate/stalemate)
        _, done = self._calculate_reward_and_done()
        return self._make_observation(done=done)

    def step(self, action: ChessAction):
        """
        Execute a chess move and return the resulting state.

        Args:
            action: The move to make in UCI format (e.g., "e2e4").

        Returns:
            Observation with reward and done flag.
        """
        if self._board is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Parse the move
        try:
            move = chess.Move.from_uci(action.move)
        except ValueError:
            return self._make_observation(reward=-0.1, done=False)

        # Check if move is legal
        if move not in self._board.legal_moves:
            return self._make_observation(reward=-0.1, done=False)

        # Execute the move
        self._board.push(move)
        self._state.step_count += 1
        self._agent_move_count += 1
        self._state.move_history.append(action.move)
        self._state.current_player = "white" if self._board.turn else "black"
        self._state.fen = self._board.fen()

        # Calculate reward and check for game end
        reward, done = self._calculate_reward_and_done()

        # If game not over and opponent is configured, make opponent move
        if not done and self._opponent is not None:
            self._make_opponent_move()
            reward, done = self._calculate_reward_and_done()

        obs = self._make_observation(reward=reward, done=done)

        # Feed to rubric for trajectory accumulation
        self._apply_rubric(action, obs)

        # Note: obs.reward is kept from _calculate_reward_and_done() rather than
        # overridden by the rubric. The rubric here is a TrajectoryRubric that
        # returns 0.0 until done then the final score — the inline reward logic
        # already provides the correct per-step reward including penalties for
        # invalid/illegal moves. The rubric's value is in compute_step_rewards()
        # for training infrastructure, not in replacing obs.reward.

        return obs

    def _make_observation(self, reward: float = 0.0, done: bool = False):
        """Build observation from current board state."""
        legal_moves = [move.uci() for move in self._board.legal_moves]

        result = None
        if done:
            result = self._get_result_string()

        with _MOONFISH_LOCK:
            evaluation = board_evaluation(self._board)
        metadata = {
            "evaluation": evaluation,
            "fullmove_number": self._board.fullmove_number,
            "halfmove_clock": self._board.halfmove_clock,
        }

        # Compute discounted rewards for all agent moves when episode ends
        if done and self._agent_move_count > 0:
            discounted_rewards = self._compute_discounted_rewards(reward)
            metadata["discounted_rewards"] = discounted_rewards
            metadata["gamma"] = self._gamma

        return ChessObservation(
            fen=self._board.fen(),
            legal_moves=legal_moves,
            is_check=self._board.is_check(),
            done=done,
            reward=reward,
            result=result,
            metadata=metadata,
        )

    def _calculate_reward_and_done(self):
        """Calculate reward and check if episode is done."""
        if self._board.is_checkmate():
            winner = not self._board.turn
            if winner == self._agent_color:
                return 1.0, True
            else:
                return -1.0, True

        if self._board.is_stalemate():
            return 0.0, True

        if self._board.is_insufficient_material():
            return 0.0, True

        if self._board.is_fifty_moves():
            return 0.0, True

        if self._board.is_repetition(3):
            return 0.0, True

        if self._state.step_count >= self._max_moves:
            return 0.0, True

        return 0.0, False

    def _get_result_string(self):
        """Get the game result as a string."""
        if self._board.is_checkmate():
            return "1-0" if not self._board.turn else "0-1"
        return "1/2-1/2"

    def _compute_discounted_rewards(self, terminal_reward: float) -> list[float]:
        """
        Compute temporally discounted rewards for all agent moves.

        Uses exponential discounting: r_t = γ^(T-1-t) * R_final
        where T is total agent moves, t is move index, R_final is terminal reward.

        Earlier moves get less credit, later moves get more credit.
        This helps with credit assignment in long games.

        Args:
            terminal_reward: The final reward (+1 win, -1 loss, 0 draw)

        Returns:
            List of discounted rewards, one per agent move
        """
        T = self._agent_move_count
        discounted_rewards = []
        for t in range(T):
            # γ^(T-1-t) means last move (t=T-1) gets γ^0 = 1.0
            # First move (t=0) gets γ^(T-1)
            discount = self._gamma ** (T - 1 - t)
            discounted_rewards.append(discount * terminal_reward)
        return discounted_rewards

    def _make_opponent_move(self):
        """Make a move for the opponent using configured strategy."""
        if not self._board.legal_moves:
            return

        if self._opponent == "moonfish":
            with _MOONFISH_LOCK:
                move = search_move(self._board, depth=self._opponent_depth)
        elif self._opponent == "random":
            move = random.choice(list(self._board.legal_moves))
        else:
            return

        self._board.push(move)
        self._state.step_count += 1
        self._state.move_history.append(move.uci())
        self._state.current_player = "white" if self._board.turn else "black"
        self._state.fen = self._board.fen()

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The legal moves (UCI) as buttons, sorted so moves of the same piece sit together."""
        return [
            (move, {"move": move})
            for move in sorted(observation.get("legal_moves") or [])
        ]

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the board from the FEN, white at the bottom, the king in check highlighted."""
        if "fen" not in observation:
            return None
        board = chess.Board(observation["fen"])
        checked = board.king(board.turn) if observation["is_check"] else None
        cells = []
        for rank in range(7, -1, -1):
            cells.append(f'<span style="opacity:.6;font-size:12px">{rank + 1}</span>')
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                # Fixed board colours, so a1 stays dark in both themes.
                if square == checked:
                    bg = "var(--color-accent)"
                elif (file + rank) % 2:
                    bg = "hsl(37 50% 84%)"
                else:
                    bg = "hsl(28 33% 55%)"
                # U+FE0E keeps the pawn as a text glyph instead of an emoji.
                glyph = piece.unicode_symbol() + "\ufe0e" if piece else ""
                cells.append(
                    f'<span style="background:{bg};color:black">{glyph}</span>'
                )
        cells.append("<span></span>")
        cells += [
            f'<span style="opacity:.6;font-size:12px;line-height:20px">{f}</span>'
            for f in "abcdefgh"
        ]
        return (
            '<div role="img" aria-label="Chess board" style="display:inline-grid;'
            "grid-template-columns:16px repeat(8,40px);grid-template-rows:repeat(8,40px) 20px;"
            "line-height:40px;text-align:center;font-size:30px;color:var(--body-text-color);padding:10px;"
            'border:1px solid var(--border-color-primary);border-radius:10px;background:var(--background-fill-secondary)">'
            + "".join(cells)
            + "</div>"
        )

    @property
    def state(self):
        """Return the current episode state."""
        return self._state
