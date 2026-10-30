# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minesweeper Environment Implementation.

A Minesweeper game environment where agents must reveal cells and place flags
to identify mines on a grid board without triggering any mines.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import (
    GameStatus,
    MinesweeperAction,
    MinesweeperObservation,
    MinesweeperState,
)


class MinesweeperEnvironment(
    Environment[MinesweeperAction, MinesweeperObservation, State]
):
    """
    Minesweeper game environment implementation for Reinforcement Learning.
    The environment consists of a grid with hidden mines. The agent can reveal cells or place flags.
    The goal is to reveal all non-mine cells without triggering a mine.
    The agent must:
    - Reveal cells to uncover numbers indicating adjacent mines.
    - Place flags on suspected mine locations.
    The game ends when all non-mine cells are revealed (win) or a mine is revealed (loss).

    Observation encoding:
        -1: unrevealed
        0-8: number of adjacent mines (if revealed)
        'F': flagged cell
        '*': mine (only revealed if game is lost)

    Example:
        >>> env = MinesweeperEnvironment(height=5, width=5, num_mines=5)
        >>> obs = env.reset()
        >>> action = MinesweeperAction(row=2, col=3, action_type='reveal')
    """

    def __init__(self, height: int = 5, width: int = 5, num_mines: int = 5):
        """Initialize the minesweeper_env environment.
        Args:
            height: Height of the minesweeper board.
            width: Width of the minesweeper board.
            num_mines: Number of mines to place on the board.
        """
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError("Board dimensions must be positive")
        if not 0 <= num_mines < height * width:
            raise ValueError("Mine count must leave at least one safe cell")
        self.height = height
        self.width = width
        self.num_mines = num_mines
        self._rng = random.Random()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # Internal game state
        self._mine_positions: Set[Tuple[int, int]] = set()
        self._revealed_cells: Set[Tuple[int, int]] = set()
        self._flags_placed: Set[Tuple[int, int]] = set()
        self._rewarded_flags: Set[Tuple[int, int]] = set()
        self._mine_counts: List[List[int]] = [
            [0 for _ in range(width)] for _ in range(height)
        ]
        self._game_status = GameStatus.ONGOING

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MinesweeperObservation:
        """
        Reset the environment and starts a new game.

        Returns:
            MinesweeperObservation with initial board state
        """
        if seed is not None:
            self._rng.seed(seed)
        self._state = State(
            episode_id=episode_id if episode_id is not None else str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        # Reset internal game state
        self._revealed_cells.clear()
        self._flags_placed.clear()
        self._rewarded_flags.clear()
        self._game_status = GameStatus.ONGOING

        # Place mines randomly
        self._place_mines()

        # Compute mine counts for each cell
        self._compute_mine_counts()

        return self._create_observation(
            done=False,
            reward=0.0,
        )

    def step(self, action: MinesweeperAction) -> MinesweeperObservation:  # type: ignore[override]
        """
        Execute a step in the environment by performing the given action.

        Args:
            action: MinesweeperAction specifying row, col and action_type

        Returns:
            MinesweeperObservation with updated board state and reward
        """
        self._state.step_count += 1

        row, col = action.row, action.col

        # Validate action
        if not self._is_valid_position(row, col):
            # Invalid action or game already over
            return self._create_observation(
                done=self._game_status != GameStatus.ONGOING,
                reward=-0.1,
                metadata={"error": "Invalid action"},
            )

        # If game already over, no further actions allowed
        if self._game_status != GameStatus.ONGOING:
            return self._create_observation(
                done=True,
                reward=0.0,
                metadata={"info": "Game already over"},
            )

        reward = 0.0
        error: Optional[str] = None

        if action.action_type == "reveal":
            reward, error = self._reveal_cell(row, col)
        elif action.action_type == "flag":
            reward, error = self._toggle_flag(row, col)
        else:
            reward = -0.1
            error = f"Unknown action_type: {action.action_type!r}"

        self._check_win_condition()

        metadata = {"error": error} if error else None
        return self._create_observation(
            done=self._game_status != GameStatus.ONGOING,
            reward=reward,
            metadata=metadata,
        )

    def _place_mines(self) -> None:
        """Randomly place mines on the board."""
        self._mine_positions.clear()
        while len(self._mine_positions) < self.num_mines:
            r = self._rng.randint(0, self.height - 1)
            c = self._rng.randint(0, self.width - 1)
            self._mine_positions.add((r, c))

    def _compute_mine_counts(self) -> None:
        """Compute the number of adjacent mines for each cell."""
        self._mine_counts = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) not in self._mine_positions:
                    count = self._count_adjacent_mines(row, col)
                    self._mine_counts[row][col] = count

    def _count_adjacent_mines(self, row: int, col: int) -> int:
        """Count the number of mines adjacent to the given cell."""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self._is_valid_position(r, c) and (r, c) in self._mine_positions:
                    count += 1
        return count

    def _reveal_cell(self, row: int, col: int) -> Tuple[float, Optional[str]]:
        """Reveal the cell at (row, col).

        Returns:
            (reward, error). ``error`` is a human-readable reason the action was
            a no-op (e.g. cell already revealed / flagged) or ``None``.
        """
        if (row, col) in self._revealed_cells:
            return -0.05, f"Cell ({row}, {col}) is already revealed"
        if (row, col) in self._flags_placed:
            return -0.05, f"Cell ({row}, {col}) is flagged; remove the flag first"

        if (row, col) in self._mine_positions:
            self._game_status = GameStatus.LOST
            self._revealed_cells.add((row, col))
            return -10.0, None

        self._reveal_region(row, col)
        return 1.0, None

    def _reveal_region(self, row: int, col: int) -> None:
        """Reveal connected empty cells and their numbered boundary."""
        pending = [(row, col)]
        while pending:
            row, col = pending.pop()
            if not self._is_valid_position(row, col):
                continue
            if (row, col) in self._revealed_cells or (row, col) in self._flags_placed:
                continue
            if (row, col) in self._mine_positions:
                continue

            self._revealed_cells.add((row, col))

            if self._mine_counts[row][col] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        pending.append((row + dr, col + dc))

    def _toggle_flag(self, row: int, col: int) -> Tuple[float, Optional[str]]:
        """Toggle a flag on the cell at (row, col).

        Returns:
            (reward, error). ``error`` is set when the action is rejected
            (e.g. trying to flag a revealed cell).
        """
        if (row, col) in self._revealed_cells:
            return -0.05, f"Cannot flag revealed cell ({row}, {col})"

        if (row, col) in self._flags_placed:
            self._flags_placed.remove((row, col))
            return 0.0, None

        self._flags_placed.add((row, col))
        if (row, col) in self._mine_positions and (
            row,
            col,
        ) not in self._rewarded_flags:
            self._rewarded_flags.add((row, col))
            return 0.5, None
        return 0.0, None

    def _check_win_condition(self) -> None:
        """Check if the game has been won.

        Only counts non-mine cells: hitting a mine adds it to ``_revealed_cells``
        but must never satisfy the win threshold (which would otherwise overwrite
        a freshly-set LOST status on a near-complete board).
        """
        if self._game_status != GameStatus.ONGOING:
            return
        safe_revealed = len(self._revealed_cells - self._mine_positions)
        if safe_revealed == self.height * self.width - self.num_mines:
            self._game_status = GameStatus.WON

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if the given (row, col) is within board bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def _create_observation(
        self,
        done: bool,
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MinesweeperObservation:
        """Create the current observation of the board.
        Args:
            done: Whether the episode is done.
            reward: Reward obtained from the last action.
            metadata: Additional metadata to include.
        Returns:
            MinesweeperObservation representing the current board state.
        """
        board = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) in self._revealed_cells:
                    if (r, c) in self._mine_positions:
                        row.append("*")
                    else:
                        row.append(self._mine_counts[r][c])
                elif (r, c) in self._flags_placed:
                    row.append("F")
                else:
                    row.append(-1)
            board.append(row)

        return MinesweeperObservation(
            board=board,
            num_mines=self.num_mines,
            flags_placed=len(self._flags_placed),
            cells_revealed=len(self._revealed_cells),
            game_status=self._game_status,
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _get_full_state(self) -> MinesweeperState:
        """
        Get the full internal state of the environment (server-side only).

        Exposes the mine layout and is intended for testing / introspection.
        Never call this from a code path reachable by an agent.

        Returns:
            MinesweeperState representing the full internal state
        """
        return MinesweeperState(
            episode_id=self._state.episode_id or "",
            step_count=self._state.step_count,
            board_height=self.height,
            board_width=self.width,
            mine_locations=self._mine_positions,
            revealed_cells=self._revealed_cells,
            flags=self._flags_placed,
            mine_counts=self._mine_counts,
            game_status=self._game_status,
        )

    def get_legal_actions(self) -> List[MinesweeperAction]:
        """
        Get the list of legal actions available in the current state.

        Returns:
            List of MinesweeperAction instances representing legal actions
        """
        legal_actions = []

        # If game is over, no legal actions
        if self._game_status != GameStatus.ONGOING:
            return legal_actions

        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self._revealed_cells and (
                    r,
                    c,
                ) not in self._flags_placed:
                    legal_actions.append(
                        MinesweeperAction(row=r, col=c, action_type="reveal")
                    )
                if (r, c) not in self._revealed_cells:
                    legal_actions.append(
                        MinesweeperAction(row=r, col=c, action_type="flag")
                    )
        return legal_actions
