# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Minesweeper Environment.

The minesweeper_env environment is a Minesweeper game where agents reveal cells and place flags
to identify mines on a grid board.
"""

from enum import Enum
from typing import List, Literal, Set, Tuple

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class GameStatus(Enum):
    """Status of the Minesweeper game."""

    ONGOING = "ongoing"
    WON = "won"
    LOST = "lost"


class MinesweeperAction(Action):
    """
    Action for the Minesweeper environment.

    Attributes:
        row: Row index of the cell to act on (0-indexed).
        col: Column index of the cell to act on (0-indexed).
        action_type: Type of action - 'reveal' to uncover a cell, 'flag' to place/remove a flag.
    """

    row: int = Field(..., ge=0, description="Row index of the cell")
    col: int = Field(..., ge=0, description="Column index of the cell")
    action_type: str = Field(
        ..., pattern="^(reveal|flag)$", description="Type of action: 'reveal' or 'flag'"
    )


class MinesweeperObservation(Observation):
    """
    Observation from the Minesweeper environment.

    This represents what the agent can see - a partial view of the board with hidden mine locations (unless revealed).

    Attributes:
        board: 2D list representing the current state of the board. Each cell can be:
            - -1: unrevealed
            - 0-8: number of adjacent mines (if revealed)
            - 'F': flagged cell
            - '*': mine (only revealed if game is lost)
        num_mines: Total number of mines on the board.
        flags_placed: Number of flags currently placed by the agent.
        cells_revealed: Number of cells that have been revealed so far.
        game_status: Current status of the game - ongoing, won, or lost.
    """

    board: List[List[Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, "F", "*"]]] = Field(
        default_factory=list, description="2D board state"
    )
    num_mines: int = Field(..., ge=0, description="Total number of mines")
    flags_placed: int = Field(..., ge=0, description="Number of flags placed")
    cells_revealed: int = Field(..., ge=0, description="Number of cells revealed")
    game_status: GameStatus = Field(..., description="Current game status")

    @property
    def board_height(self) -> int:
        """Height of the board (number of rows)."""
        return len(self.board)

    @property
    def board_width(self) -> int:
        """Width of the board (number of columns)."""
        return len(self.board[0]) if self.board else 0


class MinesweeperState(BaseModel):
    """
    Internal state of the Minesweeper environment.

    This represents the full internal state of the environment, including hidden information.

    Attributes:
        episode_id: Unique identifier for the current episode.
        step_count: Number of steps taken in the current episode.
        board_height: Height of the board (number of rows).
        board_width: Width of the board (number of columns).
        mine_locations: Set of (row, col) tuples indicating where mines are located.
        revealed_cells: Set of (row, col) tuples indicating which cells have been revealed.
        flags: Set of (row, col) tuples indicating where flags have been placed.
        mine_counts: 2D list with counts of adjacent mines for each cell.
        game_status: Current status of the game - ongoing, won, or lost.
    """

    episode_id: str
    step_count: int
    board_height: int
    board_width: int
    mine_locations: Set[Tuple[int, int]]
    revealed_cells: Set[Tuple[int, int]]
    flags: Set[Tuple[int, int]]
    mine_counts: List[List[int]]
    game_status: GameStatus

    def to_observation(self) -> MinesweeperObservation:
        """
        Convert the full state to a partial observation for the agent.

        Returns:
            MinesweeperObservation representing the agent's view of the board.
        """
        board = []
        for r in range(self.board_height):
            row = []
            for c in range(self.board_width):
                if (r, c) in self.revealed_cells:
                    if (r, c) in self.mine_locations:
                        cell_value = "*"  # Revealed mine
                    else:
                        cell_value = self.mine_counts[r][c]  # Number of adjacent mines
                elif (r, c) in self.flags:
                    cell_value = "F"  # Flagged cell
                else:
                    cell_value = -1  # Unrevealed cell
                row.append(cell_value)
            board.append(row)

        return MinesweeperObservation(
            board=board,
            num_mines=len(self.mine_locations),
            flags_placed=len(self.flags),
            cells_revealed=len(self.revealed_cells),
            game_status=self.game_status,
            done=self.game_status != GameStatus.ONGOING,
            reward=0.0,
            metadata={
                "episode_id": self.episode_id,
                "step_count": self.step_count,
            },
        )
