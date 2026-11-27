"""Minesweeper Agent Loop Example.

This script demonstrates how to create an agent that interacts with the Minesweeper
environment. The agent uses a simple strategy to play the game.

Prerequisites:
- Minesweeper Docker container must be running
- Default URL: http://localhost:8000

Usage:
    python examples/minesweeper_agent.py
"""

import random
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Add the local library and environment packages to the import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

sys.path.insert(0, str(Path(__file__).parent.parent / "envs"))

from minesweeper_env import GameStatus, MinesweeperAction, MinesweeperEnv

# Configuration
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 100
BASE_URL = "http://localhost:8000"


class SimpleMinesweeperAgent:
    """
    A simple agent that plays Minesweeper using a basic strategy:
    1. Start by revealing a corner cell (safer statistically)
    2. Reveal cells with 0 adjacent mines first
    3. Flag cells that are definitely mines based on revealed numbers
    4. Make random safe moves when no obvious safe cells exist
    """

    def __init__(self):
        self.revealed_positions: Set[Tuple[int, int]] = set()
        self.flagged_positions: Set[Tuple[int, int]] = set()
        self.board_height = 0
        self.board_width = 0

    def reset(self):
        """Reset agent state for a new episode."""
        self.revealed_positions.clear()
        self.flagged_positions.clear()
        self.board_height = 0
        self.board_width = 0

    def choose_action(self, observation) -> MinesweeperAction:
        """
        Choose the next action based on the current board state.

        Args:
            observation: MinesweeperObservation from the environment

        Returns:
            MinesweeperAction to take
        """
        board = observation.board
        self.board_height = len(board)
        self.board_width = len(board[0]) if board else 0

        # Update what we know
        self._update_state(board)

        # Strategy 1: If this is the first move, reveal a corner (statistically safer)
        if len(self.revealed_positions) == 0:
            return MinesweeperAction(row=0, col=0, action_type="reveal")

        # Strategy 2: Look for cells that are definitely safe
        safe_cells = self._find_safe_cells(board)
        if safe_cells:
            row, col = random.choice(list(safe_cells))
            return MinesweeperAction(row=row, col=col, action_type="reveal")

        # Strategy 3: Look for cells that are definitely mines and flag them
        mine_cells = self._find_definite_mines(board)
        if mine_cells:
            row, col = random.choice(list(mine_cells))
            return MinesweeperAction(row=row, col=col, action_type="flag")

        # Strategy 4: Make a random move on unrevealed cells (risky but necessary)
        unrevealed = self._get_unrevealed_unflagged_cells(board)
        if unrevealed:
            row, col = random.choice(list(unrevealed))
            return MinesweeperAction(row=row, col=col, action_type="reveal")

        # Fallback: no-op (shouldn't reach here)
        return MinesweeperAction(row=0, col=0, action_type="reveal")

    def _update_state(self, board):
        """Update internal state based on current board."""
        for r in range(len(board)):
            for c in range(len(board[0])):
                cell = board[r][c]
                if cell != -1 and cell != "F":
                    self.revealed_positions.add((r, c))
                if cell == "F":
                    self.flagged_positions.add((r, c))

    def _get_unrevealed_unflagged_cells(self, board) -> Set[Tuple[int, int]]:
        """Get all cells that are unrevealed and not flagged."""
        unrevealed = set()
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == -1:
                    unrevealed.add((r, c))
        return unrevealed

    def _find_safe_cells(self, board) -> Set[Tuple[int, int]]:
        """
        Find cells that are definitely safe based on revealed numbers.

        A cell is safe if it's adjacent to a revealed cell whose mine count
        equals the number of flags around it.
        """
        safe_cells = set()

        for r in range(len(board)):
            for c in range(len(board[0])):
                cell = board[r][c]

                # Only check revealed numbered cells
                if isinstance(cell, int) and 0 <= cell <= 8:
                    neighbors = self._get_neighbors(r, c)
                    unrevealed_neighbors = [
                        (nr, nc) for nr, nc in neighbors if board[nr][nc] == -1
                    ]
                    flagged_neighbors = [
                        (nr, nc) for nr, nc in neighbors if board[nr][nc] == "F"
                    ]

                    # If all mines are flagged, remaining unrevealed cells are safe
                    if len(flagged_neighbors) == cell:
                        safe_cells.update(unrevealed_neighbors)

        return safe_cells

    def _find_definite_mines(self, board) -> Set[Tuple[int, int]]:
        """
        Find cells that are definitely mines based on revealed numbers.

        A cell is definitely a mine if it's adjacent to a revealed cell whose
        mine count equals the number of unrevealed + flagged neighbors.
        """
        mine_cells = set()

        for r in range(len(board)):
            for c in range(len(board[0])):
                cell = board[r][c]

                # Only check revealed numbered cells with value > 0
                if isinstance(cell, int) and 1 <= cell <= 8:
                    neighbors = self._get_neighbors(r, c)
                    unrevealed_neighbors = [
                        (nr, nc) for nr, nc in neighbors if board[nr][nc] == -1
                    ]
                    flagged_neighbors = [
                        (nr, nc) for nr, nc in neighbors if board[nr][nc] == "F"
                    ]

                    # If unrevealed + flagged equals mine count,
                    # all unrevealed are mines
                    if len(unrevealed_neighbors) + len(flagged_neighbors) == cell:
                        mine_cells.update(unrevealed_neighbors)

        return mine_cells

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid neighboring cells."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_height and 0 <= nc < self.board_width:
                    neighbors.append((nr, nc))
        return neighbors


def print_board(board):
    """Pretty print the Minesweeper board."""
    print("\n  ", end="")
    if board:
        for c in range(len(board[0])):
            print(f"{c:3}", end="")
    print()

    for r, row in enumerate(board):
        print(f"{r:2} ", end="")
        for cell in row:
            if cell == -1:
                print(" . ", end="")
            elif cell == "F":
                print(" F ", end="")
            elif cell == "*":
                print(" * ", end="")
            else:
                print(f" {cell} ", end="")
        print()
    print()


def main():
    """Run the agent loop."""
    print("=" * 60)
    print("Minesweeper Agent Loop")
    print("=" * 60)

    # Connect to the environment
    print(f"Connecting to Minesweeper server at {BASE_URL}...")

    # Option 1: Connect to existing server
    env = MinesweeperEnv(base_url=BASE_URL).sync()

    # Option 2: Start from Docker image (uncomment to use)
    # env = MinesweeperEnv.from_docker_image("minesweeper_env-env:latest")

    agent = SimpleMinesweeperAgent()

    total_wins = 0
    total_losses = 0

    try:
        for episode in range(1, MAX_EPISODES + 1):
            print(f"\n{'=' * 60}")
            print(f"Episode {episode}/{MAX_EPISODES}")
            print(f"{'=' * 60}")

            agent.reset()
            result = env.reset()
            observation = result.observation

            print(f"Board size: {observation.board_height}x{observation.board_width}")
            print(f"Number of mines: {observation.num_mines}")
            print_board(observation.board)

            episode_reward = 0.0

            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                if result.done:
                    break

                # Agent chooses action
                action = agent.choose_action(observation)

                # Execute action
                result = env.step(action)
                observation = result.observation
                reward = result.reward or 0.0
                episode_reward += reward

                print(
                    f"Step {step}: {action.action_type} ({action.row}, {action.col}) -> reward: {reward:+.2f}"
                )
                print_board(observation.board)

                # Handle game_status being either an Enum or an int
                if isinstance(observation.game_status, GameStatus):
                    status_name = observation.game_status.name
                elif isinstance(observation.game_status, int):
                    status_name = GameStatus(observation.game_status).name
                else:
                    status_name = str(observation.game_status)

                print(
                    f"Status: {status_name} | Revealed: {observation.cells_revealed} | Flags: {observation.flags_placed}"
                )

                if result.done:
                    if status_name == "WON":
                        print(
                            f"\n🎉 Episode {episode}: WON! Total reward: {episode_reward:.2f}"
                        )
                        total_wins += 1
                    elif status_name == "LOST":
                        print(
                            f"\n💥 Episode {episode}: LOST! Total reward: {episode_reward:.2f}"
                        )
                        total_losses += 1
                    break
            else:
                print(
                    f"\n⏰ Episode {episode}: Reached max steps ({MAX_STEPS_PER_EPISODE})"
                )

        # Summary
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"Episodes played: {MAX_EPISODES}")
        print(f"Wins: {total_wins} ({100 * total_wins / MAX_EPISODES:.1f}%)")
        print(f"Losses: {total_losses} ({100 * total_losses / MAX_EPISODES:.1f}%)")

    finally:
        env.close()
        print("\nEnvironment closed.")


if __name__ == "__main__":
    main()
