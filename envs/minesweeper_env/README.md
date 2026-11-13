# Minesweeper Environment

A Minesweeper game environment for reinforcement learning agents. The environment consists of a grid with hidden mines where the agent must reveal all non-mine cells without triggering any mines.

## Overview

The agent can perform two types of actions:
- Reveal cells to uncover numbers indicating adjacent mines
- Place or remove flags on suspected mine locations

The game ends when all non-mine cells are revealed (win) or a mine is revealed (loss).

## Quick Start

```python
from minesweeper_env import MinesweeperAction, MinesweeperEnv

# Create environment from Docker image
minesweeper_env = MinesweeperEnv.from_docker_image("minesweeper-env:latest").sync()

try:
    # Reset the environment
    result = minesweeper_env.reset()
    print(f"Board size: {result.observation.board_height}x{result.observation.board_width}")
    print(f"Number of mines: {result.observation.num_mines}")

    # Reveal a cell
    result = minesweeper_env.step(MinesweeperAction(row=2, col=2, action_type="reveal"))
    print(f"Cells revealed: {result.observation.cells_revealed}")
    print(f"Reward: {result.observation.reward}")

    # Place a flag
    result = minesweeper_env.step(MinesweeperAction(row=1, col=1, action_type="flag"))
    print(f"Flags placed: {result.observation.flags_placed}")

finally:
    minesweeper_env.close()
```

## Building the Docker Image

Build the Docker image from the project root:

```bash
docker build -t minesweeper-env:latest -f envs/minesweeper_env/server/Dockerfile envs/minesweeper_env
```

Or use the build script:

```bash
cd envs/minesweeper_env/server
./build_docker.sh latest
```

## Environment Details

### Action

**MinesweeperAction**: Specifies the cell and action type
- `row` (int) - Row index (0-indexed)
- `col` (int) - Column index (0-indexed)
- `action_type` (str) - Either "reveal" or "flag"

### Observation

**MinesweeperObservation**: Current board state and game information
- `board` (list[list]) - 2D grid showing the current state of each cell:
  - `-1`: Unrevealed cell
  - `0-8`: Number of adjacent mines (revealed cell)
  - `'F'`: Flagged cell
  - `'*'`: Mine (only shown when game is lost)
- `num_mines` (int) - Total number of mines on the board
- `flags_placed` (int) - Number of flags currently placed
- `cells_revealed` (int) - Number of cells that have been revealed
- `game_status` (GameStatus) - Current game status (ONGOING, WON, or LOST)
- `done` (bool) - Whether the game has ended
- `reward` (float) - Reward from the last action
- `metadata` (dict) - Additional information

### Rewards

- Revealing a safe cell: +1.0
- Placing a flag on a mine: +0.5 once per mine per episode. Removing or replacing a previously rewarded flag gives 0; reset restores bonus eligibility.
- Revealing a mine (game over): -10.0
- Revealing an already revealed cell: -0.05
- Invalid action: -0.1

### Game Status

- `GameStatus.ONGOING`: Game is still in progress
- `GameStatus.WON`: All non-mine cells have been revealed
- `GameStatus.LOST`: A mine was revealed

## Configuration

The default configuration is:
- Board height: 5
- Board width: 5
- Number of mines: 5

These can be configured when initializing the environment server.

## Connecting to an Existing Server

If you have a server already running:

```python
from minesweeper_env import MinesweeperEnv

# Connect to existing server
minesweeper_env = MinesweeperEnv(base_url="http://localhost:8000")

# Use as normal
result = minesweeper_env.reset()
```

Note: When connecting to an existing server, `close()` will not stop the server.

## Running Tests

Run the test suite:

```bash
python tests/envs/test_minesweeper_env.py
```

## Project Structure

```
minesweeper_env/
├── __init__.py                    # Module exports
├── README.md                      # This file
├── client.py                      # MinesweeperEnv client implementation
├── models.py                      # Action, Observation, and State models
├── openenv.yaml                   # Environment configuration
├── pyproject.toml                 # Package dependencies
└── server/
    ├── __init__.py                # Server module exports
    ├── minesweeper_environment.py # Core game logic
    ├── app.py                     # FastAPI application
    ├── Dockerfile                 # Container image definition
    └── build_docker.sh            # Build script
```
