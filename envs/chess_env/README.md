---
title: Chess Environment Server
emoji: ♟️
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Chess Environment

A chess reinforcement learning environment for OpenEnv, powered by the [moonfish](https://github.com/luccabb/moonfish) chess engine.

![Chess web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/chess.png)

*The `/web` playground after a few moves: the board and one button per legal move.*

## Features

- **Full chess rules** via python-chess library
- **Configurable opponent**: moonfish engine, random moves, or self-play
- **Position evaluation**: Uses moonfish's PSQT-based evaluation
- **Standard OpenEnv interface**: reset(), step(), state

## Quick Start

### Using Docker

```bash
# Build the image
docker build -t chess-env:latest -f envs/chess_env/server/Dockerfile .

# Run the server
docker run -p 8000:8000 chess-env:latest
```

### Using the Client

The client is **async by default**:

```python
import asyncio
from chess_env import ChessEnv, ChessAction

async def main():
    async with ChessEnv(base_url="http://localhost:8000") as env:
        # Reset for a new game
        result = await env.reset()
        print(f"Starting position: {result.observation.fen}")
        print(f"Legal moves: {result.observation.legal_moves}")

        # Make a move
        result = await env.step(ChessAction(move="e2e4"))
        print(f"Reward: {result.reward}, Done: {result.done}")

        # Play until game ends
        while not result.done:
            move = result.observation.legal_moves[0]
            result = await env.step(ChessAction(move=move))

        print(f"Game result: {result.observation.result}")

asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
from chess_env import ChessEnv, ChessAction

with ChessEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(ChessAction(move="e2e4"))
    print(f"Reward: {result.reward}")
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `fen` | str | Board position in FEN notation |
| `legal_moves` | List[str] | Legal moves in UCI format |
| `is_check` | bool | Whether current player is in check |
| `done` | bool | Whether game has ended |
| `reward` | float | Reward for last action |
| `result` | str | Game result ("1-0", "0-1", "1/2-1/2") |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `move` | str | UCI format move (e.g., "e2e4", "e7e8q") |

## Rewards

| Outcome | Reward |
|---------|--------|
| Win | +1.0 |
| Loss | -1.0 |
| Draw | 0.0 |
| Illegal move | -0.1 |

## Configuration

The environment supports these configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opponent` | "moonfish" | Opponent type: "moonfish", "random", or None |
| `opponent_depth` | 2 | Search depth for moonfish opponent |
| `max_moves` | 500 | Maximum half-moves before draw |
| `agent_color` | None | Agent color: "white", "black", or None (alternate each episode) |
| `gamma` | 0.99 | Discount factor for temporal credit assignment |

## Temporal Discounting

For RL training, the environment computes temporally discounted rewards at episode end. This helps with credit assignment in long games where only the final outcome is known.

When an episode ends, the terminal observation's `metadata` includes:
- `discounted_rewards`: List of discounted rewards for each agent move
- `gamma`: The discount factor used

The formula is `r_t = γ^(T-1-t) × R_final` where:
- `T` = total agent moves
- `t` = move index (0-indexed)
- `R_final` = terminal reward (+1, -1, or 0)

Example for a 5-move win with γ=0.99:
```
Move 0: 0.99^4 × 1.0 = 0.961
Move 1: 0.99^3 × 1.0 = 0.970
Move 2: 0.99^2 × 1.0 = 0.980
Move 3: 0.99^1 × 1.0 = 0.990
Move 4: 0.99^0 × 1.0 = 1.000
```

## Links

- [moonfish GitHub](https://github.com/luccabb/moonfish)
- [Play online](https://huggingface.co/spaces/luccabb/moonfish_chess)
