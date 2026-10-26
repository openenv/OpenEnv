---
title: Connect4 Environment
emoji: 🔴
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Connect4 Environment

A classic Connect Four board game environment for training agents on turn-based strategy with a 6×7 grid. Players alternate dropping pieces into columns, aiming to connect four in a row horizontally, vertically, or diagonally.

![Connect4 web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/connect4.png)

*The `/web` playground after a few moves: the board and one button per column.*

## Quick Start

```python
import asyncio
from connect4_env import Connect4Action, Connect4Env

async def main():
    async with Connect4Env(base_url="http://localhost:8000") as client:
        result = await client.reset()
        print(f"Board: {result.observation.board}")
        print(f"Legal moves: {result.observation.legal_actions}")

        # Drop a piece in column 3
        result = await client.step(Connect4Action(column=3))
        print(f"Reward: {result.reward}, Done: {result.done}")

asyncio.run(main())
```

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `column` | `int` | Column index (0–6) where the piece will be dropped |

Invalid moves (out-of-range or full column) result in a reward of `-1` and end the episode.

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `board` | `list[list[int]]` | 6×7 grid — `1` = current player, `-1` = opponent, `0` = empty |
| `legal_actions` | `list[int]` | Column indices that are valid moves |

## Rewards

| Outcome | Reward |
|---------|--------|
| Win (4 in a row) | `+1.0` |
| Draw (board full) | `0.0` |
| Invalid move | `-1.0` |
| Otherwise | `0.0` |

## State

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | `str` | Unique ID for the current game |
| `board` | `list[list[int]]` | Current board state |
| `next_player` | `int` | Whose turn it is (`1` or `-1`) |
| `step_count` | `int` | Number of steps taken |
