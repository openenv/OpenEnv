---
title: Maze Environment Server
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Maze Environment

A gridworld maze where the agent must navigate from a start cell to an exit while avoiding walls.

![Maze web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/maze.png)

*The `/web` playground after a few steps: the maze, the agent, the exit and one button per legal move.*

## Maze Layout

The default environment ships with a single 8x8 maze:

```text
0 0 0 0 0 1 0 0
1 1 0 1 0 1 0 1
0 0 0 1 0 0 0 1
0 1 1 1 1 1 0 0
0 0 0 0 0 0 1 0
1 0 1 1 1 0 0 0
0 0 0 0 1 0 1 0
0 1 1 0 0 0 0 0
```

`0` is an empty cell and `1` is a wall. Coordinates are `(col, row)` with `(0, 0)` at the upper-left.

## Quick Start

The simplest way to use the Maze environment is through the `MazeEnv` client:

```python
from maze_env import MazeAction, MazeEnv

try:
    # Create environment from Docker image
    env = MazeEnv.from_docker_image("maze_env-env:latest").sync()

    # Reset to start a new episode
    result = env.reset()
    print(f"Start position: {result.observation.current_position}")
    print(f"Legal actions: {result.observation.legal_actions}")

    # Play until done
    while not result.done:
        action_id = result.observation.legal_actions[0]
        result = env.step(MazeAction(action=action_id))
        print(f"Position: {result.observation.current_position}")
        print(f"Reward: {result.reward}, Done: {result.done}")

finally:
    # Always clean up
    env.close()
```

That's it! The `MazeEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

From the **environment directory** (`envs/maze_env/`):

```bash
docker build -t maze_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

From `envs/maze_env/`, run `openenv push` (or `openenv push --repo-id my-org/maze-env`). See the [`openenv push` reference](https://huggingface.co/docs/openenv/reference/cli#openenv-push) for all options.

## Running the Default Maze

```bash
docker run -p 8000:8000 maze_env-env:latest
```

## Environment Details

### Action
**MazeAction**: Contains a single field
- `action` (int) - Movement action
  - 0 = move up
  - 1 = move down
  - 2 = move left
  - 3 = move right

### Observation
**MazeObservation**: Per-step observation payload
- `current_position` (List[int]) - Agent position as `[col, row]`
- `legal_actions` (List[int]) - Valid actions from the current position
- `metadata` (dict) - Extra info:
  - `maze`: 2D grid (0 = empty, 1 = wall)
  - `status`: "playing", "win", or "lose"
  - `exit_cell`: Exit position as `[col, row]`
  - `step`: Current step count

### State
**MazeState**: Server-side state snapshot
- `episode_id` (str) - Unique identifier for the current episode
- `step_count` (int) - Number of steps taken
- `done` (bool) - Whether the episode has ended
- `current_position` (List[int]) - Current agent position
- `exit_cell` (List[int]) - Exit position
- `status` (str) - "playing", "win", or "lose"

### Reward
The reward follows the underlying maze rules:
- Small penalty for a move (`-0.05`)
- Penalty for revisiting a cell (`-0.25`)
- Penalty for invalid move (`-0.75`)
- Reward for reaching the exit (`+10.0`)

The episode ends with `lose` when the accumulated reward drops below `-0.5 * maze.size` (`-32` for the default maze).

## Configuration

`MAX_CONCURRENT_ENVS` sets the maximum number of concurrent WebSocket sessions (default: `8`). To use another layout, construct `MazeEnvironment(maze_array=..., start_cell=(col, row), exit_cell=(col, row))` in `server/app.py`. The exit defaults to the bottom-right cell.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Maze environment server running:

```python
from maze_env import MazeAction, MazeEnv

# Connect to existing server
env = MazeEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
result = env.step(MazeAction(action=result.observation.legal_actions[0]))

# Close connection (does NOT stop the server)
env.close()
```

### Connecting to HuggingFace Space

```python
from maze_env import MazeAction, MazeEnv

# Connect to remote Space
env = MazeEnv(base_url="https://your-username-maze-env.hf.space")

result = env.reset()
print(f"Position: {result.observation.current_position}")
print(f"Legal actions: {result.observation.legal_actions}")

result = env.step(MazeAction(action=result.observation.legal_actions[0]))
env.close()
```


## References

- [Reinforcement-Learning-Maze (original implementation)](https://github.com/erikdelange/Reinforcement-Learning-Maze)
