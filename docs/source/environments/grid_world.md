<!-- openenv-source: grid_world_env -->
# Grid World Environment

A deterministic 5x5 grid world. The agent starts in the top-left corner and must reach the goal in the bottom-right corner. It's a small testbed for RL agents and a compact example of an OpenEnv environment (models, environment, server and client in a few files).

![Grid World web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/grid_world.png)

*The `/web` playground after a few steps: the agent, the goal and one button per direction.*

## Quick Start

Start the server from `envs/grid_world_env/`:

```bash
uv run server
```

Then connect with the client:

```python
from grid_world_env import GridWorldAction, GridWorldEnv

with GridWorldEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(result.observation.message)  # Welcome to Grid World! Goal is at [4, 4].

    for move in ["DOWN"] * 4 + ["RIGHT"] * 4:
        result = env.step(GridWorldAction(action=move))
        print(move, result.observation.x, result.observation.y, result.reward, result.done)
```

The last step reaches `(4, 4)` with reward `1.0` and `done=True`. The API docs are at `http://localhost:8000/docs`.

## Docker

The image builds on the OpenEnv base image. From the repository root:

```bash
docker build -t envtorch-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t grid-world-env:latest -f envs/grid_world_env/server/Dockerfile .
docker run -p 8000:8000 grid-world-env:latest
```

`./envs/grid_world_env/test_grid_world.sh` (run from the repository root) builds both images, starts the container, checks the endpoints with `curl` and cleans up.

## Environment Details

### Action

**GridWorldAction**: `action`, one of `UP`, `DOWN`, `LEFT`, `RIGHT` (`MoveAction` enum or the string). `UP` and `DOWN` change `x`, `LEFT` and `RIGHT` change `y`. A move off the grid leaves the agent in place.

### Observation

**GridWorldObservation**:

- `x`, `y`: agent position, from `(0, 0)` to `(4, 4)`
- `message`: status message
- `reward`, `done`

`state()` returns the standard `State` (`episode_id`, `step_count`).

### Reward

- `-0.1` for every step that doesn't reach the goal
- `+1.0` for reaching the goal at `(4, 4)`, which ends the episode

There is no step limit: the episode only ends at the goal.

## Code Layout

- `models.py`: `MoveAction`, `GridWorldAction` and `GridWorldObservation`
- `server/grid_world_environment.py`: `GridWorldEnvironment` (reset, step, reward)
- `server/app.py`: the server, built with `create_app`
- `client.py`: `GridWorldEnv`, with a `step_move(MoveAction.UP)` helper

To build your own environment, see [Your First Environment](https://huggingface.co/docs/openenv/guides/first-environment).
