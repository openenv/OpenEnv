---
title: Snake Environment Server
emoji: 🐉
colorFrom: 'blue'
colorTo: 'green'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Snake Environment

A multi-agent snake game environment for OpenEnv, based on [marlenv](https://github.com/kc-ml2/marlenv)'s Snake-v1. This environment provides a single-agent interface to the classic snake game where the snake must navigate a grid, eat fruits, and avoid walls and its own body.

![Snake web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/snake.png)

*The `/web` playground after a few steps: the snake, the fruit and one button per move.*

## Overview

The Snake environment wraps the marlenv Snake-v1 environment to provide a clean OpenEnv-compatible interface. Multiple snakes can battle on a fixed size grid map, but this implementation focuses on single-agent gameplay.

### Features

- **Grid-based gameplay**: Configurable grid size (default: 20x20)
- **Fruit collection**: Snake grows when eating fruits
- **Partial observability**: Optional vision range for limited field of view
- **Customizable rewards**: Configurable reward function for different game aspects
- **Two control modes**:
  - `snake`: Relative actions (turn left/right)
  - `human`: Global directions (up/down/left/right)

### Game Rules

- Snake dies when its head hits a wall or its own body
- Snake grows by one unit when it eats a fruit
- Episode ends when the snake dies or reaches maximum steps
- Rewards can be customized for: eating fruits, survival time, and death penalty

## Quick Start

### Using Docker (Recommended)

```python
from envs.snake_env import SnakeAction, SnakeEnv

# Start environment from Docker image
client = SnakeEnv.from_docker_image("snake-env:latest")

# Reset to start new episode
result = client.reset()
print(f"Snake alive: {result.observation.alive}")
print(f"Grid shape: {len(result.observation.grid)}x{len(result.observation.grid[0])}")

# Take actions
result = client.step(SnakeAction(action=0))  # Continue straight
print(f"Reward: {result.reward}")
print(f"Score: {result.observation.episode_score}")

result = client.step(SnakeAction(action=1))  # Turn left
result = client.step(SnakeAction(action=2))  # Turn right

# Check game state
state = client.state()
print(f"Episode: {state.episode_id}")
print(f"Steps: {state.step_count}")

# Cleanup
client.close()
```

### Using Local Server

```bash
# Install dependencies
cd envs/snake_env
pip install -e .

# Run server
uv run --project . server
```

Then connect from another terminal:

```python
from envs.snake_env import SnakeAction, SnakeEnv

# Connect to running server
client = SnakeEnv(base_url="http://localhost:8000")
result = client.reset()
result = client.step(SnakeAction(action=0))
```

## Actions

The action space depends on the `observer` mode:

### Snake Mode (Default)
Relative actions based on current direction:
- `0`: No-op (continue in same direction)
- `1`: Turn left (90 degrees counterclockwise)
- `2`: Turn right (90 degrees clockwise)

### Human Mode
Global directional actions:
- `0`: No-op
- `1`: Move left
- `2`: Move right
- `3`: Move down
- `4`: Move up

## Observations

Each observation includes:

- `grid`: The full game grid as a 2D array (height × width)
- `observation`: Encoded observation based on vision range
- `episode_score`: Cumulative score in current episode
- `episode_steps`: Number of steps taken
- `episode_fruits`: Number of fruits eaten
- `episode_kills`: Number of kills (always 0 in single-agent mode)
- `alive`: Whether the snake is still alive

## Configuration

### Environment Parameters

```python
from envs.snake_env.server.snake_environment import SnakeEnvironment

env = SnakeEnvironment(
    height=20,           # Grid height (default: 20)
    width=20,            # Grid width (default: 20)
    snake_length=3,      # Initial snake length (default: 3)
    vision_range=5,      # Partial observability (None for full grid)
    observer='snake',    # 'snake' or 'human' mode
    max_episode_steps=1000,  # Maximum steps per episode
    reward_dict={        # Custom reward function
        'fruit': 1.0,    # Reward for eating fruit
        'kill': 0.0,     # Reward for kills (multi-agent)
        'lose': -1.0,    # Penalty for death
        'win': 0.0,      # Reward for winning (multi-agent)
        'time': 0.0,     # Reward per timestep
    }
)
```

### Custom Rewards

You can customize the reward function to encourage different behaviors:

```python
# Encourage survival
reward_dict = {
    'fruit': 1.0,
    'lose': -10.0,
    'time': 0.01,  # Small reward for staying alive
}

# Fast fruit collection
reward_dict = {
    'fruit': 10.0,
    'lose': -1.0,
    'time': -0.01,  # Penalty for taking too long
}
```

## Building and Deployment

### Build Docker Image

From the repository root:

```bash
# Build base image first (if not already built)
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build snake environment image
docker build -t snake-env:latest -f envs/snake_env/server/Dockerfile .
```

The Dockerfile uses `pip install` with `requirements.txt` for maximum compatibility.

### Run Docker Container

```bash
# Run the container
docker run -p 8000:8000 snake-env:latest

# Or with environment variables
docker run -p 8000:8000 \
  -e ENABLE_WEB_INTERFACE=true \
  snake-env:latest
```

### Web Interface

When `ENABLE_WEB_INTERFACE=true` is set, you can access the web interface at `http://localhost:8000/web` to interact with the environment through your browser.

## Dependencies

The snake environment requires:

- `marlenv`: Multi-agent snake game implementation
- `gym==0.24.1`: OpenAI Gym (required by marlenv)
- `numpy`: Numerical operations
- Standard OpenEnv dependencies (fastapi, pydantic, uvicorn)

These are automatically installed when using Docker or installing via pip.

## Example Training Loop

```python
from envs.snake_env import SnakeAction, SnakeEnv
import random

# Connect to environment
env = SnakeEnv.from_docker_image("snake-env:latest")

# Training loop
for episode in range(10):
    result = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Simple random policy (replace with your agent)
        action = SnakeAction(action=random.randint(0, 2))
        result = env.step(action)

        total_reward += result.reward
        done = result.done

    print(f"Episode {episode}: Reward={total_reward}, "
          f"Fruits={result.observation.episode_fruits}, "
          f"Steps={result.observation.episode_steps}")

env.close()
```

## Troubleshooting

### marlenv Installation Issues

If you encounter issues installing marlenv, you can install it from source:

```bash
pip install git+https://github.com/kc-ml2/marlenv.git
```

### Import Errors

Make sure you're in the correct directory when running the server:

```bash
cd envs/snake_env
uv run --project . server
```

### Docker Build Issues

Ensure the base image is built first:

```bash
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
```

## Citation

The underlying snake game is from marlenv:

```bibtex
@MISC{marlenv2021,
    author = {ML2},
    title = {Marlenv, Multi-agent Reinforcement Learning Environment},
    howpublished = {\url{http://github.com/kc-ml2/marlenv}},
    year = {2021}
}
```

## License

BSD 3-Clause License - See LICENSE file in the root directory.
