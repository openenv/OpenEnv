# Building Your Own Environment with OpenEnv

This guide walks you through creating a custom environment using the `OpenEnv` framework and the `openenv` CLI.

The CLI handles scaffolding, builds, validation, and deployment so you can stay focused on environment logic.

## Overview

A typical workflow looks like:

1. Scaffold a new environment with `openenv init`.
2. Customize your models, environment logic, and FastAPI server.
3. Implement a typed `EnvClient` (WebSocket-based for persistent sessions).
4. Configure dependencies and the Dockerfile once.
5. Use the CLI (`openenv build`, `openenv validate`, `openenv push`) to package and share your work.

!!! note
    These integrations are handled automatically by the `openenv` CLI when you run `openenv init`.

### Prerequisites

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) for dependency locking
- Docker Desktop / Docker Engine
- The OpenEnv library installed: `pip install https://github.com/huggingface/OpenEnv.git`

## Step-by-Step Guide

Let's walk through the process of building a custom environment with OpenEnv.

### 1. Scaffold with `openenv init`

```bash
# Run from anywhere – defaults to current directory
openenv init my_env

# Optionally choose an output directory
openenv init my_env --output-dir /Users/you/envs
```

The command creates a fully-typed template with `openenv.yaml`, `pyproject.toml`, `uv.lock`, Docker assets, and stub implementations. If you're working inside this repo, move the generated folder under `envs/`.

Typical layout:

```
my_env/
├── __init__.py
├── README.md
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py
    ├── my_environment.py
    ├── requirements.txt
    └── Dockerfile
```

Python classes are generated for the action, observation, environment, and client. For example, you will find `MyEnvironment`, `MyAction`, `MyObservation`, and `MyEnv` (client) in the `my_env` directory based on the name you provided. The environment uses the core `State` class from `openenv.core.env_server.types`.

### 2. Define Models

Edit `models.py` to describe your action and observation using Pydantic:

```python
# models.py
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class MyAction(Action):
    """Your custom action."""
    command: str = Field(..., description="Command to execute")
    parameters: dict = Field(default_factory=dict, description="Command parameters")

class MyObservation(Observation):
    """Your custom observation."""
    result: str = Field(..., description="Result of the action")
    success: bool = Field(..., description="Whether the action succeeded")
```

### 3. Implement Environment Logic

Customize `server/my_environment.py` by extending `Environment`:

```python
# server/my_environment.py
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import MyAction, MyObservation

class MyEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> MyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return MyObservation(result="Ready", success=True, done=False, reward=0.0)

    def step(self, action: MyAction) -> MyObservation:
        # Implement your logic here
        self._state.step_count += 1
        result = self._execute_command(action.command)
        return MyObservation(result=result, success=True, done=False, reward=1.0)

    @property
    def state(self) -> State:
        return self._state
```

### 4. Create the FastAPI Server

`server/app.py` should expose the environment through `create_app`.

**Important:** You must pass a class or factory function (not an instance) to enable WebSocket-based concurrent sessions:

```python
# server/app.py
from openenv.core.env_server import create_app
from ..models import MyAction, MyObservation
from .my_environment import MyEnvironment

# Pass the class (factory) - each WebSocket session gets its own instance
app = create_app(MyEnvironment, MyAction, MyObservation, env_name="my_env")
```

For environments with constructor arguments, create a factory function:

```python
# server/app.py
import os
from openenv.core.env_server import create_app
from ..models import MyAction, MyObservation
from .my_environment import MyEnvironment

# Read config from environment variables
api_key = os.getenv("MY_API_KEY")
timeout = int(os.getenv("MY_TIMEOUT", "30"))

def create_my_environment():
    """Factory function that creates MyEnvironment with config."""
    return MyEnvironment(api_key=api_key, timeout=timeout)

# Pass the factory function
app = create_app(create_my_environment, MyAction, MyObservation, env_name="my_env")
```

### 5. Implement the Client

`client.py` extends `EnvClient` so users can interact with your server via WebSocket for persistent sessions:

```python
# client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import MyAction, MyObservation, MyState

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"command": action.command, "parameters": action.parameters}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        obs = MyObservation(
            result=obs_data.get("result", ""),
            success=obs_data.get("success", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
```

The `EnvClient` maintains a persistent WebSocket connection to the server, enabling efficient multi-step interactions with lower latency compared to HTTP. Each client instance gets its own dedicated environment session on the server.

### 6. Configure Dependencies & Dockerfile

The CLI template ships with `pyproject.toml` and `server/Dockerfile`. You should manage your python dependencies with `uv` or `pip` in the `pyproject.toml` file. Other dependencies should be installed in the Dockerfile.

Keep building from the `openenv-base` image so shared tooling stays available:

<details>
<summary>Dockerfile</summary>

```dockerfile
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local src/core)
# - Standalone environments (with openenv from pip)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=__ENV_NAME__

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already in the pyproject.toml dependencies
# For standalone builds, openenv will be installed from pip via pyproject.toml
WORKDIR /app/env

# Install dependencies using uv sync
# If uv.lock exists, use it; otherwise resolve on the fly
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# The module path is constructed to work with the /app/env structure
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]

```

</details>

If you introduced extra dependencies in the Dockerfile, you should install them in the Dockerfile before removing temp files.

### 7. Build & Validate with the CLI

From the environment directory:

```bash
cd envs/my_env
openenv build          # Builds Docker image (auto-detects context)
openenv validate --level static --skip-build
```

`openenv build` understands both standalone environments and in-repo ones. Useful flags:

- `--tag/-t`: override the default `openenv-<env_name>` tag
- `--build-arg KEY=VALUE`: pass multiple Docker build arguments
- `--dockerfile` / `--context`: custom locations when experimenting
- `--no-cache`: force fresh dependency installs

`openenv validate` reads the `validation:` contract in `openenv.yaml`, checks
the normalized manifest against the selected severity policy, and exits
non-zero when a required check fails. The current walking skeleton runs the
static manifest check; the report's `levels_run` field records exactly which
levels executed. Use `openenv validate --url http://localhost:8000` separately
to validate a running endpoint.

You can also validate a running environment endpoint and get criteria-level JSON:

```bash
openenv validate --url http://localhost:8000
# or
openenv validate https://username-my-env.hf.space
```

Example runtime output:

```json
{
  "target": "http://localhost:8000",
  "validation_type": "running_environment",
  "standard_version": "1.0.0",
  "passed": true,
  "summary": {
    "passed_count": 6,
    "total_count": 6,
    "failed_criteria": []
  },
  "criteria": [
    {"id": "health_endpoint", "passed": true},
    {"id": "metadata_endpoint", "passed": true},
    {"id": "schema_endpoint", "passed": true}
  ]
}
```

### 8. Push & Share with `openenv push`

Once validation passes, the CLI can deploy directly to Hugging Face Spaces or any registry:

```bash
# Push to HF Spaces (auto enables web UI and prompts for login if needed)
openenv push

# Push to a specific repo or namespace
openenv push --repo-id my-org/my-env

# Push to Docker/ghcr (interface disabled by default)
openenv push --registry ghcr.io/my-org --tag my-env:latest

# Customize image base or visibility
openenv push --base-image ghcr.io/huggingface/openenv-base:latest --private
```

Key options:

- `--directory`: path to the environment (defaults to `cwd`)
- `--repo-id`: explicit Hugging Face space name
- `--registry`: push to Docker Hub, GHCR, etc.
- `--interface/--no-interface`: toggle the optional web UI
- `--base-image`: override the Dockerfile `FROM`
- `--private`: mark the space as private
- Hidden files/directories (`.*`) are excluded by default for Hugging Face uploads
- `--exclude/-e`: optional additional ignore file (newline-separated glob patterns) merged on top of defaults

The command validates your `openenv.yaml`, injects Hugging Face frontmatter when needed, and uploads the prepared bundle.

### 9. Automate Builds (optional)

To trigger Docker builds on every push to `main`, add your environment to the matrix in `.github/workflows/docker-build.yml`:

```yaml
strategy:
  matrix:
    image:
      - name: echo-env
        dockerfile: envs/echo_env/server/Dockerfile
      - name: chat-env
        dockerfile: envs/chat_env/server/Dockerfile
      - name: coding-env
        dockerfile: envs/coding_env/server/Dockerfile
      - name: my-env  # Add your environment here
        dockerfile: envs/my_env/server/Dockerfile
```

### Use Your Environment

For an end-to-end example of using your environment, see the [Quick Start](quickstart.md) guide. Here is a simple example of using your environment:

```python
from envs.my_env import MyAction, MyEnv

# Create environment from Docker image (starts a container)
client = MyEnv.from_docker_image("my-env:latest").sync()
# Or, run the image of a Hugging Face Space locally
client = MyEnv.from_env("my-org/my-env").sync()
# Or, connect to an already running server
client = MyEnv(base_url="http://localhost:8000").sync()

# Use context manager for automatic cleanup (recommended)
with client:
    # Reset
    result = client.reset()
    print(result.observation.result)  # "Ready"

    # Execute actions
    result = client.step(MyAction(command="test", parameters={}))
    print(result.observation.result)
    print(result.observation.success)

    # Get state
    state = client.state()
    print(state.episode_id)
    print(state.step_count)

# Or manually manage the connection
try:
    client = MyEnv(base_url="http://localhost:8000").sync()
    result = client.reset()
    result = client.step(MyAction(command="test", parameters={}))
finally:
    client.close()
```

`from_docker_image()` and `from_env()` do not return a connected client. They
return a lazy bootstrap handle, and nothing starts until you resolve it: chain
`.sync()` for a synchronous client (as above), or `await` the handle from async
code. Using the handle directly in a `with` block raises
`TypeError: '_BootstrapResult' object does not support the context manager protocol`.

The equivalent async usage is:

```python
import asyncio

from envs.my_env import MyAction, MyEnv


async def main():
    client = await MyEnv.from_docker_image("my-env:latest")
    async with client:
        result = await client.reset()
        result = await client.step(MyAction(command="test", parameters={}))
        state = await client.state()


asyncio.run(main())
```

See [Async vs Sync Usage](../guides/async-sync) for when to prefer each style.

## Troubleshooting

### WebSocket Connection Closed During RL Training

**Symptom:** When training with high token generation or long-running operations, you may see:

```
websockets.exceptions.ConnectionClosedError: keepalive ping timeout
```

**Cause:** Uvicorn's default WebSocket ping timeout is 20 seconds. During heavy workloads (e.g., LLM token generation), the client may not respond to pings in time, causing the server to close the connection.

**Solution:** Increase the WebSocket ping interval and timeout in your Dockerfile:

```dockerfile
# Before (default - 20 second timeout)
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]

# After (5 minute timeout)
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000 --ws-ping-interval 300 --ws-ping-timeout 300"]
```

After modifying the Dockerfile, rebuild and redeploy:

```bash
openenv build
openenv push
```

**Available options:**

| Parameter | Description | Default | Recommended for RL |
|-----------|-------------|---------|-------------------|
| `--ws-ping-interval` | Seconds between server pings | 20 | 300 (5 min) |
| `--ws-ping-timeout` | Seconds to wait for pong response | 20 | 300 (5 min) |

!!! tip
    For local development without Docker, pass the same flags directly to uvicorn:
    ```bash
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --ws-ping-interval 300 --ws-ping-timeout 300
    ```

---

## Nice work! You've now built and used your own OpenEnv environment.

Your next steps are to:

- [Try out the end-to-end tutorial](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)
