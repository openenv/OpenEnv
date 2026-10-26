---
title: Echo Environment Server
emoji: 🔊
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Echo Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

Try it on the [`openenv/echo_env`](https://huggingface.co/spaces/openenv/echo_env) Space, or connect a client to `https://openenv-echo-env.hf.space`.

## Quick Start

The simplest way to use the Echo environment is through the `EchoEnv` class. The client is **async by default**:

```python
import asyncio
from echo_env import CallToolAction, EchoEnv

async def main():
    # Create environment from Docker image
    client = await EchoEnv.from_docker_image("echo-env:latest")

    async with client:
        await client.reset()

        # Send multiple messages
        messages = ["Hello, World!", "Testing echo", "Final message"]

        for msg in messages:
            result = await client.step(
                CallToolAction(
                    tool_name="echo_message",
                    arguments={"message": msg},
                )
            )
            print(f"Sent: '{msg}'")
            print(f"  → Echoed: '{result.observation.result}'")
            print(f"  → Reward: {result.reward}")

asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
from echo_env import CallToolAction, EchoEnv

with EchoEnv(base_url="http://localhost:8000").sync() as client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello!"},
        )
    )
    print(result.observation.result)
```

The `EchoEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when the context manager exits

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .
```

## Environment Details

### Tools
- `echo_message(message)` - Echo the provided message
- `echo_with_length(message)` - Echo the message and include its length

### Observation
**CallToolObservation**: Contains the tool result and metadata
- `result` - The tool return value
- `reward` (float) - Reward returned by the environment
- `done` (bool) - Always `False` for the echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The echo environment returns `0.0` reward for tool calls. It is intended as a
minimal MCP integration example rather than a reward-shaping reference.

## Advanced Usage

### Connecting to an Existing Server

If you already have an Echo environment server running, you can connect directly:

```python
from echo_env import CallToolAction, EchoEnv

# Async usage
async with EchoEnv(base_url="http://localhost:8000") as client:
    await client.reset()
    result = await client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello!"},
        )
    )

# Sync usage
with EchoEnv(base_url="http://localhost:8000").sync() as client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello!"},
        )
    )
```

Note: When connecting to an existing server, closing the client will NOT stop the server.

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 envs/echo_env/server/test_echo_env.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running the Full Example

Run the complete example that demonstrates the full workflow:

```bash
python3 examples/local_echo_env.py
```

This example shows:
- Creating an environment from a Docker image
- Resetting and stepping through the environment
- Automatic cleanup with `close()`

## Project Structure

```
echo_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── client.py              # EchoEnv client implementation
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── echo_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    ├── test_echo_env.py   # Direct environment tests
    └── Dockerfile         # Container image definition
```
