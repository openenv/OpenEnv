<!-- openenv-source: coding_env -->
# Coding Environment

A Python code execution environment that runs arbitrary Python code and returns results. Perfect for testing code execution infrastructure and demonstrating environment usage patterns.

Try it on the [`openenv/coding_env`](https://huggingface.co/spaces/openenv/coding_env) Space, or connect a client to `https://openenv-coding-env.hf.space`.

## Quick Start

The simplest way to use the Coding environment is through the `CodingEnv` class. The client is **async by default**:

```python
import asyncio
from coding_env import CodeAction, CodingEnv

async def main():
    # Create environment from Docker image
    client = await CodingEnv.from_docker_image("coding-env:latest")

    async with client:
        # Reset
        result = await client.reset()
        print(f"Reset complete: exit_code={result.observation.exit_code}")

        # Execute Python code
        code_samples = [
            "print('Hello, World!')",
            "x = 5 + 3\nprint(f'Result: {x}')",
            "import math\nprint(math.pi)"
        ]

        for code in code_samples:
            result = await client.step(CodeAction(code=code))
            print(f"Code: {code}")
            print(f"  → stdout: {result.observation.stdout.strip()}")
            print(f"  → exit_code: {result.observation.exit_code}")

asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
from coding_env import CodeAction, CodingEnv

with CodingEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    result = client.step(CodeAction(code="print('Hello!')"))
    print(result.observation.stdout)
```

The `CodingEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when the context manager exits

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t coding-env:latest -f envs/coding_env/server/Dockerfile .
```

## Environment Details

### Action
**CodeAction**: Contains a single field
- `code` (str) - The Python code to execute

### Observation
**CodeObservation**: Contains the execution results
- `stdout` (str) - Standard output from code execution
- `stderr` (str) - Standard error from code execution
- `exit_code` (int) - Exit code (0 for success, non-zero for errors)

### State
**CodeState**: Tracks execution state
- `episode_id` (str) - Unique identifier for the episode
- `step_count` (int) - Number of steps taken
- `last_exit_code` (int) - Exit code from the last execution

## Advanced Usage

### Connecting to an Existing Server

If you already have a Coding environment server running, you can connect directly:

```python
from coding_env import CodeAction, CodingEnv

# Async usage
async with CodingEnv(base_url="http://localhost:8000") as client:
    result = await client.reset()
    result = await client.step(CodeAction(code="print('Hello!')"))

# Sync usage
with CodingEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    result = client.step(CodeAction(code="print('Hello!')"))
```

Note: When connecting to an existing server, closing the client will NOT stop the server.

## Development & Testing

### Running Tests

Install the coding_env package with dev dependencies and run the tests from the repo root:

```bash
# Install coding_env with dev dependencies (includes smolagents and pytest)
uv pip install -e "envs/coding_env[dev]"

# Run unit tests (no Docker required)
uv run pytest tests/envs/test_python_codeact_reset.py tests/envs/test_python_codeact_rewards.py -v

# Run integration tests (requires Docker image to be built)
docker build -t coding-env:latest -f envs/coding_env/server/Dockerfile .
SKIP_DOCKER_TESTS=0 uv run pytest tests/envs/test_coding_env_integration.py -v
```

### Running the Full Example

Run the complete example that demonstrates the full workflow:

```bash
python3 envs/coding_env/client/example_usage.py
```

This example shows:
- Creating an environment from a Docker image
- Resetting and executing code through the environment
- Automatic cleanup with `close()`

## Project Structure

```
coding_env/
├── README.md              # This file
├── models.py              # Action, Observation, and State models
├── client/
│   ├── coding_env_client.py  # CodingEnv client implementation
│   └── example_usage.py      # Usage examples
└── server/
    ├── python_codeact_env.py  # Core environment logic
    ├── app.py                 # FastAPI application
    ├── transforms.py          # Observation transforms
    ├── Dockerfile             # Container image definition
    └── README.md              # Server-specific documentation
```
