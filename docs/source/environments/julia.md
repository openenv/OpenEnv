<!-- openenv-source: julia_env -->
# Julia Environment

A Julia code execution environment that runs Julia code with test result tracking and reward calculation. Perfect for reinforcement learning training with Julia programming tasks.

## Quick Start

The simplest way to use the Julia environment is through the `JuliaEnv` class:

```python
from envs.julia_env import JuliaAction, JuliaEnv

try:
    # Create environment from Docker image
    julia_env = JuliaEnv.from_docker_image("julia-env:latest")

    # Reset
    result = julia_env.reset()
    print(f"Reset complete: exit_code={result.observation.exit_code}")

    # Execute Julia code with tests
    action = JuliaAction(
        core_code="""
        function multiply(a, b)
            return a * b
        end
        """,
        test_code="""
        using Test
        @test multiply(3, 4) == 12
        @test multiply(5, 6) == 30
        """
    )

    result = julia_env.step(action)
    print(f"Tests passed: {result.observation.tests_passed}")
    print(f"Tests failed: {result.observation.tests_failed}")
    print(f"Code compiles: {result.observation.code_compiles}")
    print(f"Reward: {result.reward}")

finally:
    # Always clean up
    julia_env.close()
```

That's it! The `JuliaEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the julia_env directory
cd envs/julia_env
docker build -t julia-env:latest -f server/Dockerfile .
```

## Environment Details

### Action

**JuliaAction**: Contains two fields for Julia code execution
- `core_code` (str) - The main Julia code to execute (e.g., function definitions)
- `test_code` (str) - Test code using Julia's `Test` module (e.g., `@test` statements)

### Observation

**JuliaObservation**: Contains the execution results and test outcomes
- `stdout` (str) - Standard output from Julia execution
- `stderr` (str) - Standard error from Julia execution
- `exit_code` (int) - Exit code (0 for success, non-zero for errors)
- `tests_passed` (int) - Number of tests that passed
- `tests_failed` (int) - Number of tests that failed
- `code_compiles` (bool) - Whether the core code compiled/executed successfully

### State

**JuliaState**: Tracks episode execution state
- `episode_id` (str) - Unique identifier for the episode
- `step_count` (int) - Number of steps taken in the episode
- `last_exit_code` (int) - Exit code from the last execution
- `last_code_compiles` (bool) - Whether the last code compiled successfully
- `total_tests_passed` (int) - Cumulative number of tests passed in the episode
- `total_tests_failed` (int) - Cumulative number of tests failed in the episode

### Reward Calculation

The environment calculates rewards based on execution success and test results:
- Code compiles successfully: Base reward
- Tests pass: Additional reward per test
- Tests fail or code doesn't compile: Negative reward

See `server/julia_transforms.py` for detailed reward logic.

## Features

- ✅ Execute Julia code in isolated subprocess
- ✅ Parse Julia `Test` module output (tests passed/failed)
- ✅ Calculate rewards based on execution results and test outcomes
- ✅ Safety transforms for output truncation (prevents excessive output)
- ✅ Docker support for reproducible execution
- ✅ Compatible with GRPO and other RL training frameworks

## Advanced Usage

### Connecting to an Existing Server

If you already have a Julia environment server running, you can connect directly:

```python
from envs.julia_env import JuliaEnv, JuliaAction

# Connect to existing server
julia_env = JuliaEnv(base_url="http://localhost:8000")

# Use as normal
result = julia_env.reset()
result = julia_env.step(JuliaAction(
    core_code="println(2 + 2)",
    test_code=""
))
```

Note: When connecting to an existing server, `julia_env.close()` will NOT stop the server.

### Custom Timeout

The Julia environment uses a longer timeout (180s) by default to accommodate Julia compilation and execution:

```python
# Custom timeout (in seconds)
julia_env = JuliaEnv(base_url="http://localhost:8000", message_timeout_s=300.0)
```

### Running with Docker Directly

```bash
# Run with default settings (port 8000)
docker run -d -p 8000:8000 --name julia-env julia-env:latest

# Check health
curl http://localhost:8000/health

# View logs
docker logs -f julia-env
```

## Example Code

Here's a more complex example demonstrating test-driven development:

```python
from envs.julia_env import JuliaAction, JuliaEnv

julia_env = JuliaEnv.from_docker_image("julia-env:latest")

try:
    # Reset the environment
    julia_env.reset()

    # Step 1: Define a function with tests
    action = JuliaAction(
        core_code="""
        function fibonacci(n)
            if n <= 1
                return n
            end
            return fibonacci(n-1) + fibonacci(n-2)
        end
        """,
        test_code="""
        using Test
        @test fibonacci(0) == 0
        @test fibonacci(1) == 1
        @test fibonacci(5) == 5
        @test fibonacci(10) == 55
        """
    )

    result = julia_env.step(action)
    print(f"Step 1 - Tests passed: {result.observation.tests_passed}/4")
    print(f"Step 1 - Reward: {result.reward}")

    # Get current state
    state = julia_env.state()
    print(f"Total tests passed so far: {state.total_tests_passed}")
    print(f"Total tests failed so far: {state.total_tests_failed}")

finally:
    julia_env.close()
```

## Compatibility

This environment is compatible with:
- **torchforge**: For GRPO training with Julia tasks
- **TRL**: Hugging Face's RL library
- **Any OpenEnv-compatible RL framework**

## Development

For local development without Docker:

```bash
# Install Julia 1.10+ from https://julialang.org/downloads/

# Install Python dependencies
cd envs/julia_env
pip install -e .

# Run the server locally
uvicorn julia_env.server.app:app --host 0.0.0.0 --port 8000
```

## License

This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
