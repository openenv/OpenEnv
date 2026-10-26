<!-- openenv-source: reasoning_gym_env -->
# Reasoning Gym Environment

An OpenEnv environment that integrates the [Reasoning Gym](https://github.com/open-thought/reasoning-gym) library to provide single-step reasoning tasks. Each episode presents one question from a configurable dataset, the agent submits an answer, and receives a score.

## Quick Start

The simplest way to use the Reasoning Gym environment is through the `ReasoningGymEnv` class:

```python
from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv

try:
    # Create environment from Docker image
    env = ReasoningGymEnv.from_docker_image("reasoning_gym-env:latest").sync()

    # Create a dataset with 10 leg_counting questions
    result = env.reset(
        dataset_name='leg_counting',
        seed=42,
        size=10
    )
    print(f"Question: {result.observation.question}")
    # Question: "How many legs does a cat have?"

    # Submit answer
    result = env.step(ReasoningGymAction(answer="4"))
    print(f"Score: {result.observation.score}")  # 1.0 (correct)
    print(f"Correct answer: {result.observation.correct_answer}")  # "4"
    print(f"Reward: {result.reward}")  # 1.0
    print(f"Done: {result.done}")  # True (single-step episodes)

    # Access dataset metadata if available
    if result.observation.dataset_metadata:
        print(f"Metadata: {result.observation.dataset_metadata}")  # Dataset-specific info

    # Get next question from same dataset
    result = env.reset()  # No params = reuse dataset
    print(f"Next question: {result.observation.question}")

    # Note: First reset() without params creates a default leg_counting dataset
    # with seed=42 and size=1000 for reproducible, out-of-the-box functionality

finally:
    # Always clean up
    env.close()
```

That's it! The `ReasoningGymEnv.from_docker_image()` method handles:

- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the reasoning_gym_env directory
docker build -t reasoning_gym-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

From `envs/reasoning_gym_env/`, run `openenv push` (or `openenv push --repo-id my-org/reasoning-gym-env`). See the [`openenv push` reference](https://huggingface.co/docs/openenv/reference/cli#openenv-push) for all options. The Space serves the web UI at `/web`, the API docs at `/docs`, a health check at `/health` and the WebSocket session endpoint at `/ws`.

## Environment Details

### Episode Structure

Each episode is **single-step**:

1. `reset()` returns a question
2. `step(answer)` returns score and marks episode as done
3. `reset()` without params gets next question from same dataset

### Action

**ReasoningGymAction**: Contains the agent's answer

- `answer` (str) - The agent's answer to the current question

### Observation

**ReasoningGymObservation**: Contains the question or result

- `question` (Optional[str]) - The current question (only in reset)
- `score` (Optional[float]) - Score for the answer, 0.0 to 1.0 (only after step)
- `correct_answer` (Optional[str]) - The correct answer (only after step)
- `dataset_metadata` (Optional[Dict]) - Metadata from the reasoning gym dataset entry
- `done` (bool) - Always True after step (single-step episodes)
- `reward` (float) - Equal to score (0.0 to 1.0)

### Reward

The reward equals the score returned by the dataset's scoring function:

- Correct answer → score: 1.0
- Incorrect answer → score: 0.0 to 1.0 (dataset-dependent, may use partial credit)

## Dataset Configuration

### Simple Datasets

Use a single dataset with configuration:

```python
result = env.reset(
    dataset_name='leg_counting',
    seed=42,
    size=10
)
```

Available datasets from reasoning_gym library:

- `leg_counting` - Count legs of various objects
- `reverse_sort` - Sort lists in reverse order
- `chess_state_eval` - Chess position evaluation
- And more (see [Reasoning Gym documentation](https://github.com/open-thought/reasoning-gym))

### Composite Datasets

Mix multiple datasets with custom weights:

```python
result = env.reset(
    dataset_name='composite',
    dataset_specs=[
        {
            "name": "leg_counting",
            "weight": 3,  # 3x more likely
            "config": {}
        },
        {
            "name": "reverse_sort",
            "weight": 1,
            "config": {"min_length": 3, "max_length": 5}
        }
    ],
    seed=42,
    size=20
)
```

### Dataset Persistence

The dataset persists across resets until configuration changes:

```python
# Create dataset
result = env.reset(dataset_name='leg_counting', seed=42, size=10)
question1 = result.observation.question

# Get next question from SAME dataset
result = env.reset()  # No params = reuse dataset
question2 = result.observation.question

# Create NEW dataset (different seed)
result = env.reset(dataset_name='leg_counting', seed=99, size=10)
question3 = result.observation.question  # From new dataset
```

## Advanced Usage

### Connecting to an Existing Server

If you already have a Reasoning Gym environment server running, you can connect directly:

```python
from reasoning_gym_env import ReasoningGymEnv, ReasoningGymAction

# Connect to existing server
env = ReasoningGymEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset(
    dataset_name='leg_counting',
    dataset_config={"min_animals": 5, "max_animals": 15}
    seed=42,
    size=10
)
result = env.step(ReasoningGymAction(answer="4"))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv

# Connect with context manager (auto-connects and closes)
with ReasoningGymEnv(base_url="http://localhost:8000") as env:
    result = env.reset(
        dataset_name='leg_counting',
        seed=42,
        size=5
    )
    print(f"Question: {result.observation.question}")

    # Multiple steps with low latency
    for i in range(5):
        result = env.step(ReasoningGymAction(answer="4"))
        print(f"Score: {result.observation.score}")
        if result.done:
            result = env.reset()  # Get next question
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server is configured to support multiple concurrent WebSocket connections:

```python
from reasoning_gym_env import ReasoningGymAction, ReasoningGymEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with ReasoningGymEnv(base_url="http://localhost:8000") as env:
        result = env.reset(
            dataset_name='leg_counting',
            seed=42 + client_id,  # Different seed per client
            size=10
        )
        total_score = 0.0
        for i in range(10):
            result = env.step(ReasoningGymAction(answer="4"))
            total_score += result.observation.score
            if result.done:
                result = env.reset()  # Next question
        return client_id, total_score

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
    for client_id, score in results:
        print(f"Client {client_id}: Total score = {score}")
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the reasoning_gym_env directory
python3 server/reasoning_gym_environment.py
```

This verifies that:

- Environment resets correctly
- Step executes actions properly
- Dataset creation and iteration works
- Scoring is calculated correctly

### Running Locally

Run the server locally for development:

```bash
# From the reasoning_gym_env directory
uvicorn server.app:app --reload
```

Then test with:

```python
from reasoning_gym_env import ReasoningGymEnv, ReasoningGymAction

env = ReasoningGymEnv(base_url="http://localhost:8000")
result = env.reset(dataset_name='leg_counting', seed=42, size=5)
print(result.observation.question)
```

## Use Cases

The Reasoning Gym environment is ideal for:

- **LLM Evaluation**: Benchmark reasoning capabilities across diverse tasks
- **Agent Training**: Train RL agents on reasoning tasks with clear score signals
- **Curriculum Learning**: Mix datasets with composite configurations for progressive difficulty
- **Research**: Reproducible environments with seed control for scientific experiments
- **Educational**: Learn about environment design and agent-environment interaction

## Learn More

- [Reasoning Gym Library](https://github.com/open-thought/reasoning-gym)
- [OpenEnv Documentation](https://github.com/huggingface/OpenEnv)
- [OpenEnv Environment Design](https://github.com/huggingface/OpenEnv/blob/main/README.md)