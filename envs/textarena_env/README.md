---
title: TextArena Environment Server
emoji: 🎮
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# TextArena Environment

An OpenEnv wrapper for [TextArena](https://github.com/textarena/textarena) game environments. Supports text-based games like Wordle, providing a standardized API for agent interaction.

![TextArena web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/textarena.png)

*The `/web` playground after three Wordle guesses: each tile shows whether the letter is correct, misplaced or absent.*

> [!NOTE]
> Generic wrapper for any [TextArena](https://www.textarena.ai/docs/overview) game inside OpenEnv. This module exposes the TextArena `Env` interface through the standard HTTP server/client APIs used by other OpenEnv environments, enabling quick experimentation with the full suite of word, reasoning, and multi-agent games.

Try Wordle on the [`openenv/wordle`](https://huggingface.co/spaces/openenv/wordle) Space, or connect a client to `https://openenv-wordle.hf.space`.

## Quick Start

The simplest way to use the TextArena environment is through the `TextArenaEnv` class:

```python
from textarena_env import TextArenaAction, TextArenaEnv

try:
    # Create environment from Docker image
    env = TextArenaEnv.from_docker_image("textarena-env:latest")

    # Reset to start a new episode
    result = env.reset()
    print(f"Game prompt:\n{result.observation.prompt}")

    # Play a few turns (example: Wordle guesses)
    guesses = ["[crane]", "[slate]", "[audio]"]

    for guess in guesses:
        result = env.step(TextArenaAction(message=guess))

        # Check messages for feedback
        for message in result.observation.messages:
            print(f"Response: {message.content}")

        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")

        if result.done:
            break

finally:
    # Always clean up
    env.close()
```

That's it! The `TextArenaEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the textarena_env directory
cd envs/textarena_env
docker build -t textarena-env:latest -f server/Dockerfile .
```

## Testing the Gradio UI locally

With the web interface enabled, the server serves a **Gradio UI** at `/web`. If your `openenv` supports `gradio_builder`, you get two tabs (see [Customizing the Web UI](https://huggingface.co/docs/openenv/guides/customizing-web-ui)):

- **Playground** – default OpenEnv UI (Reset, Step, Episode, Quick Start, README).
- **Custom** – Wordle-style HTML block (see `server/gradio_ui.py`; uses [Gradio 6 `gr.HTML`](https://gradio.app/docs/gradio/html) to render the block).

**Option A – From the OpenEnv repo root (recommended for the Custom tab)**

Use the core that includes the tabbed interface and custom builder:

```bash
cd envs/textarena_env
ENABLE_WEB_INTERFACE=true PYTHONPATH=../../src uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Option B – From the environment directory only**

```bash
cd envs/textarena_env
ENABLE_WEB_INTERFACE=true uv run server
```

Or:

```bash
ENABLE_WEB_INTERFACE=true uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/web**. Use the **Playground** tab to Reset and Step with guesses (e.g. `[crane]`, `[stone]`). If you ran with Option A, the **Custom** tab shows the Wordle-style demo block.

## Deploying to Hugging Face Spaces

From `envs/textarena_env/`:

```bash
openenv push --repo-id my-org/textarena-env -e TEXTARENA_ENV_ID=Wordle-v0
```

`-e` sets the [configuration variables](#environment-configuration) on the Space. See the [`openenv push` reference](https://huggingface.co/docs/openenv/reference/cli#openenv-push) for all options. The Space serves the web UI at `/web`, the API docs at `/docs` and a health check at `/health`.

## Environment Details

### Action

**TextArenaAction**: Contains a single field
- `message` (str) - The message/action to send to the game

### Observation

**TextArenaObservation**: Contains the game state and response

- `prompt` (str) - Game instructions and context
- `messages` (List[TextArenaMessage]) - Conversation history with the game
- `current_player_id` (int) - ID of the current player
- `legal_players` (List[int]) - List of valid player IDs
- `info` (Dict) - Additional game metadata
- `reward` (float) - Reward for the current step (inherited from Observation)
- `done` (bool) - Whether the episode has ended (inherited from Observation)

### TextArenaMessage

Each message in the conversation has:

- `sender_id` (int) - ID of the message sender
- `content` (str) - The message content
- `category` (str) - Message type (e.g., "PROMPT", "MESSAGE")

### State

**TextArenaState**: Server-side state snapshot

- `episode_id` (str) - Unique identifier for the current episode
- `step_count` (int) - Number of steps taken in the current episode
- `env_id` (str) - The TextArena environment ID (e.g., "Wordle-v0")
- `num_players` (int) - Number of players in the game
- `max_turns` (Optional[int]) - Maximum turns allowed
- `turn` (int) - Current turn number
- `last_reward` (float) - Most recent reward
- `last_info` (Dict) - Most recent info dictionary
- `raw_state` (Dict) - Raw TextArena state snapshot

### Reward

Rewards are determined by the underlying TextArena game. For example:
- **Wordle-v0**: Positive reward for winning, includes reward signals for green/yellow letter matches

## Advanced Usage

### Connecting to an Existing Server

If you already have a TextArena environment server running, you can connect directly:

```python
from textarena_env import TextArenaEnv, TextArenaAction

# Connect to existing server
env = TextArenaEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = env.reset()
result = env.step(TextArenaAction(message="[crane]"))

# Close connection (does NOT stop the server)
env.close()
```

### Environment Configuration

The server supports configuration via environment variables:

- `TEXTARENA_ENV_ID` - Game to load (default: "Wordle-v0")
- `TEXTARENA_NUM_PLAYERS` - Number of players (default: 1)
- `TEXTARENA_MAX_TURNS` - Maximum turns per episode
- `TEXTARENA_DOWNLOAD_NLTK` - Download NLTK data (default: "1")
- `TEXTARENA_KW_*` - Pass additional kwargs to TextArena (e.g., `TEXTARENA_KW_difficulty=hard`)

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```python
from textarena_env.server.environment import TextArenaEnvironment
from textarena_env.models import TextArenaAction

# Create environment directly
env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

# Test reset
obs = env.reset()
print(f"Prompt: {obs.prompt}")

# Test step
obs = env.step(TextArenaAction(message="[crane]"))
print(f"Done: {obs.done}, Reward: {obs.reward}")
```

### Running Locally

Run the server locally for development:

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e .

# Start the server
python -m uvicorn server.app:app --reload
```

Or using the CLI entry point:

```bash
uv run --project . server --port 8000
```

## Project Structure

```
textarena_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TextArenaEnv client implementation
├── models.py              # Action, Observation, and State models
├── rewards.py             # Reward provider utilities
└── server/
    ├── __init__.py        # Server module exports
    ├── environment.py     # Core TextArenaEnvironment implementation
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```
