<!-- openenv-source: openspiel_env -->
# OpenSpiel Environment

Integration of OpenSpiel games with the OpenEnv framework. [OpenSpiel](https://github.com/google-deepmind/open_spiel) is DeepMind's collection of 70+ game environments for RL research.

![OpenSpiel web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/openspiel.png)

*The `/web` playground on 2048 after some moves: the board and one button per legal move.*

## Supported Games

This environment supports 6 games across different categories:

![OpenSpiel games in the web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/openspiel-games.png)

*The `/web` playground drawing for the OpenSpiel games, each after a few moves.*

### Single-Player Games (No Opponent)
1. **Catch** - Move horizontally to catch a falling ball
2. **Cliff Walking** - Navigate grid without falling off cliff (Sutton & Barto benchmark)
3. **2048** - Classic tile-merging puzzle game
4. **Blackjack** - Simplified blackjack (HIT/STAND only)

### Multi-Player Games (with Bot Opponent)
5. **Tic-Tac-Toe** - Classic 3x3 game
6. **Kuhn Poker** - 2-player simplified poker (game theory benchmark)

## Quick Start

The simplest way to use the OpenSpiel environment is through the `OpenSpielEnv` class:

```python
from openspiel_env import OpenSpielEnv, OpenSpielAction

try:
    # Create environment from Docker image
    env = OpenSpielEnv.from_docker_image("openspiel-env:latest").sync()

    # Reset to start a new episode
    result = env.reset()
    print(f"Initial state: {result.observation.info_state}")
    print(f"Legal actions: {result.observation.legal_actions}")

    # Play until done
    while not result.done:
        action_id = result.observation.legal_actions[0]
        result = env.step(OpenSpielAction(action_id=action_id))
        print(f"Reward: {result.reward}, Done: {result.done}")

finally:
    # Always clean up
    env.close()
```

That's it! The `OpenSpielEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

OpenSpiel requires compilation from C++ source. The Docker build uses a **pre-built base image** by default to avoid long build times.

### Default Build (Recommended)

From the **environment directory** (`envs/openspiel_env/`):

```bash
# Uses pre-built base image from GHCR (fast, ~1-2 min)
docker build -t openspiel-env:latest -f server/Dockerfile .
```

This uses the pre-built `ghcr.io/huggingface/openenv-openspiel-base` image which already contains compiled OpenSpiel.

### Building Your Own Base Image (Optional)

If you need to customize OpenSpiel or can't access the pre-built image:

```bash
# Step 1: Build the base image (compiles OpenSpiel, ~30-60 min)
docker build -t openspiel-base:latest -f server/Dockerfile.openspiel-base .

# Step 2: Build the environment using your local base image
docker build -t openspiel-env:latest \
  --build-arg OPENSPIEL_BASE_IMAGE=openspiel-base:latest \
  -f server/Dockerfile .
```

## Running Specific Games

Pick the game and opponent with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENSPIEL_GAME` | `catch` | `catch`, `tic_tac_toe`, `kuhn_poker`, `2048`, `blackjack` or `cliff_walking` |
| `OPENSPIEL_AGENT_PLAYER` | `0` | Player ID the agent plays as |
| `OPENSPIEL_OPPONENT_POLICY` | `random` | Opponent in multi-player games: `random`, `first` (first legal action) or `last` (last legal action) |

```bash
docker run -p 8000:8000 \
  -e OPENSPIEL_GAME=tic_tac_toe \
  -e OPENSPIEL_OPPONENT_POLICY=first \
  openspiel-env:latest
```

## Deploying to Hugging Face Spaces

From `envs/openspiel_env/`:

```bash
openenv push --repo-id my-org/openspiel-env -e OPENSPIEL_GAME=tic_tac_toe
```

`-e` sets the variables above on the Space. See the [`openenv push` reference](https://huggingface.co/docs/openenv/reference/cli#openenv-push) for all options. The Space serves the web UI at `/web`, the API docs at `/docs` and a health check at `/health`. The default Dockerfile uses the pre-built OpenSpiel base image, so it runs on standard CPU hardware.

## Environment Details

### Action
**OpenSpielAction**: Contains the action to take
- `action_id` (int) - Action ID to execute
- `game_name` (str) - Game name (default: "catch")
- `game_params` (Dict) - Optional game parameters

### Observation
**OpenSpielObservation**: Contains the game state
- `info_state` (List[float]) - Agent's information state vector
- `legal_actions` (List[int]) - Legal action IDs
- `game_phase` (str) - "initial", "playing", or "terminal"
- `current_player_id` (int) - Current player (-1 for simultaneous)
- `opponent_last_action` (Optional[int]) - Last opponent action
- `done` (bool) - Whether the episode has ended
- `reward` (Optional[float]) - Reward for the last action

### State
**OpenSpielState**: Server-side state snapshot
- `episode_id` (str) - Unique identifier for the current episode
- `step_count` (int) - Number of steps taken
- `game_name` (str) - Game name
- `agent_player` (int) - Agent's player ID
- `opponent_policy` (str) - Opponent policy name
- `num_players` (int) - Total players

## Advanced Usage

### Connecting to an Existing Server

If you already have an OpenSpiel environment server running:

```python
from openspiel_env import OpenSpielEnv, OpenSpielAction

# Connect to existing server
env = OpenSpielEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
result = env.step(OpenSpielAction(action_id=result.observation.legal_actions[0]))

# Close connection (does NOT stop the server)
env.close()
```

### Connecting to HuggingFace Space

```python
from openspiel_env import OpenSpielEnv, OpenSpielAction

# Connect to remote Space
env = OpenSpielEnv(base_url="https://your-username-openspiel.hf.space")

result = env.reset()
print(f"Game: {result.observation.game_phase}")
print(f"Legal actions: {result.observation.legal_actions}")

result = env.step(OpenSpielAction(action_id=result.observation.legal_actions[0]))
env.close()
```

## Game-Specific Information

### 1. Catch
- **Type**: Single-player
- **Action Space**: 3 actions (left, stay, right)
- **Observation**: 5x5 grid flattened (25 dimensions)
- **Reward**: +1 for catching ball, 0 otherwise
- **Episode Length**: ~10 steps

### 2. Tic-Tac-Toe
- **Type**: 2-player turn-based, perfect information
- **Players**: Agent (X) vs Random Bot (O)
- **Action Space**: 9 positions
- **Observation**: 27 dimensions (3x3 board + game state)
- **Reward**: +1 win, -1 loss, 0 draw/mid-game

### 3. Kuhn Poker
- **Type**: 2-player turn-based, imperfect information
- **Players**: Agent vs Random Bot
- **Action Space**: 2 actions (pass/fold, bet/call)
- **Observation**: 6 dimensions (card + betting history)
- **Reward**: Pot winnings (typically -1, 0, +1, +2)
- **Notes**: THE benchmark for imperfect-information RL

### 4. Cliff Walking
- **Type**: Single-player grid world
- **Action Space**: 4 actions (up, down, left, right)
- **Observation**: Position encoding
- **Reward**: -1 per step, -100 for falling off cliff
- **Notes**: Classic RL benchmark from Sutton & Barto

### 5. 2048
- **Type**: Single-player puzzle
- **Action Space**: 4 actions (up, down, left, right)
- **Observation**: 4x4 grid with tile values
- **Reward**: Points from merging tiles
- **Notes**: Stochastic tile spawning

### 6. Blackjack
- **Type**: Single-player vs dealer
- **Action Space**: 2 actions (HIT, STAND)
- **Observation**: Player hand + dealer's visible card
- **Reward**: +1 win, -1 loss, 0 draw
- **Notes**: Simplified version, no double/split

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server (requires OpenSpiel installed locally):

```python
from openspiel_env.server.openspiel_environment import OpenSpielEnvironment
from openspiel_env.models import OpenSpielAction

# Create environment directly
env = OpenSpielEnvironment(game_name="catch")

# Test reset
obs = env.reset()
print(f"Info state: {obs.info_state}")

# Test step
obs = env.step(OpenSpielAction(action_id=0))
print(f"Done: {obs.done}, Reward: {obs.reward}")
```

### Running Locally

Run the server locally for development (requires OpenSpiel installed):

```bash
# From the environment directory
cd envs/openspiel_env

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

### Automated Testing (All 6 Games)

```bash
./test_docker_all_games.sh
```

This script will build and test all 6 supported games in Docker.

## Limitations

- **Simultaneous-move games**: Only agent_player=0 supported
- **Multi-agent training**: Single agent only (no self-play yet)
- **Opponent policies**: Random and fixed only (no MCTS yet)
- **Build time**: Building your own base image takes ~30-60 min (compiles OpenSpiel C++). Using the pre-built image is fast (~1-2 min) and works with standard hardware.

## References

- [OpenSpiel Paper (2019)](https://arxiv.org/abs/1908.09453)
- [OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)
- [OpenSpiel Documentation](https://openspiel.readthedocs.io/)
