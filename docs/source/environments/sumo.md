<!-- openenv-source: sumo_rl_env -->
# SUMO-RL Environment

Integration of traffic signal control with the OpenEnv framework via SUMO (Simulation of Urban MObility) and SUMO-RL.

## Overview

This environment enables reinforcement learning for **traffic signal control** using SUMO, a microscopic traffic simulation package. Train RL agents to optimize traffic light timing and minimize vehicle delays.

**Key Features**:
- **Realistic traffic simulation** via SUMO
- **Single-agent mode** for single intersection control
- **Configurable rewards** (waiting time, queue, pressure, speed)
- **Multiple networks** supported (custom .net.xml and .rou.xml files)
- **Docker-ready** with pre-bundled example network

## Quick Start

### Using Docker (Recommended)

```python
from envs.sumo_rl_env import SumoRLEnv, SumoAction

# Automatically starts container
env = SumoRLEnv.from_docker_image("sumo-rl-env:latest")

# Reset environment
result = env.reset()
print(f"Observation shape: {result.observation.observation_shape}")
print(f"Available actions: {result.observation.action_mask}")

# Take action (select next green phase)
result = env.step(SumoAction(phase_id=1))
print(f"Reward: {result.reward}, Done: {result.done}")

# Get state
state = env.state()
print(f"Simulation time: {state.sim_time}")
print(f"Total vehicles: {state.total_vehicles}")
print(f"Mean waiting time: {state.mean_waiting_time}")

# Cleanup
env.close()
```

### Building the Docker Image

```bash
cd OpenEnv

# Build base image first (if not already built)
docker build -t envtorch-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build SUMO-RL environment
docker build -f envs/sumo_rl_env/server/Dockerfile -t sumo-rl-env:latest .
```

### Running with Different Configurations

```bash
# Default: single-intersection
docker run -p 8000:8000 sumo-rl-env:latest

# Longer simulation
docker run -p 8000:8000 \
  -e SUMO_NUM_SECONDS=50000 \
  sumo-rl-env:latest

# Different reward function
docker run -p 8000:8000 \
  -e SUMO_REWARD_FN=queue \
  sumo-rl-env:latest

# Custom seed for reproducibility
docker run -p 8000:8000 \
  -e SUMO_SEED=123 \
  sumo-rl-env:latest
```

## Observation

The observation is a vector containing:
- **Phase one-hot**: Current active green phase (one-hot encoded)
- **Min green flag**: Binary indicator if minimum green time has passed
- **Lane densities**: Number of vehicles / lane capacity for each incoming lane
- **Lane queues**: Number of queued vehicles / lane capacity for each incoming lane

Observation size varies by network topology (depends on number of phases and lanes).

**Default (single-intersection)**:
- 4 green phases
- 8 incoming lanes
- Observation size: ~21 elements

## Action Space

The action space is discrete and represents selecting the next green phase to activate.

- **Action type**: Discrete
- **Action range**: `[0, num_green_phases - 1]`
- **Default (single-intersection)**: 4 actions (one per green phase)

When a phase change is requested, SUMO automatically inserts a yellow phase before switching.

## Rewards

Default reward function is **change in cumulative waiting time**:
```
reward = -(total_waiting_time_now - total_waiting_time_previous)
```

Positive rewards indicate waiting time decreased (good).

### Available Reward Functions

Set via `SUMO_REWARD_FN` environment variable:

- **`diff-waiting-time`** (default): Change in cumulative waiting time
- **`average-speed`**: Average speed of all vehicles
- **`queue`**: Negative total queue length
- **`pressure`**: Pressure metric (incoming - outgoing vehicles)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMO_NET_FILE` | `/app/nets/single-intersection.net.xml` | Network topology file |
| `SUMO_ROUTE_FILE` | `/app/nets/single-intersection.rou.xml` | Vehicle routes file |
| `SUMO_NUM_SECONDS` | `20000` | Simulation duration (seconds) |
| `SUMO_DELTA_TIME` | `5` | Seconds between agent actions |
| `SUMO_YELLOW_TIME` | `2` | Yellow phase duration (seconds) |
| `SUMO_MIN_GREEN` | `5` | Minimum green time (seconds) |
| `SUMO_MAX_GREEN` | `50` | Maximum green time (seconds) |
| `SUMO_REWARD_FN` | `diff-waiting-time` | Reward function name |
| `SUMO_SEED` | `42` | Random seed (use for reproducibility) |

### Using Custom Networks

To use your own SUMO network:

```python
from envs.sumo_rl_env import SumoRLEnv

env = SumoRLEnv.from_docker_image(
    "sumo-rl-env:latest",
    volumes={
        "/path/to/your/nets": {"bind": "/nets", "mode": "ro"}
    },
    environment={
        "SUMO_NET_FILE": "/nets/my-network.net.xml",
        "SUMO_ROUTE_FILE": "/nets/my-routes.rou.xml",
    }
)
```

Your network directory should contain:
- `.net.xml` - Network topology (roads, junctions, traffic lights)
- `.rou.xml` - Vehicle routes (trip definitions, flow rates)

## API Reference

### SumoAction

```python
@dataclass
class SumoAction(Action):
    phase_id: int  # Green phase to activate (0 to num_phases-1)
    ts_id: str = "0"  # Traffic signal ID (for multi-agent)
```

### SumoObservation

```python
@dataclass
class SumoObservation(Observation):
    observation: List[float]  # Observation vector
    observation_shape: List[int]  # Shape for reshaping
    action_mask: List[int]  # Valid action indices
    sim_time: float  # Current simulation time
    done: bool  # Episode finished
    reward: Optional[float]  # Reward from last action
    metadata: Dict  # System metrics
```

### SumoState

```python
@dataclass
class SumoState(State):
    episode_id: str  # Unique episode ID
    step_count: int  # Steps taken
    net_file: str  # Network file path
    route_file: str  # Route file path
    sim_time: float  # Current simulation time
    total_vehicles: int  # Total vehicles in simulation
    total_waiting_time: float  # Cumulative waiting time
    mean_waiting_time: float  # Mean waiting time
    mean_speed: float  # Mean vehicle speed
    # ... configuration parameters
```

## Example Training Loop

```python
from envs.sumo_rl_env import SumoRLEnv, SumoAction
import numpy as np

# Start environment
env = SumoRLEnv.from_docker_image("sumo-rl-env:latest")

# Training loop
for episode in range(10):
    result = env.reset()
    episode_reward = 0
    steps = 0

    while not result.done and steps < 1000:
        # Random policy (replace with your RL agent)
        action_id = np.random.choice(result.observation.action_mask)

        # Take action
        result = env.step(SumoAction(phase_id=int(action_id)))

        episode_reward += result.reward or 0
        steps += 1

        # Print progress every 100 steps
        if steps % 100 == 0:
            state = env.state()
            print(f"Step {steps}: "
                  f"reward={result.reward:.2f}, "
                  f"vehicles={state.total_vehicles}, "
                  f"waiting={state.mean_waiting_time:.2f}")

    print(f"Episode {episode}: total_reward={episode_reward:.2f}, steps={steps}")

env.close()
```

## Performance Notes

### Simulation Speed

- **Reset time**: 1-5 seconds (starts new SUMO simulation)
- **Step time**: ~50-200ms per step (depends on network size)
- **Episode duration**: Minutes (20,000 sim seconds with delta_time=5 → ~4,000 steps)

### Optimization

For faster simulation:
1. Reduce `SUMO_NUM_SECONDS` for shorter episodes
2. Increase `SUMO_DELTA_TIME` for fewer decisions
3. Use simpler networks with fewer vehicles

## Architecture

```
┌─────────────────────────────────┐
│ Client: SumoRLEnv               │
│  .step(phase_id=1)              │
└──────────────┬──────────────────┘
               │ HTTP
┌──────────────▼──────────────────┐
│ FastAPI Server (Docker)         │
│   SumoEnvironment               │
│     ├─ Wraps sumo_rl           │
│     ├─ Single-agent mode       │
│     └─ No GUI                  │
└──────────────┬──────────────────┘
               │
┌──────────────▼──────────────────┐
│ SUMO Simulator                  │
│  - Reads .net.xml (network)     │
│  - Reads .rou.xml (routes)      │
│  - Simulates traffic flow       │
│  - Provides observations        │
└─────────────────────────────────┘
```

## Bundled Network

The default `single-intersection` network is a simple 4-way intersection with:
- **4 incoming roads** (North, South, East, West)
- **4 green phases** (NS straight, NS left, EW straight, EW left)
- **Vehicle flow**: Continuous stream with varying rates

## Limitations

- **No GUI in Docker**: SUMO GUI requires X server (not available in containers)
- **Single-agent only**: Multi-agent (multiple intersections) coming in future version
- **Fixed network per container**: Each container uses one network topology
- **Memory usage**: ~500MB for small networks, 2-4GB for large city networks

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs <container-id>

# Verify network files exist
docker run sumo-rl-env:latest ls -la /app/nets/
```

### "SUMO_HOME not set" error
This should be automatic in Docker. If running locally:
```bash
export SUMO_HOME=/usr/share/sumo
```

### Slow performance
- Reduce simulation duration: `SUMO_NUM_SECONDS=5000`
- Increase action interval: `SUMO_DELTA_TIME=10`
- Use smaller networks with fewer vehicles

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [SUMO-RL GitHub](https://github.com/LucasAlegre/sumo-rl)
- [SUMO-RL Paper](https://peerj.com/articles/cs-575/)
- [RESCO Benchmarks](https://github.com/jault/RESCO)

## Citation

If you use SUMO-RL in your research, please cite:

```bibtex
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}
```

## License

This integration is licensed under the BSD-style license. SUMO-RL and SUMO have their own licenses.
