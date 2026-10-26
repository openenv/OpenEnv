<!-- openenv-source: wildfire_env -->
# Wildfire Environment

A wildfire-control simulation for reinforcement learning. The agent contains spreading fires with **water** and **firebreaks** under **wind** and **humidity**, with limited resources. The spread model is inspired by the Rothermel surface fire spread model and MITRE Fireline's SimFire.

![Wildfire web playground](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/wildfire.png)

*The `/web` playground after a few steps: the spreading fire, a line of firebreaks and the episode stats.*

## Quick Start

Build and run the server from `envs/wildfire_env/`:

```bash
docker build -t wildfire-env:latest -f server/Dockerfile .
docker run -p 8000:8000 wildfire-env:latest
```

The image enables the web interface at `http://localhost:8000/web`.

```python
from wildfire_env import WildfireAction, WildfireEnv

with WildfireEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    obs = result.observation
    print(f"Grid: {obs.width}x{obs.height}, Fires: {obs.burning_count}, Water: {obs.remaining_water}")

    result = env.step(WildfireAction(action="water", x=10, y=15))  # water a cell
    result = env.step(WildfireAction(action="break", x=12, y=15))  # build a firebreak
    result = env.step(WildfireAction(action="wait"))               # let the fire evolve
    print(f"Reward: {result.reward:.2f}, Burning: {result.observation.burning_count}")
```

To run without Docker, `pip install -e envs/wildfire_env` and start `server` (or `python -m wildfire_env.server.app`).

## Grid

The observation's `grid` is a flat list of `width * height` cells. Cell `(x, y)` is `grid[y * width + x]`:

```python
import numpy as np
grid_2d = np.array(obs.grid).reshape(obs.height, obs.width)  # grid_2d[y][x]
```

| Code | Cell | Behavior |
|------|------|----------|
| `0` | Ash | Burned out, can't reignite |
| `1` | Fuel | Can ignite |
| `2` | Burning | Spreads to neighbors, turns to ash after 3 ticks |
| `3` | Firebreak | Fire can't cross it |
| `4` | Water/damp | Can't ignite, reverts to fuel after 6 ticks |

## Actions

**WildfireAction**: `action` (`"water"`, `"break"` or `"wait"`), plus `x` and `y` for water and break.

- `water` uses 1 water unit. It extinguishes a burning cell or dampens fuel (both become `4`).
- `break` uses 1 firebreak unit and turns the cell into a firebreak (`3`).
- `wait` does nothing, and the fire keeps spreading.

## Observation

**WildfireObservation**:

- `grid`, `width`, `height`: the grid (see above)
- `step`: step number (0 after reset)
- `wind_dir`: `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW` or `CALM`
- `humidity`: 0.0 to 1.0, higher means less spread
- `burning_count`, `burned_count`: cells on fire and ash cells
- `remaining_water`, `remaining_breaks`: resources left
- `reward_hint`: same value as the step reward
- `reward`, `done`

`env.state()` returns a `WildfireState` with `episode_id`, `step_count`, `total_burned`, `total_extinguished`, `last_action` and the full grid and timers.

The episode ends when no cell is burning or after `max_steps` steps (default 128).

## Reward

Each step sums the action reward, the fire dynamics and a time penalty:

| Event | Reward |
|-------|--------|
| Water a burning cell | +0.25 |
| Water a fuel cell | -0.10 |
| Water a damp, ash or firebreak cell | -0.05 |
| Firebreak on fuel or a damp cell | +0.15 |
| Firebreak on a burning cell | -0.02 |
| Firebreak on ash | -0.02 |
| Firebreak on a firebreak | -0.01 |
| Invalid action (unknown, out of bounds, missing coordinates, no resources left) | -0.05 |
| Fire spread: each extra burning cell after the fire update | -0.15 |
| Fire shrink: each burning cell fewer after the fire update | +0.10 |
| Each new ash cell | -0.05 |
| Every step | -0.01 |

When the episode ends, it adds `0.2 * (1 - burned_ratio)`, plus `0.5 + 0.5 * saved_ratio` if the fire is out.

## Fire Spread

Each burning cell can ignite its 8 neighbors. The ignition probability is `0.30 * (1 - humidity)`, times 2.0 downwind, 0.5 upwind and 1.0 across the wind, and times 0.6 for diagonal neighbors. Humidity varies by ±0.05 around the configured value at each reset.

## Configuration

Set these environment variables before starting the server:

| Variable | Default | Description |
|----------|---------|-------------|
| `WILDFIRE_WIDTH` | `16` | Grid width |
| `WILDFIRE_HEIGHT` | `16` | Grid height |
| `WILDFIRE_HUMIDITY` | `0.25` | Base humidity |
| `WILDFIRE_WIND` | random | Fixed wind direction (`N` ... `NW`, `CALM`) |
| `MAX_CONCURRENT_ENVS` | `8` | Maximum concurrent WebSocket sessions |
| `ENABLE_WEB_INTERFACE` | `true` in the Docker image | Serve the web interface at `/web` |

```bash
docker run -p 8000:8000 -e WILDFIRE_WIDTH=32 -e WILDFIRE_HEIGHT=32 -e WILDFIRE_WIND=N wildfire-env:latest
```

Other settings (`init_sources=2` initial fires, `max_steps=128`, `water_capacity=8`, `break_capacity=50`, `seed=3407`) are constructor arguments of `WildfireEnvironment` in `server/wildfire_environment.py`.

## Web Interface

The web interface shows the grid as colored cells. Click a cell to fill in the coordinates, pick an action and run it. It also shows the remaining resources, wind, humidity and an action log.

## References

- [Rothermel surface fire spread model (USDA Forest Service)](https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf)
- [SimFire (MITRE Fireline)](https://github.com/mitrefireline/simfire)
- [Reinforcement Learning for Wildfire Mitigation in Simulated Disaster Environments](https://huggingface.co/papers/2311.15925)

## Citation

```bibtex
@techreport{rothermel2022surface,
  title     = {The Rothermel Surface Fire Spread Model and Associated Developments},
  author    = {Andrews, Patricia L. and Rothermel, Richard C.},
  year      = {2022},
  institution = {USDA Forest Service},
  number    = {RMRS-GTR-371},
  url       = {https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf}
}

@article{tapley2023reinforcement,
  title   = {Reinforcement Learning for Wildfire Mitigation in Simulated Disaster Environments},
  author  = {Tapley, A. and Dotter, M. and Doyle, M. and others},
  journal = {arXiv preprint arXiv:2311.15925},
  year    = {2023},
  url     = {https://arxiv.org/abs/2311.15925}
}

@misc{mitrefireline2023simfire,
  author = {{MITRE Fireline Project}},
  title  = {SimFire: Wildfire Simulator for Decision-Support and AI Research},
  year   = {2023},
  howpublished = {\url{https://github.com/mitrefireline/simfire}}
}

@misc{wildfire-openenv-2025,
  title  = {Wildfire Environment for OpenEnv: Containment-Focused RL Simulation},
  author = {OpenEnv Contributors},
  year   = {2025},
  url    = {https://github.com/huggingface/OpenEnv}
}
```
