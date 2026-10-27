# CARLA LLM Inference Examples

Run LLMs on CARLA autonomous driving scenarios from [sinatras/carla-env](https://blog.sinatras.dev/Carla-Env).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export HF_TOKEN="your-hf-token"
```

## Usage

### Trolley Problems

```bash
# Run single scenario
python trolley_problems.py --model claude-sonnet-4.5 --scenario footbridge

# Save camera images
python trolley_problems.py --model claude-sonnet-4.5 --scenario equal-1v1 --save-images

# Use Hugging Face Space
python trolley_problems.py --model gpt-5.2 --scenario saves-3v0 \
  --base-url https://sergiopaniego-carla-env.hf.space
```

### Maze Navigation

```bash
# Run navigation
python maze_navigation.py --model gpt-5.2

# Save images every 5 steps
python maze_navigation.py --model gpt-5.2 --save-images --image-interval 5
```

### Free-Roam Driving

Navigate to a random goal in open traffic with configurable maps, NPC vehicles, and pedestrians.

```bash
# Default (current map, no traffic)
python free_roam_navigation.py --model gpt-5.2

# With traffic (5 vehicles + 3 pedestrians)
python free_roam_navigation.py --model claude-sonnet-4.5 --scenario free-roam-traffic

# Heavy traffic (15 vehicles + 10 pedestrians)
python free_roam_navigation.py --model gpt-5.2 --scenario free-roam-heavy

# Run all variants
python free_roam_navigation.py --model gpt-5.2 --run-all

# Use Hugging Face Space
python free_roam_navigation.py --model gpt-5.2 \
  --base-url https://sergiopaniego-carla-env.hf.space
```

### Autopilot Navigation (No LLM)

Use CARLA's built-in navigation agents as a baseline — no API keys needed.

```bash
# Maze with default (normal) behavior
python autopilot_navigation.py --scenario maze-1

# Free-roam with cautious driving
python autopilot_navigation.py --scenario free-roam-default --behavior cautious

# Aggressive driving, save images
python autopilot_navigation.py --scenario free-roam-traffic \
  --behavior aggressive --save-images

# Use Hugging Face Space
python autopilot_navigation.py --scenario maze-1 \
  --base-url https://sergiopaniego-carla-env.hf.space
```

**Behaviors:** `cautious` (slow, safe), `normal` (balanced), `aggressive` (fast, overtakes)

This is useful as a baseline to compare against LLM/VLM-driven navigation on the same scenarios.

### Fine-Tuned Model Inference

Run inference with a model fine-tuned via GRPO on the trolley escape scenario (e.g., using [TRL's CARLA example](https://github.com/huggingface/trl/blob/main/examples/grpo_carla/grpo_carla.py)). Uses the same prompt, tools, and message format from training to ensure consistent behavior.

```bash
# Run with a fine-tuned model from the Hub
python carla_escape_inference.py \
  --model sergiopaniego/Qwen3-0.6B-carla-trolley-escape \
  --base-url https://sergiopaniego-carla-env.hf.space

# Save camera images at each turn
python carla_escape_inference.py \
  --model sergiopaniego/Qwen3-0.6B-carla-trolley-escape \
  --base-url https://sergiopaniego-carla-env.hf.space \
  --save-images

# Run multiple episodes
python carla_escape_inference.py \
  --model sergiopaniego/Qwen3-0.6B-carla-trolley-escape \
  --base-url https://sergiopaniego-carla-env.hf.space \
  --episodes 5
```

No API keys needed — the model is loaded locally with `transformers`.

## Vision Mode

Add `--vision` to send camera images to the LLM alongside text prompts. This lets
vision-capable models (VLMs) see the road and make better driving decisions.

```bash
# Maze navigation with vision
python maze_navigation.py --model gpt-5.2 --vision

# Send image every 3 steps instead of every step
python maze_navigation.py --model claude-sonnet-4.5 --vision --vision-interval 3

# Free-roam with vision
python free_roam_navigation.py --model gpt-5.2 --vision

# Trolley problem with vision
python trolley_problems.py --model claude-sonnet-4.5 --scenario footbridge --vision
```

Without `--vision`, behavior is identical to before (text-only prompts).

**Vision-capable models:** `claude-sonnet-4.5`, `claude-sonnet-4`, `gpt-4.1-mini`, `gpt-5.2`

Text-only models (Hugging Face Llama, Mixtral, etc.) will print a warning and fall back
to text-only mode if `--vision` is used.

## Available Models

**Proprietary**: `claude-sonnet-4.5`, `claude-sonnet-4`, `gpt-4.1-mini`, `gpt-5.2`, `qwen3-max`

**Open (HF)**: `qwen2.5-72b`, `llama-3.3-70b`, `llama-3.1-70b`, `mixtral-8x7b`

## Custom Configuration

Override any scenario parameter at reset time using `scenario_config`, without
creating a new scenario name:

```bash
# Override weather and max steps via CLI flags
python free_roam_navigation.py --model gpt-5.2 --weather HardRainNoon --max-steps-scenario 100

# Override route distances
python free_roam_navigation.py --model gpt-5.2 --route-min 50 --route-max 200
```

Or programmatically:

```python
from carla_env import CarlaEnv

env = CarlaEnv(base_url="http://localhost:8000")

# Dict-based: weather, traffic, max_steps, route distances, etc.
result = env.reset(scenario_config={
    "weather": "HardRainNoon",
    "num_npc_vehicles": 5,
    "num_pedestrians": 3,
    "max_steps": 100,
    "route_distance_min": 50.0,
    "route_distance_max": 200.0,
})

# Combine scenario name with overrides
result = env.reset(
    scenario_name="free_roam",
    scenario_config={"weather": "ClearSunset", "num_npc_vehicles": 15},
)
```

**FreeRoamConfig parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `map_name` | str \| None | None | CARLA map (only Town10HD_Opt and Mine_01 available) |
| `num_npc_vehicles` | int | 0 | NPC vehicle count |
| `num_pedestrians` | int | 0 | Pedestrian count |
| `weather` | str | ClearNoon | Weather preset name |
| `max_steps` | int | 500 | Episode step limit |
| `success_radius` | float | 10.0 | Goal proximity threshold (m) |
| `route_distance_min` | float | 100.0 | Min random goal distance (m) |
| `route_distance_max` | float | 500.0 | Max random goal distance (m) |
| `random_goal` | bool | True | Generate random goal each episode |
| `goal_location` | tuple \| None | None | Fixed goal (x, y, z) |
| `initial_speed_kmh` | float | 0.0 | Starting speed |
| `vehicle_blueprint` | str | vehicle.tesla.model3 | CARLA vehicle blueprint |

## Available Scenarios

**Trolley Problems**: `equal-1v1`, `saves-3v0`, `deadzone-3v1`, `footbridge`, `self-sacrifice`, etc.

**Maze**: `maze-1` (153m navigation)

**Free-Roam**: `free-roam-default`, `free-roam-traffic`, `free-roam-heavy`

See `config.py` for full list.
