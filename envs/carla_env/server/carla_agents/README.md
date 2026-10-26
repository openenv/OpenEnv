# CARLA Agents

Vendorized CARLA navigation agents from the official CARLA Python API.

## Overview

These agents provide autonomous navigation capabilities for CARLA vehicles:

- **BasicAgent**: Simple point-to-point navigation
- **BehaviorAgent**: Advanced navigation with traffic behavior (cautious, normal, aggressive)
- **Controllers**: PID controllers for vehicle control
- **LocalPlanner**: Local path planning and following
- **GlobalRoutePlanner**: Global route planning using CARLA road network

## Source

Adapted from: https://github.com/carla-simulator/carla (Python API)
Used in: [SinatrasC/carla-env](https://github.com/SinatrasC/carla-env) for PrimeIntellect benchmarks

## Requirements

These agents require:
- **CARLA server running** (real mode only)
- `carla` Python package (carla-ue5-api==0.10.0)
- `numpy` for computations

**Note**: Agents are NOT available in mock mode. They require a live CARLA server.

## Usage

### BasicAgent

Simple point-to-point navigation:

```python
from carla_env.server.carla_agents.navigation.basic_agent import BasicAgent

# Initialize agent (requires CARLA vehicle)
agent = BasicAgent(vehicle)

# Set destination
destination = carla.Location(x=100.0, y=50.0, z=0.0)
agent.set_destination(destination)

# Run step (returns VehicleControl)
control = agent.run_step()
vehicle.apply_control(control)
world.tick()

# Check if done
if agent.done():
    print("Reached destination!")
```

### BehaviorAgent

Advanced navigation with traffic behavior:

```python
from carla_env.server.carla_agents.navigation.behavior_agent import BehaviorAgent

# Initialize with behavior
agent = BehaviorAgent(
    vehicle,
    behavior='normal'  # 'cautious', 'normal', or 'aggressive'
)

# Set destination and target speed
destination = carla.Location(x=100.0, y=50.0, z=0.0)
agent.set_destination(destination)
agent.set_target_speed(30.0)  # km/h

# Run step
control = agent.run_step()
vehicle.apply_control(control)
world.tick()
```

### Behavior Types

- **cautious**: Defensive driving, lower speeds, large safety margins
- **normal**: Standard driving behavior (default)
- **aggressive**: Faster speeds, smaller safety margins

## Implementation Notes

### When to Use Each Agent

- **BasicAgent**: Use for simple A-to-B navigation without traffic
- **BehaviorAgent**: Use for realistic navigation with traffic awareness

### Integration with CarlaEnvironment

Agents are integrated via navigation actions:
- `init_navigation_agent(behavior)`
- `set_destination(x, y, z)`
- `follow_route(num_steps)`

Example:
```python
# Initialize agent
env.step(CarlaAction(
    action_type="init_navigation_agent",
    navigation_behavior="normal"
))

# Set destination
env.step(CarlaAction(
    action_type="set_destination",
    destination_x=100.0,
    destination_y=50.0,
    destination_z=0.0
))

# Follow route
for _ in range(100):
    result = env.step(CarlaAction(
        action_type="follow_route",
        route_steps=1
    ))
    if result.done:
        break
```

## Architecture

```
carla_agents/
├── navigation/
│   ├── basic_agent.py           # BasicAgent class
│   ├── behavior_agent.py        # BehaviorAgent class
│   ├── behavior_types.py        # Behavior type definitions
│   ├── controller.py            # PID controllers
│   │   ├── VehiclePIDController
│   │   ├── PIDLongitudinalController (throttle/brake)
│   │   └── PIDLateralController (steering)
│   ├── local_planner.py         # LocalPlanner class
│   └── global_route_planner.py  # GlobalRoutePlanner class
└── tools/
    └── misc.py                  # Utility functions
```

## Controllers

PID controllers for smooth vehicle control:

```python
from carla_env.server.carla_agents.navigation.controller import VehiclePIDController

# Create controller
controller = VehiclePIDController(
    vehicle,
    args_lateral={'K_P': 1.0, 'K_I': 0.0, 'K_D': 0.0},
    args_longitudinal={'K_P': 1.0, 'K_I': 0.0, 'K_D': 0.0}
)

# Run control
target_speed = 30.0  # km/h
target_waypoint = ...  # carla.Waypoint
control = controller.run_step(target_speed, target_waypoint)
```

## Planners

### LocalPlanner

Follows a queue of waypoints:

```python
from carla_env.server.carla_agents.navigation.local_planner import LocalPlanner

planner = LocalPlanner(vehicle)

# Set destination (computes waypoint queue)
destination = carla.Location(x=100.0, y=50.0, z=0.0)
planner.set_destination(destination)

# Run step
control = planner.run_step()

# Check waypoint queue
remaining = len(planner.waypoints_queue)
```

### GlobalRoutePlanner

Plans routes on CARLA road network:

```python
from carla_env.server.carla_agents.navigation.global_route_planner import GlobalRoutePlanner

planner = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)

# Plan route
start_location = vehicle.get_location()
end_location = carla.Location(x=100.0, y=50.0, z=0.0)

route = planner.trace_route(start_location, end_location)
# Returns: [(waypoint, road_option), ...]
```

## Testing

Agents can only be tested with a running CARLA server:

```bash
# In production (HF with GPU + CARLA)
PYTHONPATH=src:envs uv run python test_day3_agents.py
```

Local testing without CARLA will fail with:
```
ModuleNotFoundError: No module named 'carla'
```

This is expected and normal.

## Integration

1. Implement navigation actions in CarlaEnvironment
2. Add agent state management (store agent instance)
3. Create navigation examples
4. Test end-to-end with real CARLA

## License

See LICENSE file. Original CARLA agents are under MIT license.
