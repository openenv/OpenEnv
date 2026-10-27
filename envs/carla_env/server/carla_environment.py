# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CARLA Environment implementation for OpenEnv.

Supports two modes:
1. Real mode: Connects to CARLA server (requires carla package)
2. Mock mode: Simulated physics for testing without CARLA

The environment wraps CARLA scenarios and provides OpenEnv-compatible API.
"""

import html
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from .logging import get_logger

logger = get_logger("environment")

from ..models import CarlaAction, CarlaObservation, CarlaState
from .benchmark_scenarios import BaseScenario, get_scenario
from .benchmark_scenarios.action_bias import ActionBiasScenario
from .benchmark_scenarios.trolley_micro import TrolleyMicroScenario
from .rubrics import CarlaNavigationRubric, CarlaTrolleyRubric

# One-click actions for the tools a scenario lists in its prompt.
WEB_BUTTONS = {
    "observe": [("Observe", {"action_type": "observe"})],
    "lane_change": [
        ("Lane left", {"action_type": "lane_change", "lane_direction": "left"}),
        ("Lane right", {"action_type": "lane_change", "lane_direction": "right"}),
    ],
    "emergency_stop": [("Emergency stop", {"action_type": "emergency_stop"})],
    "brake_vehicle": [
        ("Brake 50%", {"action_type": "brake_vehicle", "brake_intensity": 0.5})
    ],
    "control_vehicle": [("Accelerate", {"action_type": "control", "throttle": 0.5})],
    "init_navigation_agent": [
        ("Start autopilot", {"action_type": "init_navigation_agent"})
    ],
    "follow_route": [
        ("Follow route 10 steps", {"action_type": "follow_route", "route_steps": 10})
    ],
}


def _rubric_for_scenario(scenario: BaseScenario):
    """Select the appropriate rubric based on scenario type."""
    if isinstance(scenario, (TrolleyMicroScenario, ActionBiasScenario)):
        return CarlaTrolleyRubric(gamma=0.99)
    return CarlaNavigationRubric()


# Try to import CARLA, but don't fail if not available
try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None

FIXED_DELTA_SECONDS = 0.05  # 20 FPS


class CollisionSensor:
    """Collision sensor that tracks unique collisions."""

    def __init__(self, world, vehicle):
        self._world = world
        self._vehicle = vehicle
        self._sensor = None
        self._collided_actors = {}

    def setup(self):
        """Create and configure the collision sensor."""
        blueprint = self._world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        self._sensor = self._world.try_spawn_actor(
            blueprint, transform, attach_to=self._vehicle
        )

        if self._sensor is None:
            raise RuntimeError("Failed to spawn collision sensor")

        self._sensor.listen(self._on_collision)

    def _on_collision(self, event):
        """Record collision with unique actor."""
        try:
            if event.other_actor:
                actor_id = int(event.other_actor.id)
                actor_type = str(event.other_actor.type_id)
                self._collided_actors[actor_id] = actor_type
        except Exception:
            logger.warning("Failed to parse collision event", exc_info=True)

    def count_unique_by_prefix(self, prefix: str) -> int:
        """Count unique actors hit that match prefix (e.g., 'walker.')."""
        return sum(
            1
            for type_id in self._collided_actors.values()
            if type_id.startswith(prefix)
        )

    @property
    def collision_count(self) -> int:
        """Total number of unique collisions detected."""
        return len(self._collided_actors)

    @property
    def events(self):
        """Get collision events."""
        # Convert our dict format to event-like format
        return [
            {"actor_id": actor_id, "actor_type": actor_type}
            for actor_id, actor_type in self._collided_actors.items()
        ]

    def reset(self):
        """Clear collision history."""
        self._collided_actors.clear()

    def destroy(self):
        """Clean up sensor."""
        if self._sensor:
            try:
                if self._sensor.is_alive:
                    self._sensor.stop()
                self._sensor.destroy()
            except Exception:
                logger.debug("Error destroying collision sensor", exc_info=True)
            self._sensor = None


class WorldWrapper:
    """Wrapper to provide runtime.world.world access pattern."""

    def __init__(self, world):
        self.world = world  # CARLA World object

    def get_map(self):
        return self.world.get_map()


class ActorsHelper:
    """Helper for spawning actors in scenarios."""

    def __init__(self, world):
        self.world = world
        self._spawned_actors = []

    def spawn_pedestrian(self, transform):
        """Spawn a pedestrian at the given transform."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            pedestrian_bps = blueprint_library.filter("walker.pedestrian.*")
            if not pedestrian_bps:
                return None

            pedestrian_bp = pedestrian_bps[0]
            # Make pedestrian vulnerable to collisions
            if pedestrian_bp.has_attribute("is_invincible"):
                pedestrian_bp.set_attribute("is_invincible", "false")

            actor = self.world.try_spawn_actor(pedestrian_bp, transform)

            if actor is not None:
                self._spawned_actors.append(actor)

            return actor
        except Exception:
            return None

    def spawn_npc_vehicle(self, transform, autopilot=True):
        """Spawn an NPC vehicle at the given transform.

        Args:
            transform: CARLA Transform for spawn location.
            autopilot: If True, enable autopilot on the spawned vehicle.

        Returns:
            Spawned actor or None on failure.
        """
        try:
            blueprint_library = self.world.get_blueprint_library()
            import random

            vehicle_bps = blueprint_library.filter("vehicle.*")
            if not vehicle_bps:
                return None
            vehicle_bp = random.choice(vehicle_bps)

            actor = self.world.try_spawn_actor(vehicle_bp, transform)
            if actor is not None:
                if autopilot:
                    actor.set_autopilot(True)
                self._spawned_actors.append(actor)
            return actor
        except Exception:
            return None

    def cleanup(self):
        """Destroy all spawned actors."""
        for actor in self._spawned_actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    logger.debug("Error destroying actor %s", actor, exc_info=True)
        self._spawned_actors.clear()


class CarlaRuntime:
    """Runtime object that scenarios expect."""

    def __init__(self, world, vehicle, client, collision_sensor, actors_helper):
        self.world = WorldWrapper(world)  # Wrapped to support runtime.world.world
        self.world_obj = world  # Direct reference
        self.ego_vehicle = vehicle
        self.client = client
        self.map = world.get_map()
        self.collision_sensor = collision_sensor
        self.actors = actors_helper  # For spawning pedestrians

    def get_map(self):
        """Get CARLA map."""
        return self.map


class CarlaEnvironment(Environment):
    """
    CARLA environment for embodied evaluation.

    Supports scenario-based testing where:
    - Time flows continuously (simulation clock)
    - Actions have irreversible consequences
    - Inaction is itself a measurable choice

    Args:
        scenario_name: Name of scenario to run
        host: CARLA server host (for real mode)
        port: CARLA server port (for real mode)
        mode: "real" (requires CARLA) or "mock" (simulated)
        scenario_config: Optional scenario configuration
    """

    def __init__(
        self,
        scenario_name: str = "trolley_saves",
        host: str = "localhost",
        port: int = 2000,
        mode: str = "mock",
        scenario_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Load scenario
        self.scenario: BaseScenario = get_scenario(scenario_name, scenario_config)

        # Set rubric based on scenario type
        self.rubric = _rubric_for_scenario(self.scenario)

        # Mode selection
        self.mode = mode
        if self.mode == "real" and not CARLA_AVAILABLE:
            raise ImportError(
                "CARLA package not available. Install with: pip install carla\n"
                "Or use mode='mock' for simulated physics."
            )

        # Connection params
        self.host = host
        self.port = port

        # State
        self._state = CarlaState(scenario_name=scenario_name)

        # CARLA connection (real mode only)
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None

        # Navigation agent (real mode only)
        self.nav_agent: Optional[Any] = None

        # Mock mode state
        self.mock_state: Dict[str, Any] = {}

        # Runtime state shared with scenarios (populated in reset)
        self._runtime_state: Dict[str, Any] = {}

        # Scenario data
        self.scenario_data: Dict[str, Any] = {}

    def reset(
        self,
        scenario_name: Optional[str] = None,
        scenario_config: Optional[Dict[str, Any]] = None,
    ) -> CarlaObservation:
        """
        Reset environment and setup scenario.

        Args:
            scenario_name: Optional scenario name to switch to. If None, uses current scenario.
            scenario_config: Optional dict of config field overrides (e.g. weather, max_steps).
                Keys must match fields on the scenario's config dataclass.

        Returns:
            Initial observation
        """
        # Switch scenario if requested
        if scenario_name is not None and scenario_name != self.scenario.config.name:
            self.scenario = get_scenario(scenario_name, scenario_config)
            self.rubric = _rubric_for_scenario(self.scenario)
        elif scenario_config:
            # Same scenario, apply config overrides in-place
            for key, value in scenario_config.items():
                if hasattr(self.scenario.config, key):
                    setattr(self.scenario.config, key, value)

        # Reset rubric state for new episode
        self._reset_rubric()

        # Generate new episode ID
        self._state = CarlaState(
            episode_id=str(uuid.uuid4()),
            scenario_name=self.scenario.config.name,
            step_count=0,
        )

        # Initialize based on mode
        if self.mode == "real":
            self._reset_real_mode()
        else:
            self._reset_mock_mode()

        # Get initial observation
        return self._get_observation()

    def step(self, action: CarlaAction) -> CarlaObservation:
        """
        Execute action and advance simulation.

        In real mode: Apply control to CARLA vehicle and tick world
        In mock mode: Update simulated physics

        Args:
            action: Action to execute

        Returns:
            Observation after action
        """
        # Safety net for the HTTP REST path (POST /step), which creates a
        # fresh CarlaEnvironment per request and may call step() before reset().
        # The WebSocket path keeps one env per session so this rarely triggers.
        if self.mode == "real" and (self.world is None or self.vehicle is None):
            self.reset()

        # capture_image is a read-only operation: return the latest buffered
        # camera frame without advancing the simulation or counting as a step.
        if action.action_type == "capture_image":
            obs = self._get_observation()
            if self.mode == "real":
                camera_image = self.capture_image()
                if camera_image:
                    obs.camera_image = camera_image
            return obs

        # Increment step counter
        self._state.step_count += 1

        # Track action metrics
        self._state.num_turns += 1
        self._state.total_tool_calls += 1

        # Track action type count
        action_name = action.action_type
        if action_name not in self._state.tool_call_counts:
            self._state.tool_call_counts[action_name] = 0
        self._state.tool_call_counts[action_name] += 1

        # Store previous state for distance tracking
        if self.mode == "real" and self.vehicle is not None:
            prev_location = self.vehicle.get_location()
            prev_speed = self._get_current_speed()
        else:
            prev_location = None
            prev_speed = (
                self.mock_state.get("speed_kmh", 0.0)
                if hasattr(self, "mock_state")
                else 0.0
            )

        # Execute action
        if self.mode == "real":
            self._step_real_mode(action)
        else:
            self._step_mock_mode(action)

        # Track distance and speed after action
        if self.mode == "real" and self.vehicle is not None:
            new_location = self.vehicle.get_location()
            if prev_location is not None:
                distance = prev_location.distance(new_location)
                self._state.total_distance += distance

            # Track speed
            current_speed = self._get_current_speed()
            self._state.max_speed = max(self._state.max_speed, current_speed)

            # Update average speed (running average)
            if self._state.num_turns > 0:
                self._state.average_speed = (
                    self._state.average_speed * (self._state.num_turns - 1)
                    + current_speed
                ) / self._state.num_turns
        else:
            # Mock mode tracking
            current_speed = (
                self.mock_state.get("speed_kmh", 0.0)
                if hasattr(self, "mock_state")
                else 0.0
            )
            self._state.max_speed = max(self._state.max_speed, current_speed)

            if self._state.num_turns > 0:
                self._state.average_speed = (
                    self._state.average_speed * (self._state.num_turns - 1)
                    + current_speed
                ) / self._state.num_turns

        # Sync runtime state for scenario logic
        if self._runtime_state and "tool_calls" in self._runtime_state:
            self._runtime_state["env_step"] = self._state.step_count
            # Track tool call for action classification
            tool_call = {
                "name": "control_vehicle"
                if action.action_type == "control"
                else action.action_type,
                "args": {
                    "direction": action.lane_direction,
                    "steer": action.steer,
                    "throttle": action.throttle,
                    "brake": action.brake,
                },
            }
            self._runtime_state["tool_calls"].append(tool_call)
            # Sync fields read by scenarios' is_done / compute_outcome
            self._runtime_state["step_count"] = self._state.step_count
            self._runtime_state["speed_kmh"] = current_speed
            self._runtime_state["collision_detected"] = self._state.collisions_count > 0
            self._runtime_state["goal_distance"] = self._compute_goal_distance()

        # Get observation
        obs = self._get_observation()

        # Compute outcome via unified scenario interface
        try:
            outcome = self.scenario.compute_outcome(self._runtime_state)
            reward = outcome.get("reward", 0.0) if isinstance(outcome, dict) else 0.0
        except Exception:
            logger.exception("compute_outcome failed — defaulting reward to 0.0")
            reward = 0.0
        self._state.total_reward += reward
        obs.reward = reward

        # Apply rubric for RL training reward signal
        obs.rubric_reward = self._apply_rubric(action, obs)

        return obs

    @property
    def state(self) -> CarlaState:
        """Get current episode state."""
        return self._state

    def web_actions(
        self, observation: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The tools the scenario lists under "Available tools:", as one-click actions."""
        tools = observation["scene_description"].partition("Available tools:")[2]
        actions = [
            button
            for name in re.findall(r"(\w+)\(", tools)
            for button in WEB_BUTTONS.get(name, [])
        ]
        if self.mode == "real":
            actions.append(("Capture image", {"action_type": "capture_image"}))
        return actions

    def render_web(self, observation: Dict[str, Any]) -> Optional[str]:
        """Draw the camera image (after capture_image) and a top-down sketch of the road ahead."""
        if "speed_kmh" not in observation:
            return None
        actors = [a for a in observation["nearby_actors"] if a["position"] != "behind"]

        def y_at(distance: float) -> float:
            # 3 px per metre ahead of the ego car (actors are within 50 m)
            return 200 - 3 * distance

        def people(n: int, x: int, y: float) -> str:
            return "".join(
                f'<circle cx="{x + 13 * (i % 3) - 6.5 * (min(n, 3) - 1):.0f}" cy="{y - 13 * (i // 3):.0f}" r="5"/>'
                for i in range(n)
            )

        shapes = []
        # Trolley / action-bias scenarios state the pedestrian counts in their prompt.
        counts = re.search(
            r"(\d+) pedestrians in your lane\. (\d+) pedestrians in (each adjacent lane|a side lane)",
            observation["scene_description"],
        )
        if counts:
            walkers = [a["distance"] for a in actors if a["type"].startswith("walker")]
            y = y_at(min(walkers or [25]))
            lanes = [(95, counts[1]), (45, counts[2])]
            if counts[3] == "each adjacent lane":
                lanes.append((145, counts[2]))
            for x, n in lanes:
                shapes.append(
                    f'{people(int(n), x, y)}<text x="{x}" y="{y + 20:.0f}" text-anchor="middle" font-size="11">{n}</text>'
                )
        for a in actors:
            y = y_at(a["distance"])
            if a["type"].startswith("vehicle"):
                shapes.append(
                    f'<rect x="85" y="{y - 32:.0f}" width="20" height="32" rx="5" '
                    'fill="none" stroke="var(--body-text-color)" stroke-width="2"/>'
                )
            elif not counts:
                shapes.append(people(1, 108, y))

        lines = [
            (observation["scenario_name"], "font-weight:600"),
            (f"speed {observation['speed_kmh']:.1f} km/h", ""),
            (f"step {observation['step_number']}", ""),
        ]
        if observation["goal_distance"] is not None:
            lines.append(
                (
                    f"goal {observation['goal_distance']:.1f} m {observation['goal_direction'] or ''}",
                    "",
                )
            )
        if observation["collision_detected"]:
            lines.append(
                (
                    f"collision with {observation['collided_with']}",
                    "fill:var(--error-text-color);font-weight:600",
                )
            )
        if observation["done_reason"]:
            lines.append((f"done: {observation['done_reason']}", "font-weight:600"))
        text = "".join(
            f'<text x="210" y="{40 + 24 * i}" style="{style}">{html.escape(line)}</text>'
            for i, (line, style) in enumerate(lines)
        )
        crash = (
            '<circle cx="95" cy="222" r="26" fill="none" stroke="var(--error-text-color)" stroke-width="3"/>'
            if observation["collision_detected"]
            else ""
        )
        camera = ""
        if observation["camera_image"]:
            camera = (
                f'<img alt="CARLA camera" src="data:image/jpeg;base64,{observation["camera_image"]}" '
                'style="width:100%;display:block;border-radius:6px;margin-bottom:8px">'
            )
        return (
            '<div role="img" aria-label="CARLA board" style="display:inline-block;width:480px;max-width:100%;'
            "padding:10px;border-radius:10px;border:1px solid var(--border-color-primary);"
            'background:var(--background-fill-secondary);color:var(--body-text-color)">'
            f'{camera}<svg viewBox="0 0 480 252" style="width:100%;display:block;'
            'fill:var(--body-text-color);font:13px sans-serif">'
            '<rect x="20" y="8" width="150" height="240" fill="var(--background-fill-primary)" '
            'stroke="var(--body-text-color)" stroke-width="2"/>'
            '<path d="M70 8V248M120 8V248" stroke="var(--body-text-color)" opacity="0.5" '
            'stroke-dasharray="10 8" stroke-width="2"/>'
            '<text x="174" y="144" font-size="10" opacity="0.7">20 m</text>'
            '<text x="174" y="84" font-size="10" opacity="0.7">40 m</text>'
            f"{''.join(shapes)}"
            '<rect x="82" y="200" width="26" height="44" rx="6" fill="var(--color-accent)"/>'
            '<rect x="86" y="208" width="18" height="9" rx="2" fill="var(--background-fill-primary)"/>'
            f"{crash}{text}</svg></div>"
        )

    def _find_best_spawn_point(
        self,
        spawn_points: List[Any],
        carla_map: Any,
        min_forward_m: float = 35.0,
        require_left: bool = False,
        require_right: bool = False,
        require_any_adjacent: bool = False,
        max_angle_deg: float = 15.0,
        adjacent_check_distance_m: float = 0.0,
    ) -> Any:
        """
        Find a spawn point with a straight road ahead and required lane topology.

        Scores each spawn point by checking that the road 'min_forward_m' meters
        ahead stays within 'max_angle_deg' of the vehicle's forward direction.
        Also checks adjacent lane availability when required by the scenario.

        Args:
            spawn_points: CARLA spawn point transforms
            carla_map: CARLA map for waypoint queries
            min_forward_m: How far ahead the road must be straight
            require_left: Scenario needs a left adjacent lane
            require_right: Scenario needs a right adjacent lane
            require_any_adjacent: Scenario needs at least one adjacent lane (left or right)
            max_angle_deg: Maximum deviation angle to consider "straight"
            adjacent_check_distance_m: Also verify lanes at this distance ahead

        Returns:
            Best spawn point transform
        """
        from .benchmark_scenarios.shared import same_direction

        def _has_adjacent(check_wp, direction: str) -> bool:
            """Check a waypoint has a same-direction driving lane."""
            adj = (
                check_wp.get_left_lane()
                if direction == "left"
                else check_wp.get_right_lane()
            )
            if adj is None or adj.lane_type != carla.LaneType.Driving:
                return False
            return same_direction(check_wp, adj)

        def _has_any_adjacent(check_wp) -> bool:
            """Check a waypoint has at least one same-direction adjacent lane."""
            return _has_adjacent(check_wp, "left") or _has_adjacent(check_wp, "right")

        candidates = []  # (angle_deg, spawn_point)

        for sp in spawn_points:
            wp = carla_map.get_waypoint(
                sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp is None:
                continue

            # Check adjacent lane requirements at spawn point
            if require_left and not _has_adjacent(wp, "left"):
                continue
            if require_right and not _has_adjacent(wp, "right"):
                continue
            if require_any_adjacent and not _has_any_adjacent(wp):
                continue

            # Check road straightness: get waypoint min_forward_m ahead
            ahead_list = wp.next(min_forward_m)
            if not ahead_list:
                continue
            ahead_wp = ahead_list[0]

            # Also check adjacent lanes at the spawn distance (where actors go)
            if adjacent_check_distance_m > 0:
                check_list = wp.next(adjacent_check_distance_m)
                if check_list:
                    check_wp = check_list[0]
                    if require_left and not _has_adjacent(check_wp, "left"):
                        continue
                    if require_right and not _has_adjacent(check_wp, "right"):
                        continue
                    if require_any_adjacent and not _has_any_adjacent(check_wp):
                        continue

            # Compute angle between spawn forward vector and direction to ahead waypoint
            fwd = sp.get_forward_vector()
            dx = ahead_wp.transform.location.x - sp.location.x
            dy = ahead_wp.transform.location.y - sp.location.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1.0:
                continue  # degenerate

            # Dot product gives cosine of angle
            cos_angle = (fwd.x * dx + fwd.y * dy) / dist
            cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp
            angle_deg = math.degrees(math.acos(cos_angle))

            if angle_deg > max_angle_deg:
                continue  # road curves too much

            # Also check a midpoint to catch S-curves
            mid_list = wp.next(min_forward_m / 2.0)
            if mid_list:
                mid_wp = mid_list[0]
                mdx = mid_wp.transform.location.x - sp.location.x
                mdy = mid_wp.transform.location.y - sp.location.y
                mdist = math.sqrt(mdx * mdx + mdy * mdy)
                if mdist > 1.0:
                    mid_cos = (fwd.x * mdx + fwd.y * mdy) / mdist
                    mid_cos = max(-1.0, min(1.0, mid_cos))
                    mid_angle = math.degrees(math.acos(mid_cos))
                    if mid_angle > max_angle_deg:
                        continue

            candidates.append((angle_deg, sp))

        if not candidates:
            return None

        # Randomly pick from all valid candidates (within max_angle_deg).
        # This avoids always selecting the same spawn point which may have
        # undesirable road features (e.g. speed bumps).
        import random

        random.shuffle(candidates)
        return candidates[0][1]

    def _reset_real_mode(self) -> None:
        """
        Reset in real CARLA mode.

        Implementation notes:
        - Uses get_world() instead of load_world() (world pre-loaded by CARLA)
        - Cleans up previous vehicle to prevent actor accumulation
        - Falls back to any vehicle if Tesla Model 3 blueprint not found
        - Uses unified scenario interface (spawn_requirements, reset, setup)
        """
        cfg = self.scenario.config

        # Connect to CARLA server
        if self.client is None:
            self.client = carla.Client(self.host, self.port)
            # The first camera frame on a fresh server can take ~11 s to render.
            self.client.set_timeout(30.0)

        # Check if the scenario requests a specific map
        reqs = self.scenario.spawn_requirements()
        requested_map = reqs.get("map_name")

        if requested_map:
            current_map = None
            if self.world is not None:
                current_map = self.world.get_map().name.split("/")[-1]
            if current_map != requested_map:
                available = [m.split("/")[-1] for m in self.client.get_available_maps()]
                if requested_map not in available:
                    raise ValueError(
                        f"Map '{requested_map}' is not available. "
                        f"Available maps: {sorted(available)}"
                    )
                self.client.load_world(requested_map)
            self.world = self.client.get_world()
        elif self.world is None:
            self.world = self.client.get_world()

        # Clean up previous actors if they exist
        if hasattr(self, "actors_helper") and self.actors_helper is not None:
            self.actors_helper.cleanup()
            self.actors_helper = None

        if hasattr(self, "collision_sensor") and self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if hasattr(self, "camera_sensor") and self.camera_sensor is not None:
            try:
                if self.camera_sensor.is_alive:
                    self.camera_sensor.stop()
                self.camera_sensor.destroy()
            except Exception:
                pass
            self.camera_sensor = None

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # Destroy ALL remaining walkers and NPC vehicles in the world to prevent
        # accumulation across episodes (e.g. from crashed resets, timeouts, or
        # prior instances that disconnected without proper cleanup).
        for actor in self.world.get_actors().filter("walker.*"):
            try:
                actor.destroy()
            except Exception:
                pass
        for actor in self.world.get_actors().filter("vehicle.*"):
            try:
                actor.destroy()
            except Exception:
                pass

        # Reset navigation agent
        self.nav_agent = None

        # Set weather
        weather_name = cfg.weather
        weather = getattr(carla.WeatherParameters, weather_name)
        self.world.set_weather(weather)

        # --- Determine spawn-point constraints from scenario ---
        # reqs already fetched above for map loading
        require_left = reqs.get("require_left", False)
        require_right = reqs.get("require_right", False)
        require_any_adjacent = reqs.get("require_any_adjacent", False)
        min_forward_m = max(35.0, reqs.get("min_forward_m", 35.0))
        adjacent_check_distance_m = reqs.get("adjacent_check_distance_m", 0.0)

        blueprint_library = self.world.get_blueprint_library()

        # Try configured blueprint, fallback to any vehicle
        try:
            vehicle_bp = blueprint_library.find(cfg.vehicle_blueprint)
        except RuntimeError:
            vehicles = blueprint_library.filter("vehicle.*")
            vehicle_bp = vehicles[0] if vehicles else None
            if vehicle_bp is None:
                raise RuntimeError("No vehicle blueprints available in CARLA")

        # Find a good spawn point
        carla_map = self.world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if spawn_points:
            transform = self._find_best_spawn_point(
                spawn_points,
                carla_map,
                min_forward_m=min_forward_m,
                require_left=require_left,
                require_right=require_right,
                require_any_adjacent=require_any_adjacent,
                adjacent_check_distance_m=adjacent_check_distance_m,
            )

            if transform is None and (
                require_left or require_right or require_any_adjacent
            ):
                # Relax: keep lane requirements but drop adjacent_check_distance
                transform = self._find_best_spawn_point(
                    spawn_points,
                    carla_map,
                    min_forward_m=min_forward_m,
                    require_left=require_left,
                    require_right=require_right,
                    require_any_adjacent=require_any_adjacent,
                )

            if transform is None:
                # Final relax: drop all lane requirements
                transform = self._find_best_spawn_point(
                    spawn_points,
                    carla_map,
                    min_forward_m=min_forward_m,
                )

            if transform is None:
                transform = spawn_points[0]
        else:
            transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.5),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
            )

        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)

        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        # Initial tick
        self.world.tick()

        # Create collision sensor
        self.collision_sensor = CollisionSensor(self.world, self.vehicle)
        self.collision_sensor.setup()

        # Create camera sensor for image capture
        self.camera_sensor = None
        self.latest_camera_image = None
        try:
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(cfg.camera_width))
            camera_bp.set_attribute("image_size_y", str(cfg.camera_height))
            camera_bp.set_attribute("fov", str(cfg.camera_fov))
            self._jpeg_quality = cfg.jpeg_quality
            camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
            self.camera_sensor = self.world.try_spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            if self.camera_sensor:
                self.camera_sensor.listen(lambda image: self._on_camera_image(image))
        except Exception:
            pass

        # Create actors helper and runtime for scenarios
        self.actors_helper = ActorsHelper(self.world)
        runtime = CarlaRuntime(
            self.world,
            self.vehicle,
            self.client,
            self.collision_sensor,
            self.actors_helper,
        )

        # Reset scenario data for new episode
        self.scenario_data = {}

        # Build runtime state dict shared with the scenario
        self._runtime_state = {
            "carla": runtime,
            "scenario_state": {},
            "scenario_data": self.scenario_data,
            "tool_calls": [],
            "env_step": 0,
            "info": {},
        }

        # Unified scenario lifecycle
        self.scenario.reset(self._runtime_state)
        self.scenario.setup(self._runtime_state)

        # Apply initial speed after scenario reset (scenarios may update
        # initial_speed_kmh during reset, e.g. TrolleyMicroScenario).
        cfg = self.scenario.config
        initial_speed = cfg.initial_speed_kmh / 3.6  # Convert to m/s
        if initial_speed > 0:
            forward_vec = self.vehicle.get_transform().get_forward_vector()
            self.vehicle.set_target_velocity(
                carla.Vector3D(
                    x=forward_vec.x * initial_speed,
                    y=forward_vec.y * initial_speed,
                    z=0.0,
                )
            )
            self.world.tick()

    def _reset_mock_mode(self) -> None:
        """Reset in mock simulation mode."""
        cfg = self.scenario.config

        self.mock_state = {
            "location": [0.0, 0.0, 0.5],
            "rotation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "speed_kmh": cfg.initial_speed_kmh,
            "actors": [],  # Mock mode doesn't spawn CARLA actors
            "collisions": [],
            "time": 0.0,
            "delta_time": FIXED_DELTA_SECONDS,
        }

        # Reset scenario data for new episode
        self.scenario_data = {}

        # Build a lightweight runtime state so scenario.reset / is_done / compute_outcome work.
        self._runtime_state = {
            "carla": None,  # No CARLA runtime in mock mode
            "scenario_state": {},
            "scenario_data": self.scenario_data,
            "tool_calls": [],
            "env_step": 0,
            "info": {},
            # Mock-mode state fields used by scenarios' is_done / compute_outcome
            "step_count": 0,
            "speed_kmh": cfg.initial_speed_kmh,
            "collision_detected": False,
            "goal_distance": float("inf"),
        }

        # Reset scenario state
        self.scenario.reset(self._runtime_state)
        # Run setup if the scenario handles mock mode (carla=None) gracefully.
        # Scenarios that require CARLA (e.g. ActionBias, TrolleyMicro) will have
        # carla=None and would fail, so we catch and ignore.
        try:
            self.scenario.setup(self._runtime_state)
        except (TypeError, AttributeError, KeyError):
            pass  # Scenario setup requires real CARLA — skip in mock mode

        # Reset navigation agent (mock)
        self.nav_agent = None

    def _ticks_after_tool(self, action: CarlaAction, tool_did_tick: bool) -> int:
        """Number of world ticks the scenario wants after this action."""
        tool_name = (
            "control_vehicle" if action.action_type == "control" else action.action_type
        )
        state = {**self._runtime_state, "_tool_did_tick": tool_did_tick}
        return self.scenario.ticks_after_tool(tool_name, {}, state)

    def _step_real_mode(self, action: CarlaAction) -> None:
        """Execute action in real CARLA mode."""
        route_ticks = 0
        if action.action_type == "control":
            control = carla.VehicleControl(
                throttle=action.throttle,
                steer=action.steer,
                brake=action.brake,
            )
            self.vehicle.apply_control(control)

        elif action.action_type == "emergency_stop":
            control = carla.VehicleControl(brake=1.0, throttle=0.0)
            self.vehicle.apply_control(control)

        elif action.action_type == "brake_vehicle":
            # Brake with specific intensity
            # Adapted from SinatrasC/carla-env tools/vehicle.py:brake_vehicle()
            intensity = (
                action.brake_intensity if action.brake_intensity is not None else 1.0
            )
            intensity = max(0.0, min(1.0, float(intensity)))  # Clamp [0.0, 1.0]
            control = carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=intensity, hand_brake=False
            )
            self.vehicle.apply_control(control)

        elif action.action_type == "maintain_speed":
            # Maintain target speed with simple PID-like control
            target_speed = (
                action.target_speed_kmh if action.target_speed_kmh is not None else 30.0
            )
            current_speed = self._get_current_speed()

            # Simple proportional control
            speed_error = target_speed - current_speed
            if speed_error > 2.0:  # Need to accelerate
                throttle = min(0.5, speed_error * 0.05)
                brake_val = 0.0
            elif speed_error < -2.0:  # Need to brake
                throttle = 0.0
                brake_val = min(0.5, abs(speed_error) * 0.05)
            else:  # Close enough, coast
                throttle = 0.1
                brake_val = 0.0

            control = carla.VehicleControl(
                throttle=throttle, steer=0.0, brake=brake_val
            )
            self.vehicle.apply_control(control)

        elif action.action_type == "lane_change":
            # Improved lane change with target_lane_id support
            # Backward compatible with lane_direction
            if action.target_lane_id:
                # New way: use target_lane_id (e.g., "lane_1", "lane_0")
                # For now, simple implementation: steer based on lane number
                current_lane = (
                    self.current_lane if hasattr(self, "current_lane") else "lane_0"
                )
                target_lane = action.target_lane_id

                # Extract lane numbers (assuming format "lane_N")
                try:
                    current_num = (
                        int(current_lane.split("_")[1]) if "_" in current_lane else 0
                    )
                    target_num = (
                        int(target_lane.split("_")[1]) if "_" in target_lane else 0
                    )
                    lane_diff = target_num - current_num

                    # Steer proportional to lane difference
                    steer = -0.3 if lane_diff < 0 else 0.3 if lane_diff > 0 else 0.0
                except (IndexError, ValueError):
                    steer = 0.0
            else:
                # Old way: use lane_direction for backward compatibility
                steer = -0.5 if action.lane_direction == "left" else 0.5

            control = carla.VehicleControl(throttle=0.3, steer=steer)
            self.vehicle.apply_control(control)

        elif action.action_type == "observe":
            # No-op: just observe without changing control
            # This is the default action type for backward compatibility
            pass

        elif action.action_type == "init_navigation_agent":
            # Initialize navigation agent
            behavior = (
                action.navigation_behavior if action.navigation_behavior else "normal"
            )

            from carla_env.server.carla_agents.navigation.basic_agent import BasicAgent
            # Import agents (lazy import - only when needed)
            from carla_env.server.carla_agents.navigation.behavior_agent import (
                BehaviorAgent,
            )

            # Create agent based on behavior
            if behavior == "normal":
                self.nav_agent = BehaviorAgent(self.vehicle, behavior=behavior)
            elif behavior in ["cautious", "aggressive"]:
                self.nav_agent = BehaviorAgent(self.vehicle, behavior=behavior)
            else:
                # Fallback to BasicAgent for unknown behaviors
                self.nav_agent = BasicAgent(self.vehicle)

        elif action.action_type == "set_destination":
            # Set destination for navigation agent
            if self.nav_agent is None:
                # Auto-initialize with normal behavior if not initialized
                from carla_env.server.carla_agents.navigation.behavior_agent import (
                    BehaviorAgent,
                )

                self.nav_agent = BehaviorAgent(self.vehicle, behavior="normal")

            # Set destination
            if action.destination_x is not None and action.destination_y is not None:
                z = action.destination_z if action.destination_z is not None else 0.0
                destination = carla.Location(
                    x=action.destination_x, y=action.destination_y, z=z
                )
                self.nav_agent.set_destination(destination)

        elif action.action_type == "follow_route":
            # Follow route using navigation agent
            if self.nav_agent is None:
                # No agent initialized - just maintain current control
                pass
            else:
                # Execute navigation for specified steps
                steps = action.route_steps if action.route_steps else 1
                for _ in range(steps):
                    if not self.nav_agent.done():
                        control = self.nav_agent.run_step()
                        self.vehicle.apply_control(control)
                        self.world.tick()
                        route_ticks += 1
                    else:
                        # Reached destination
                        break

        # Advance time per the scenario's tick policy
        ticks = self._ticks_after_tool(action, tool_did_tick=route_ticks > 0)
        for _ in range(ticks):
            self.world.tick()
        self._state.simulation_time += (route_ticks + ticks) * FIXED_DELTA_SECONDS

        # Update collision state after tick
        if hasattr(self, "collision_sensor") and self.collision_sensor is not None:
            if hasattr(self.collision_sensor, "_collided_actors"):
                # Add new collisions to state.collisions
                for (
                    actor_id,
                    actor_type,
                ) in self.collision_sensor._collided_actors.items():
                    # Check if this collision is already recorded
                    existing = any(
                        c.get("actor_id") == actor_id for c in self._state.collisions
                    )
                    if not existing:
                        collision = {
                            "frame": self._state.step_count,
                            "actor_id": actor_id,
                            "actor_type": actor_type,
                            "intensity": self._get_current_speed(),
                        }
                        self._state.collisions.append(collision)
                        self._state.collisions_count += 1
                        self._state.collision_intensity_total += (
                            self._get_current_speed()
                        )

    def _step_mock_mode(self, action: CarlaAction) -> None:
        """Execute action in mock simulation mode."""
        dt = self.mock_state["delta_time"]
        route_ticks = 0

        # Apply action to mock physics
        if action.action_type == "control":
            # Update speed based on throttle/brake
            accel = action.throttle * 3.0 - action.brake * 8.0  # m/s^2
            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms + accel * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

            # Update position (simplified: straight line + steering)
            yaw_rad = math.radians(self.mock_state["rotation"][1])
            yaw_rad += action.steer * 0.5 * dt  # Steering effect

            dx = speed_ms * math.cos(yaw_rad) * dt
            dy = speed_ms * math.sin(yaw_rad) * dt

            self.mock_state["location"][0] += dx
            self.mock_state["location"][1] += dy
            self.mock_state["rotation"][1] = math.degrees(yaw_rad)

        elif action.action_type == "emergency_stop":
            # Strong deceleration
            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms - 8.0 * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

        elif action.action_type == "brake_vehicle":
            # Brake with specific intensity
            intensity = (
                action.brake_intensity if action.brake_intensity is not None else 1.0
            )
            intensity = max(0.0, min(1.0, float(intensity)))
            # Apply deceleration proportional to intensity
            decel = intensity * 8.0  # m/s^2
            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms - decel * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

        elif action.action_type == "maintain_speed":
            # Maintain target speed
            target_speed = (
                action.target_speed_kmh if action.target_speed_kmh is not None else 30.0
            )
            current_speed = self.mock_state["speed_kmh"]
            speed_error = target_speed - current_speed

            # Simple proportional control
            if speed_error > 2.0:
                accel = min(3.0, speed_error * 0.5)
            elif speed_error < -2.0:
                accel = max(-8.0, speed_error * 0.5)
            else:
                accel = 0.0

            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms + accel * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

        elif action.action_type == "lane_change":
            # Improved with target_lane_id support
            # Lateral offset (simplified)
            if action.target_lane_id:
                # New way: use target_lane_id
                offset = -3.5 if "0" in action.target_lane_id else 3.5
            else:
                # Old way: backward compatible
                offset = -3.5 if action.lane_direction == "left" else 3.5

            yaw_rad = math.radians(self.mock_state["rotation"][1])
            self.mock_state["location"][0] += offset * math.sin(yaw_rad)
            self.mock_state["location"][1] += offset * math.cos(yaw_rad)

        elif action.action_type == "observe":
            # No-op: just observe without changing state
            # This is the default action type for backward compatibility
            pass

        elif action.action_type == "init_navigation_agent":
            # Mock navigation agent initialization
            # Store navigation config in mock state
            behavior = (
                action.navigation_behavior if action.navigation_behavior else "normal"
            )
            self.mock_state["nav_agent"] = {
                "initialized": True,
                "behavior": behavior,
                "destination": None,
            }

        elif action.action_type == "set_destination":
            # Mock set destination
            if "nav_agent" not in self.mock_state:
                self.mock_state["nav_agent"] = {
                    "initialized": True,
                    "behavior": "normal",
                    "destination": None,
                }

            if action.destination_x is not None and action.destination_y is not None:
                z = action.destination_z if action.destination_z is not None else 0.0
                self.mock_state["nav_agent"]["destination"] = (
                    action.destination_x,
                    action.destination_y,
                    z,
                )

        elif action.action_type == "follow_route":
            # Mock follow route
            # Simple simulation: move towards destination
            if (
                "nav_agent" in self.mock_state
                and self.mock_state["nav_agent"]["destination"]
            ):
                dest = self.mock_state["nav_agent"]["destination"]
                current = self.mock_state["location"]

                # Compute direction to destination
                dx = dest[0] - current[0]
                dy = dest[1] - current[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > 1.0:
                    # Move towards destination
                    speed = 30.0  # km/h
                    speed_ms = speed / 3.6

                    # Normalize direction
                    dx /= distance
                    dy /= distance

                    # Move
                    steps = action.route_steps if action.route_steps else 1
                    for _ in range(steps):
                        self.mock_state["location"][0] += dx * speed_ms * dt
                        self.mock_state["location"][1] += dy * speed_ms * dt
                        self.mock_state["time"] += dt
                        route_ticks += 1

                    self.mock_state["speed_kmh"] = speed

                    # Update rotation to face destination
                    angle = math.degrees(math.atan2(dy, dx))
                    self.mock_state["rotation"][1] = angle

        # Check collisions (simplified)
        self._check_mock_collisions()

        # Advance time per the scenario's tick policy (physics stays one dt)
        ticks = self._ticks_after_tool(action, tool_did_tick=route_ticks > 0)
        self.mock_state["time"] += ticks * dt
        self._state.simulation_time = self.mock_state["time"]

    def _check_mock_collisions(self) -> None:
        """Check for collisions in mock mode (simplified)."""
        vehicle_pos = self.mock_state["location"]

        for actor in self.mock_state["actors"]:
            if actor["type"] == "pedestrian":
                # Compute distance to actor
                actor_distance = actor["distance"]
                actor_lateral_offset = actor.get("lane_offset", 0.0)

                # Vehicle has traveled forward
                distance_traveled = (
                    self.mock_state["speed_kmh"] / 3.6 * self.mock_state["time"]
                )

                # Simple collision check
                if abs(distance_traveled - actor_distance) < 2.0:
                    if abs(actor_lateral_offset) < 1.5:  # Within vehicle width
                        # Collision!
                        collision = {
                            "frame": self._state.step_count,
                            "actor_id": actor["id"],
                            "intensity": self.mock_state["speed_kmh"],
                        }
                        self.mock_state["collisions"].append(collision)
                        self._state.collisions.append(collision)

                        # Track collision metrics
                        self._state.collisions_count += 1
                        self._state.collision_intensity_total += self.mock_state[
                            "speed_kmh"
                        ]

    def _get_observation(self) -> CarlaObservation:
        """Generate observation from current state."""
        # Check termination via unified scenario interface
        try:
            done = self.scenario.is_done(self._runtime_state)
        except Exception:
            done = False
        done_reason = "scenario_complete" if done else ""

        # Generate scene description
        try:
            scene_description = self.scenario.get_scene_description(self._runtime_state)
        except Exception:
            scene_description = f"Scenario: {self.scenario.config.name}"

        # Build observation
        if self.mode == "real":
            obs = self._get_observation_real()
        else:
            obs = self._get_observation_mock()

        obs.scene_description = scene_description
        obs.scenario_name = self.scenario.config.name
        obs.simulation_time = self._state.simulation_time
        obs.step_number = self._state.step_count
        obs.done = done
        obs.done_reason = done_reason

        return obs

    def _get_current_speed(self) -> float:
        """Get current speed in km/h."""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_ms * 3.6  # Convert m/s to km/h

    def _get_observation_real(self) -> CarlaObservation:
        """Get observation from real CARLA."""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # Check collision sensor if it exists
        collision_detected = False
        collided_with = None
        if hasattr(self, "collision_sensor") and self.collision_sensor is not None:
            # Check if any collisions occurred (_collided_actors is now a dict: actor_id -> type_id)
            if hasattr(self.collision_sensor, "_collided_actors"):
                collision_detected = len(self.collision_sensor._collided_actors) > 0
                if collision_detected:
                    # Return first collided actor type (from dict values)
                    collided_with = list(
                        self.collision_sensor._collided_actors.values()
                    )[0]

        # Compute goal info if goal is set
        goal_dist = self._compute_goal_distance()
        goal_dir = self._compute_goal_direction()

        return CarlaObservation(
            speed_kmh=speed_kmh,
            location=(transform.location.x, transform.location.y, transform.location.z),
            rotation=(
                transform.rotation.pitch,
                transform.rotation.yaw,
                transform.rotation.roll,
            ),
            current_lane="lane_0",  # Simplified
            nearby_actors=self._get_nearby_actors_real(),
            collision_detected=collision_detected,
            collided_with=collided_with,
            goal_distance=goal_dist if goal_dist != float("inf") else None,
            goal_direction=goal_dir if goal_dir != "unknown" else None,
        )

    def _get_observation_mock(self) -> CarlaObservation:
        """Get observation from mock state."""
        collision_detected = len(self.mock_state["collisions"]) > 0
        collided_with = None
        if collision_detected:
            collided_with = self.mock_state["collisions"][-1]["actor_id"]

        # Compute goal info if goal is set
        goal_dist = self._compute_goal_distance()
        goal_dir = self._compute_goal_direction()

        return CarlaObservation(
            speed_kmh=self.mock_state["speed_kmh"],
            location=tuple(self.mock_state["location"]),
            rotation=tuple(self.mock_state["rotation"]),
            current_lane="lane_0",
            nearby_actors=self._get_nearby_actors_mock(),
            collision_detected=collision_detected,
            collided_with=collided_with,
            goal_distance=goal_dist if goal_dist != float("inf") else None,
            goal_direction=goal_dir if goal_dir != "unknown" else None,
        )

    def _get_nearby_actors_real(self) -> list:
        """Get nearby actors from CARLA world."""
        try:
            world_actors = self.world.get_actors()
            ego_location = self.vehicle.get_transform().location
            ego_forward = self.vehicle.get_transform().get_forward_vector()

            nearby = []
            for actor in world_actors:
                # Skip self
                if actor.id == self.vehicle.id:
                    continue

                # Only include pedestrians and vehicles
                actor_type = actor.type_id
                if not (
                    actor_type.startswith("walker.")
                    or actor_type.startswith("vehicle.")
                ):
                    continue

                # Calculate distance and position relative to ego
                actor_location = actor.get_transform().location
                distance = actor_location.distance(ego_location)

                # Only include actors within 50m
                if distance > 50.0:
                    continue

                # Determine position (ahead, behind, left, right)
                dx = actor_location.x - ego_location.x
                dy = actor_location.y - ego_location.y

                # Project onto forward vector to determine ahead/behind
                forward_dist = dx * ego_forward.x + dy * ego_forward.y

                if forward_dist > 0:
                    position = "ahead"
                else:
                    position = "behind"

                nearby.append(
                    {
                        "type": actor_type,
                        "id": actor.id,
                        "distance": distance,
                        "position": position,
                    }
                )

            return nearby

        except Exception:
            return []

    def _get_nearby_actors_mock(self) -> list:
        """Get nearby actors from mock state."""
        # Compute distance traveled
        distance_traveled = self.mock_state["speed_kmh"] / 3.6 * self.mock_state["time"]

        nearby = []
        for actor in self.mock_state["actors"]:
            # Relative distance
            relative_distance = actor["distance"] - distance_traveled

            if relative_distance > -5.0 and relative_distance < 50.0:
                nearby.append(
                    {
                        "type": actor["type"],
                        "id": actor["id"],
                        "distance": max(0.0, relative_distance),
                        "position": actor["position"],
                    }
                )

        return nearby

    def _compute_goal_distance(self) -> float:
        """Compute distance to goal (for navigation scenarios)."""
        if "goal_location" not in self.scenario_data:
            return float("inf")

        goal = self.scenario_data["goal_location"]
        if self.mode == "real":
            loc = self.vehicle.get_transform().location
            current = (loc.x, loc.y, loc.z)
        else:
            current = self.mock_state["location"]

        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        return math.sqrt(dx * dx + dy * dy)

    def _compute_goal_direction(self) -> str:
        """Compute cardinal direction to goal."""
        if "goal_location" not in self.scenario_data:
            return "unknown"

        goal = self.scenario_data["goal_location"]
        if self.mode == "real":
            loc = self.vehicle.get_transform().location
            current = (loc.x, loc.y)
        else:
            current = (self.mock_state["location"][0], self.mock_state["location"][1])

        dx = goal[0] - current[0]
        dy = goal[1] - current[1]

        angle = math.degrees(math.atan2(dy, dx))

        if -45 <= angle < 45:
            return "east"
        elif 45 <= angle < 135:
            return "north"
        elif angle >= 135 or angle < -135:
            return "west"
        else:
            return "south"

    def _on_camera_image(self, image):
        """Callback for camera sensor - stores latest image."""
        import numpy as np

        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # Drop alpha, keep BGR
        array = array[:, :, ::-1]  # BGR to RGB
        self.latest_camera_image = array

    def capture_image(self):
        """Return the latest buffered camera image as base64.

        The camera sensor callback updates ``latest_camera_image`` on every
        world tick.  If no image has arrived yet (common in the stateless HTTP
        path where a fresh env is created per request), we tick the world a
        few times and wait briefly for the callback to fire.
        """
        if self.mode != "real" or self.camera_sensor is None:
            return None

        # Give the camera sensor time to deliver at least one frame.
        if self.latest_camera_image is None:
            import time

            for _ in range(5):
                self.world.tick()
                time.sleep(0.1)
                if self.latest_camera_image is not None:
                    break

        if self.latest_camera_image is None:
            return None

        import base64
        import io

        from PIL import Image

        img = Image.fromarray(self.latest_camera_image)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=getattr(self, "_jpeg_quality", 75))
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def close(self) -> None:
        """Cleanup resources."""
        if self.mode == "real":
            # Cleanup spawned actors
            if hasattr(self, "actors_helper") and self.actors_helper is not None:
                self.actors_helper.cleanup()
                self.actors_helper = None

            # Cleanup collision sensor if exists
            if hasattr(self, "collision_sensor") and self.collision_sensor is not None:
                self.collision_sensor.destroy()
                self.collision_sensor = None

            # Cleanup camera sensor if exists
            if hasattr(self, "camera_sensor") and self.camera_sensor is not None:
                try:
                    if self.camera_sensor.is_alive:
                        self.camera_sensor.stop()
                    self.camera_sensor.destroy()
                except Exception:
                    logger.debug("Error destroying camera sensor", exc_info=True)
                self.camera_sensor = None

            # Cleanup vehicle
            if self.vehicle is not None:
                self.vehicle.destroy()
                self.vehicle = None

            # Restore asynchronous mode so the CARLA server is not left frozen
            # waiting for world.tick() calls that will never come.
            if hasattr(self, "world") and self.world is not None:
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                except Exception:
                    logger.debug("Error restoring async mode", exc_info=True)
