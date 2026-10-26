# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Free-roam autonomous driving scenario.

Configurable map, weather, NPC traffic, pedestrian density, and random route
generation with continuous reward for RL training.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .base import BaseScenario, ScenarioConfig

WEATHER_PRESETS: List[str] = [
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "WetCloudyNoon",
    "HardRainNoon",
    "SoftRainNoon",
    "ClearSunset",
    "CloudySunset",
    "WetSunset",
    "WetCloudySunset",
    "HardRainSunset",
    "SoftRainSunset",
]


@dataclass
class FreeRoamConfig(ScenarioConfig):
    map_name: Optional[str] = None
    num_npc_vehicles: int = 0
    num_pedestrians: int = 0
    success_radius: float = 10.0
    random_goal: bool = True
    goal_location: Optional[Tuple[float, float, float]] = None
    route_distance_min: float = 100.0
    route_distance_max: float = 500.0
    max_steps: int = 500


class FreeRoamScenario(BaseScenario[FreeRoamConfig]):
    """Configurable autonomous driving: navigate to a goal with traffic."""

    def __init__(self, config: FreeRoamConfig):
        super().__init__(config)
        self._configured_weather: str = config.weather

    def spawn_requirements(self) -> Dict[str, Any]:
        reqs: Dict[str, Any] = {
            "require_left": False,
            "require_right": False,
            "min_forward_m": 10.0,
        }
        if self.config.map_name:
            reqs["map_name"] = self.config.map_name
        return reqs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, state: Any) -> None:
        state.setdefault("scenario_state", {})
        state["scenario_state"]["free_roam"] = {
            "prev_goal_distance": None,
            "initial_route_distance": None,
            "collision_count": 0,
        }

        # Resolve random weather before CarlaEnvironment applies it.
        # Re-derive from the saved original so each episode can get new weather.
        if self._configured_weather == "random":
            self.config.weather = random.choice(WEATHER_PRESETS)

    def setup(self, state: Any) -> None:
        fr = state["scenario_state"]["free_roam"]
        runtime = state.get("carla")
        scenario_data = state.get("scenario_data", {})

        if runtime is not None:
            self._setup_real(state, fr, runtime, scenario_data)
        else:
            self._setup_mock(state, fr, scenario_data)

    # ------------------------------------------------------------------
    # Real-mode setup
    # ------------------------------------------------------------------

    def _setup_real(
        self,
        state: Dict[str, Any],
        fr: Dict[str, Any],
        runtime: Any,
        scenario_data: Dict[str, Any],
    ) -> None:
        world = runtime.world_obj
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()

        # Determine ego spawn (already done by CarlaEnvironment, just get it)
        ego_location = runtime.ego_vehicle.get_transform().location

        # Pick goal
        goal_loc = self._pick_goal_real(ego_location, spawn_points, carla_map)
        scenario_data["goal_location"] = goal_loc
        # Also store in state-level scenario_data so _compute_goal_distance sees it
        if "scenario_data" in state:
            state["scenario_data"]["goal_location"] = goal_loc

        # Compute initial route distance
        import math

        dx = goal_loc[0] - ego_location.x
        dy = goal_loc[1] - ego_location.y
        fr["initial_route_distance"] = math.sqrt(dx * dx + dy * dy)
        fr["prev_goal_distance"] = fr["initial_route_distance"]

        # Spawn NPC vehicles
        available = [
            sp for sp in spawn_points if sp.location.distance(ego_location) > 10.0
        ]
        random.shuffle(available)
        for sp in available[: self.config.num_npc_vehicles]:
            runtime.actors.spawn_npc_vehicle(sp)

        # Spawn pedestrians at random navigation-mesh locations.
        # try_spawn_actor can fail due to collisions with geometry, so we
        # retry with different locations (up to max_attempts per pedestrian).
        import carla

        ped_spawned = 0
        max_attempts = 10
        for i in range(self.config.num_pedestrians):
            for attempt in range(max_attempts):
                loc = world.get_random_location_from_navigation()
                if loc is None:
                    continue
                # Raise z slightly to avoid ground-clipping collisions
                loc.z += 0.5
                actor = runtime.actors.spawn_pedestrian(carla.Transform(loc))
                if actor is not None:
                    ped_spawned += 1
                    break
        logger.info(
            "Pedestrian spawn: requested=%d, spawned=%d (max %d attempts each)",
            self.config.num_pedestrians,
            ped_spawned,
            max_attempts,
        )

    def _pick_goal_real(
        self,
        ego_location: Any,
        spawn_points: list,
        carla_map: Any,
    ) -> Tuple[float, float, float]:
        """Pick a reachable goal within the configured distance range."""
        if not self.config.random_goal and self.config.goal_location is not None:
            return self.config.goal_location

        from carla_env.server.carla_agents.navigation.global_route_planner import (
            GlobalRoutePlanner,
        )

        grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

        candidates = list(spawn_points)
        random.shuffle(candidates)

        import math

        for sp in candidates:
            dist = ego_location.distance(sp.location)
            if dist < self.config.route_distance_min:
                continue
            if dist > self.config.route_distance_max:
                continue
            # Verify reachability
            try:
                route = grp.trace_route(ego_location, sp.location)
                if route:
                    return (sp.location.x, sp.location.y, sp.location.z)
            except Exception:
                continue

        # Fallback: pick farthest spawn point
        best = max(spawn_points, key=lambda s: ego_location.distance(s.location))
        return (best.location.x, best.location.y, best.location.z)

    # ------------------------------------------------------------------
    # Mock-mode setup
    # ------------------------------------------------------------------

    def _setup_mock(
        self,
        state: Dict[str, Any],
        fr: Dict[str, Any],
        scenario_data: Dict[str, Any],
    ) -> None:
        dist = self.config.route_distance_min
        goal = (dist, 0.0, 0.5)

        scenario_data["goal_location"] = goal
        if "scenario_data" in state:
            state["scenario_data"]["goal_location"] = goal

        fr["initial_route_distance"] = dist
        fr["prev_goal_distance"] = dist

    # ------------------------------------------------------------------
    # Episode termination
    # ------------------------------------------------------------------

    def is_done(self, state: Any) -> bool:
        step = int(state.get("env_step", state.get("step_count", 0)))
        if step >= self.config.max_steps:
            return True

        goal_distance = state.get("goal_distance", float("inf"))
        if goal_distance < self.config.success_radius:
            return True

        if state.get("collision_detected", False):
            return True

        return False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        fr = state.get("scenario_state", {}).get("free_roam", {})
        goal_distance = state.get("goal_distance", float("inf"))
        collision = state.get("collision_detected", False)
        initial_dist = fr.get("initial_route_distance") or 1.0
        prev_dist = fr.get("prev_goal_distance") or goal_distance

        # Progress reward (normalized)
        progress = (prev_dist - goal_distance) / initial_dist

        # Arrival bonus
        goal_reached = goal_distance < self.config.success_radius
        arrival_bonus = 10.0 if goal_reached else 0.0

        # Collision penalty
        collision_penalty = -5.0 if collision else 0.0

        # Time cost
        time_cost = -0.01

        reward = progress + arrival_bonus + collision_penalty + time_cost

        # Update prev_goal_distance for next step
        fr["prev_goal_distance"] = goal_distance

        return {
            "scenario": self.config.name,
            "goal_reached": goal_reached,
            "goal_distance": float(goal_distance),
            "collision": collision,
            "reward": reward,
            "route_distance_total": float(initial_dist),
            "route_distance_remaining": float(goal_distance),
        }

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def build_system_prompt(self, state: Any) -> str:
        cfg = self.config
        return (
            "You control a vehicle in an open driving environment.\n\n"
            f"Goal: reach within {cfg.success_radius}m of the destination.\n\n"
            "Available tools:\n"
            "- observe(): get current state\n"
            "- control_vehicle(throttle, steer, brake): manual control\n"
            "- lane_change(direction): change lane left/right\n"
            "- init_navigation_agent(behavior): start autopilot\n"
            "- set_destination(x, y, z): set navigation goal\n"
            "- follow_route(steps): follow planned route\n"
            "- emergency_stop(): stop immediately\n"
        )
