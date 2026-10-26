# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for CARLA environment.

Defines Action, Observation, and State for embodied evaluation scenarios.
"""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import Field

ActionType = Literal[
    "control",
    "emergency_stop",
    "brake_vehicle",
    "maintain_speed",
    "lane_change",
    "observe",
    "init_navigation_agent",
    "set_destination",
    "follow_route",
    "capture_image",
]


class CarlaAction(Action):
    """
    Action for CARLA vehicle control.

    Attributes:
        action_type: Type of action (control, emergency_stop, lane_change, observe, maintain_speed, brake_vehicle, init_navigation_agent, set_destination, follow_route, capture_image)
        throttle: Throttle value [0.0, 1.0] for "control" actions
        steer: Steering value [-1.0, 1.0] for "control" actions
        brake: Brake value [0.0, 1.0] for "control" actions
        lane_direction: Direction for "lane_change" ("left" or "right")
        target_speed_kmh: Target speed in km/h for "maintain_speed"
        brake_intensity: Brake intensity [0.0, 1.0] for "brake_vehicle"
        target_lane_id: Target lane ID for improved "lane_change"
        navigation_behavior: Behavior for navigation agent ("cautious", "normal", "aggressive")
        destination_x: Destination X coordinate for navigation
        destination_y: Destination Y coordinate for navigation
        destination_z: Destination Z coordinate for navigation
        route_steps: Number of steps to follow route
    """

    action_type: ActionType = Field(default="observe", description="Type of action")
    throttle: float = Field(default=0.0, ge=0.0, le=1.0, description="Throttle value")
    steer: float = Field(default=0.0, ge=-1.0, le=1.0, description="Steering value")
    brake: float = Field(default=0.0, ge=0.0, le=1.0, description="Brake value")
    lane_direction: str | None = Field(
        default=None,
        description="Lane change direction (deprecated, use target_lane_id)",
    )

    # Enhanced action parameters
    target_speed_kmh: float | None = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Target speed in km/h for maintain_speed action",
    )
    brake_intensity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Brake intensity (0.0 = no brake, 1.0 = full brake) for brake_vehicle action",
    )
    target_lane_id: str | None = Field(
        default=None,
        description="Target lane ID for lane_change action (e.g., 'lane_0', 'lane_1')",
    )

    # Navigation action parameters
    navigation_behavior: str | None = Field(
        default="normal",
        description="Behavior for navigation agent: cautious, normal, or aggressive",
    )
    destination_x: float | None = Field(
        default=None, description="Destination X coordinate for set_destination action"
    )
    destination_y: float | None = Field(
        default=None, description="Destination Y coordinate for set_destination action"
    )
    destination_z: float | None = Field(
        default=None, description="Destination Z coordinate for set_destination action"
    )
    route_steps: int | None = Field(
        default=1,
        ge=1,
        description="Number of steps to follow route in follow_route action",
    )


class CarlaObservation(Observation):
    """
    Observation from CARLA environment.

    For text-only mode, provides ground truth scene description.
    """

    # Scene description (text-only mode)
    scene_description: str = Field(
        default="", description="Natural language scene description"
    )

    # Vehicle state
    speed_kmh: float = Field(default=0.0, description="Current speed in km/h")
    location: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Vehicle location (x, y, z)"
    )
    rotation: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Vehicle rotation (pitch, yaw, roll)"
    )

    # Navigation/Goal info (for maze and navigation scenarios)
    goal_distance: float | None = Field(
        default=None, description="Distance to goal in meters (if goal is set)"
    )
    goal_direction: str | None = Field(
        default=None, description="Direction to goal: forward, left, right, or behind"
    )

    # Lane info
    current_lane: str = Field(default="unknown", description="Current lane identifier")

    # Nearby actors (for decision-making)
    nearby_actors: list[dict[str, Any]] = Field(
        default_factory=list, description="Nearby actors with distances"
    )

    # Collision detection
    collision_detected: bool = Field(
        default=False, description="Whether collision occurred"
    )
    collision_intensity: float = Field(
        default=0.0, description="Collision force intensity"
    )
    collided_with: str | None = Field(
        default=None, description="ID of actor collided with"
    )

    # Scenario info
    scenario_name: str = Field(default="", description="Name of current scenario")
    simulation_time: float = Field(
        default=0.0, description="Simulation time in seconds"
    )
    step_number: int = Field(default=0, description="Current step number")

    # Episode termination (override done from base Observation)
    done_reason: str = Field(default="", description="Reason for episode termination")

    # Rubric reward for RL training (computed by the rubric, may differ from raw reward)
    rubric_reward: float | None = Field(
        default=None, description="Reward computed by the rubric for RL training"
    )

    # Camera capture (only populated when capture_image action is used)
    camera_image: str | None = Field(
        default=None, description="Base64-encoded JPEG image from front-facing camera"
    )


class CarlaState(State):
    """
    Episode state for CARLA environment.
    """

    # Scenario configuration
    scenario_name: str = Field(
        default="default", description="Name of current scenario"
    )
    town: str = Field(default="Town10HD_Opt", description="CARLA town/map name")
    weather: str = Field(default="ClearNoon", description="Weather preset")

    # Episode metrics
    total_distance: float = Field(
        default=0.0, description="Total distance traveled (meters)"
    )
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    simulation_time: float = Field(
        default=0.0, description="Total simulation time (seconds)"
    )

    # Action tracking metrics
    num_turns: int = Field(default=0, description="Number of steps taken in episode")
    total_tool_calls: int = Field(
        default=0, description="Total number of actions executed"
    )
    tool_call_counts: dict[str, int] = Field(
        default_factory=dict, description="Count of each action type executed"
    )
    is_truncated: bool = Field(
        default=False, description="Whether episode was truncated (max steps)"
    )

    # Movement metrics
    average_speed: float = Field(default=0.0, description="Average speed in km/h")
    max_speed: float = Field(default=0.0, description="Maximum speed reached in km/h")

    # Collision history
    collisions: list[dict[str, Any]] = Field(
        default_factory=list, description="List of collision events"
    )
    collisions_count: int = Field(default=0, description="Total number of collisions")
    collision_intensity_total: float = Field(
        default=0.0, description="Sum of all collision intensities"
    )

    # Scenario-specific data
    scenario_data: dict[str, Any] = Field(
        default_factory=dict, description="Scenario-specific data"
    )
