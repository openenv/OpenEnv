# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for SUMO-RL Environment.

This module defines the Action, Observation, and State types for traffic
signal control using SUMO (Simulation of Urban MObility).
"""

from typing import Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class SumoAction(Action):
    """
    Action for SUMO traffic signal control environment.

    Represents selecting which traffic light phase to activate next.

    Attributes:
        phase_id: Index of the green phase to activate (0 to num_phases-1)
        ts_id: Traffic signal ID (for multi-agent support, default "0")
    """

    phase_id: int
    ts_id: str = "0"


class SumoObservation(Observation):
    """
    Observation from SUMO traffic signal environment.

    Contains traffic metrics for decision-making.

    Attributes:
        observation: Flattened observation vector containing:
                    - One-hot encoded current phase
                    - Min green flag (binary)
                    - Lane densities (normalized)
                    - Lane queues (normalized)
        observation_shape: Shape of observation for reshaping
        action_mask: List of valid action indices
        sim_time: Current simulation time in seconds
        done: Whether episode is complete
        reward: Reward from last action (None on reset)
        metadata: Additional info (system metrics, etc.)
    """

    observation: List[float] = Field(default_factory=list)
    observation_shape: List[int] = Field(default_factory=list)
    action_mask: List[int] = Field(default_factory=list)
    sim_time: float = 0.0
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict = Field(default_factory=dict)


class SumoState(State):
    """
    State of SUMO traffic signal environment.

    Tracks both configuration and runtime state.

    Configuration attributes:
        net_file: Path to SUMO network file (.net.xml)
        route_file: Path to SUMO route file (.rou.xml)
        num_seconds: Total simulation duration in seconds
        delta_time: Seconds between agent actions
        yellow_time: Duration of yellow phase in seconds
        min_green: Minimum green time per phase in seconds
        max_green: Maximum green time per phase in seconds
        reward_fn: Name of reward function used

    Runtime attributes:
        episode_id: Unique episode identifier
        step_count: Number of steps taken in episode
        sim_time: Current simulation time in seconds
        total_vehicles: Total number of vehicles in simulation
        total_waiting_time: Cumulative waiting time across all vehicles
    """

    # Episode tracking
    episode_id: str = ""
    step_count: int = 0

    # SUMO configuration
    net_file: str = ""
    route_file: str = ""
    num_seconds: int = 20000
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50
    reward_fn: str = "diff-waiting-time"

    # Runtime metrics
    sim_time: float = 0.0
    total_vehicles: int = 0
    total_waiting_time: float = 0.0
    mean_waiting_time: float = 0.0
    mean_speed: float = 0.0
