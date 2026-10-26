# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SUMO-RL Environment Server Implementation.

This module wraps the SUMO-RL SumoEnvironment and exposes it
via the OpenEnv Environment interface for traffic signal control.
"""

import os
import uuid
from typing import Any, Dict

# Set SUMO_HOME before importing sumo_rl
os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")

from openenv.core.env_server import Action, Environment, Observation

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import SumoAction, SumoObservation, SumoState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import SumoAction, SumoObservation, SumoState

# Import SUMO-RL
try:
    from sumo_rl import SumoEnvironment as BaseSumoEnv
except ImportError as e:
    raise ImportError(
        "sumo-rl is not installed. Please install it with: pip install sumo-rl"
    ) from e


def _json_safe(value: Any) -> Any:
    """Convert library-specific values into JSON-serializable Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(value)


class SumoEnvironment(Environment):
    """
    SUMO-RL Environment wrapper for OpenEnv.

    This environment wraps the SUMO traffic signal control environment
    for single-agent reinforcement learning.

    Args:
        net_file: Path to SUMO network file (.net.xml)
        route_file: Path to SUMO route file (.rou.xml)
        num_seconds: Simulation duration in seconds (default: 20000)
        delta_time: Seconds between agent actions (default: 5)
        yellow_time: Yellow phase duration in seconds (default: 2)
        min_green: Minimum green time in seconds (default: 5)
        max_green: Maximum green time in seconds (default: 50)
        reward_fn: Reward function name (default: "diff-waiting-time")
        sumo_seed: Random seed for reproducibility (default: 42)

    Example:
        >>> env = SumoEnvironment(
        ...     net_file="/app/nets/single-intersection.net.xml",
        ...     route_file="/app/nets/single-intersection.rou.xml"
        ... )
        >>> obs = env.reset()
        >>> print(obs.observation_shape)
        >>> obs = env.step(SumoAction(phase_id=1))
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        num_seconds: int = 20000,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        reward_fn: str = "diff-waiting-time",
        sumo_seed: int = 42,
    ):
        """Initialize SUMO traffic signal environment."""
        super().__init__()

        # Store configuration
        self.net_file = net_file
        self.route_file = route_file
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed

        # Create SUMO environment (single-agent mode)
        # Key settings:
        # - use_gui=False: No GUI in Docker
        # - single_agent=True: Returns single obs/reward (not dict)
        # - sumo_warnings=False: Suppress SUMO warnings
        # - out_csv_name=None: Don't write CSV files
        self.env = BaseSumoEnv(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            single_agent=True,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            reward_fn=reward_fn,
            sumo_seed=sumo_seed,
            sumo_warnings=False,
            out_csv_name=None,  # Disable CSV output
            add_system_info=True,
            add_per_agent_info=False,
        )

        # Initialize state
        self._state = SumoState(
            net_file=net_file,
            route_file=route_file,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            reward_fn=reward_fn,
        )

        self._last_info = {}

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial SumoObservation for the agent.
        """
        # Reset SUMO simulation
        obs, info = self.env.reset()

        # Update state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.sim_time = 0.0

        # Store info for metadata
        self._last_info = info

        return self._make_observation(obs, reward=None, done=False, info=info)

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: SumoAction containing the phase_id to execute.

        Returns:
            SumoObservation after action execution.

        Raises:
            ValueError: If action is not a SumoAction.
        """
        if not isinstance(action, SumoAction):
            raise ValueError(f"Expected SumoAction, got {type(action)}")

        # Validate phase_id
        num_phases = self.env.action_space.n
        if action.phase_id < 0 or action.phase_id >= num_phases:
            raise ValueError(
                f"Invalid phase_id: {action.phase_id}. "
                f"Valid range: [0, {num_phases - 1}]"
            )

        # Execute action in SUMO
        # Returns: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action.phase_id)
        done = terminated or truncated

        # Update state
        self._state.step_count += 1
        self._state.sim_time = info.get("step", 0.0)
        self._state.total_vehicles = info.get("system_total_running", 0)
        self._state.total_waiting_time = info.get("system_total_waiting_time", 0.0)
        self._state.mean_waiting_time = info.get("system_mean_waiting_time", 0.0)
        self._state.mean_speed = info.get("system_mean_speed", 0.0)

        # Store info for metadata
        self._last_info = info

        return self._make_observation(obs, reward=reward, done=done, info=info)

    @property
    def state(self) -> SumoState:
        """Get current environment state."""
        return self._state

    def _make_observation(
        self, obs: Any, reward: float, done: bool, info: Dict
    ) -> SumoObservation:
        """
        Create SumoObservation from SUMO environment output.

        Args:
            obs: Observation array from SUMO environment
            reward: Reward value (None on reset)
            done: Whether episode is complete
            info: Info dictionary from SUMO environment

        Returns:
            SumoObservation for the agent.
        """
        # Convert observation to list
        if hasattr(obs, "tolist"):
            obs_list = obs.tolist()
        else:
            obs_list = list(obs)

        # Get action mask (all actions valid in SUMO-RL)
        num_phases = int(self.env.action_space.n)
        action_mask = list(range(num_phases))

        # Extract system metrics for metadata
        system_info = {k: v for k, v in info.items() if k.startswith("system_")}

        # Create observation
        return SumoObservation(
            observation=obs_list,
            observation_shape=[len(obs_list)],
            action_mask=action_mask,
            sim_time=info.get("step", 0.0),
            done=done,
            reward=reward,
            metadata={
                "num_green_phases": num_phases,
                "system_info": _json_safe(system_info),
            },
        )
