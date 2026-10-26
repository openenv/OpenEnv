# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client for SUMO-RL environment.

This module provides a client to interact with the SUMO traffic signal
control environment via WebSocket for persistent sessions.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SumoAction, SumoObservation, SumoState


class SumoRLEnv(EnvClient[SumoAction, SumoObservation, SumoState]):
    """
    Client for SUMO-RL traffic signal control environment.

    This client maintains a persistent WebSocket connection to a SUMO
    environment server to control traffic signals using reinforcement learning.

    Example:
        >>> # Start container and connect
        >>> env = SumoRLEnv.from_docker_image("sumo-rl-env:latest")
        >>> try:
        ...     # Reset environment
        ...     result = env.reset()
        ...     print(f"Observation shape: {result.observation.observation_shape}")
        ...     print(f"Action space: {result.observation.action_mask}")
        ...
        ...     # Take action
        ...     result = env.step(SumoAction(phase_id=1))
        ...     print(f"Reward: {result.reward}, Done: {result.done}")
        ...
        ...     # Get state
        ...     state = env.state()
        ...     print(f"Sim time: {state.sim_time}, Total vehicles: {state.total_vehicles}")
        ... finally:
        ...     env.close()

    Example with custom network:
        >>> # Use custom SUMO network via volume mount
        >>> env = SumoRLEnv.from_docker_image(
        ...     "sumo-rl-env:latest",
        ...     port=8000,
        ...     volumes={
        ...         "/path/to/my/nets": {"bind": "/nets", "mode": "ro"}
        ...     },
        ...     environment={
        ...         "SUMO_NET_FILE": "/nets/my-network.net.xml",
        ...         "SUMO_ROUTE_FILE": "/nets/my-routes.rou.xml",
        ...     }
        ... )

    Example with configuration:
        >>> # Adjust simulation parameters
        >>> env = SumoRLEnv.from_docker_image(
        ...     "sumo-rl-env:latest",
        ...     environment={
        ...         "SUMO_NUM_SECONDS": "10000",
        ...         "SUMO_DELTA_TIME": "10",
        ...         "SUMO_REWARD_FN": "queue",
        ...         "SUMO_SEED": "123",
        ...     }
        ... )
    """

    def _step_payload(self, action: SumoAction) -> Dict[str, Any]:
        """
        Convert SumoAction to JSON payload for HTTP request.

        Args:
            action: SumoAction containing phase_id to execute.

        Returns:
            Dictionary payload for step endpoint.
        """
        return {
            "phase_id": action.phase_id,
            "ts_id": action.ts_id,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SumoObservation]:
        """
        Parse step result from HTTP response JSON.

        Args:
            payload: JSON response from step endpoint.

        Returns:
            StepResult containing SumoObservation.
        """
        obs_data = payload.get("observation", {})

        observation = SumoObservation(
            observation=obs_data.get("observation", []),
            observation_shape=obs_data.get("observation_shape", []),
            action_mask=obs_data.get("action_mask", []),
            sim_time=obs_data.get("sim_time", 0.0),
            # Read from the envelope, where `serialize_observation` puts these,
            # so `result.observation.reward` agrees with `result.reward`.
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SumoState:
        """
        Parse state from HTTP response JSON.

        Args:
            payload: JSON response from state endpoint.

        Returns:
            SumoState object.
        """
        return SumoState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            net_file=payload.get("net_file", ""),
            route_file=payload.get("route_file", ""),
            num_seconds=payload.get("num_seconds", 20000),
            delta_time=payload.get("delta_time", 5),
            yellow_time=payload.get("yellow_time", 2),
            min_green=payload.get("min_green", 5),
            max_green=payload.get("max_green", 50),
            reward_fn=payload.get("reward_fn", "diff-waiting-time"),
            sim_time=payload.get("sim_time", 0.0),
            total_vehicles=payload.get("total_vehicles", 0),
            total_waiting_time=payload.get("total_waiting_time", 0.0),
            mean_waiting_time=payload.get("mean_waiting_time", 0.0),
            mean_speed=payload.get("mean_speed", 0.0),
        )
