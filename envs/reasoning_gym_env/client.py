# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reasoning Gym Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ReasoningGymAction, ReasoningGymObservation


class ReasoningGymEnv(EnvClient[ReasoningGymAction, ReasoningGymObservation, State]):
    """
    Client for the Reasoning Gym Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> from reasoning_gym_env import ReasoningGymEnv, ReasoningGymAction
        >>>
        >>> env = ReasoningGymEnv(base_url="http://localhost:8000")
        >>> # Create dataset with 10 leg_counting questions
        >>> result = env.reset(
        ...     dataset_name='leg_counting',
        ...     dataset_config={"min_animals": 5, "max_animals": 15},
        ...     seed=42,
        ...     size=10
        ... )
        >>> print(f"Question: {result.observation.question}")
        >>>
        >>> # Answer question
        >>> result = env.step(ReasoningGymAction(answer="4"))
        >>> print(f"Score: {result.observation.score}")
        >>> print(f"Correct: {result.observation.correct_answer}")
        >>>
        >>> # Get next question (reuses dataset)
        >>> result = env.reset()
        >>> print(f"Next question: {result.observation.question}")
        >>> env.close()

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ReasoningGymEnv.from_docker_image("reasoning_gym-env:latest")
        >>> try:
        ...     result = client.reset(
        ...         dataset_name='leg_counting',
        ...         dataset_config={"min_animals": 5, "max_animals": 15},
        ...         seed=42,
        ...         size=5
        ...     )
        ...     result = client.step(ReasoningGymAction(answer="4"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ReasoningGymAction) -> Dict:
        """
        Convert ReasoningGymAction to JSON payload for step message.

        Args:
            action: ReasoningGymAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "answer": action.answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ReasoningGymObservation]:
        """
        Parse server response into StepResult[ReasoningGymObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ReasoningGymObservation
        """
        obs_data = payload.get("observation", {})
        observation = ReasoningGymObservation(
            question=obs_data.get("question"),
            score=obs_data.get("score"),
            correct_answer=obs_data.get("correct_answer"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
            dataset_metadata=obs_data.get("dataset_metadata"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
