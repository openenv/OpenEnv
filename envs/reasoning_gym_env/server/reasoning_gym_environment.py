# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reasoning Gym Environment Implementation.

Integrates the Reasoning Gym library to provide single-step reasoning tasks.
Each episode presents one question, the agent submits an answer, and receives a score.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import reasoning_gym
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from reasoning_gym.composite import DatasetSpec

try:
    from models import ReasoningGymAction, ReasoningGymObservation
except ImportError:
    from ..models import ReasoningGymAction, ReasoningGymObservation


DEFAULT_DATASET_NAME = "leg_counting"
DEFAULT_DATASET_SIZE = 1000
DEFAULT_DATASET_SEED = 42


class ReasoningGymEnvironment(Environment):
    """
    Reasoning Gym environment for single-step reasoning tasks.

    Each episode presents one question from a reasoning gym dataset. The agent
    submits an answer and receives a score. The dataset persists across resets
    unless configuration parameters change.

    Design:
    - Single-step episodes: reset() provides question, step() validates and returns done=True
    - Dataset persistence: Dataset created once and reused until config changes
    - Sequential iteration: Questions sampled sequentially using an iterator
    - Parameter-driven reset: reset() with params rebuilds dataset, without params reuses it

    Example:
        >>> env = ReasoningGymEnvironment()
        >>> # Create dataset
        >>> obs = env.reset(dataset_name='leg_counting', dataset_config={"min_animals": 5}, seed=42, size=10)
        >>> print(obs.question)  # "How many legs does a cat have?"
        >>>
        >>> # Answer question
        >>> obs = env.step(ReasoningGymAction(answer="4"))
        >>> print(obs.score)  # 1.0
        >>> print(obs.done)  # True
        >>>
        >>> # Get next question (reuses dataset)
        >>> obs = env.reset()
        >>> print(obs.question)  # Next question from same dataset
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the reasoning gym environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Dataset management
        self._dataset = None
        self._dataset_iterator = None
        self._current_entry = None

        # Dataset configuration tracking
        self._dataset_name: Optional[str] = None
        self._dataset_size: Optional[int] = None
        self._dataset_seed: Optional[int] = None
        self._dataset_config: Optional[Dict[str, Any]] = None
        self._dataset_specs: Optional[List[Dict[str, Any]]] = None

    def reset(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        dataset_specs: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        size: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> ReasoningGymObservation:
        """
        Reset the environment and get the next question.

        If no parameters are provided and no dataset exists, creates a default 'leg_counting' dataset

        Args:
            dataset_name: Name of dataset (e.g., 'leg_counting', 'composite').
                         If None and dataset exists, reuses existing dataset.
                         If None and no dataset exists, creates default 'leg_counting'.
            dataset_config: For non-composite datasets, config dict with task-specific params.
                          Required if dataset_name is provided and not 'composite'.
            dataset_specs: For composite datasets, list of dicts defining structure.
                          Required if dataset_name is 'composite'.
            seed: Random seed for dataset generation.
                 Required if dataset_name is provided.
                 Defaults to 42 when creating default dataset.
            size: Number of questions in dataset.
                 Required if dataset_name is provided.
                 Defaults to 100 when creating default dataset.
            episode_id: Optional episode ID

        Returns:
            ReasoningGymObservation with question and metadata

        Raises:
            ValueError: If parameters are invalid or missing
        """
        # Check if we need to rebuild dataset
        if dataset_name is not None:
            # Validate parameters
            if seed is None or size is None:
                raise ValueError(
                    "seed and size must be provided when dataset_name is specified"
                )

            if dataset_name == "composite":
                if dataset_specs is None:
                    raise ValueError(
                        "dataset_specs must be provided for composite datasets"
                    )
                if not dataset_specs:
                    raise ValueError(
                        "dataset_specs cannot be empty for composite datasets"
                    )
            else:
                if dataset_config is None:
                    dataset_config = {}  # okay to use default args in the config

            # Build new dataset
            if dataset_name == "composite":
                # Composite dataset - convert dicts to DatasetSpec objects
                specs = [
                    DatasetSpec(
                        name=spec["name"],
                        weight=spec.get("weight", 1),
                        config=spec.get("config", {}),
                    )
                    for spec in dataset_specs
                ]
                self._dataset = reasoning_gym.create_dataset(
                    "composite",
                    size=size,
                    seed=seed,
                    datasets=specs,
                )
            else:
                # Simple dataset with config
                self._dataset = reasoning_gym.create_dataset(
                    dataset_name,
                    size=size,
                    seed=seed,
                    **dataset_config,
                )

            # Reset iterator for new dataset
            self._dataset_iterator = iter(self._dataset)

            # Store configuration for tracking
            self._dataset_name = dataset_name
            self._dataset_size = size
            self._dataset_seed = seed
            self._dataset_config = dataset_config
            self._dataset_specs = dataset_specs

        elif self._dataset is None:
            # No dataset exists and none specified - create default
            self._dataset = reasoning_gym.create_dataset(
                name=DEFAULT_DATASET_NAME,
                size=DEFAULT_DATASET_SIZE,
                seed=DEFAULT_DATASET_SEED,
            )
            self._dataset_iterator = iter(self._dataset)
            self._dataset_name = DEFAULT_DATASET_NAME
            self._dataset_size = DEFAULT_DATASET_SIZE
            self._dataset_seed = DEFAULT_DATASET_SEED
            self._dataset_config = {}
            self._dataset_specs = None

        # Get next question from iterator
        try:
            self._current_entry = next(self._dataset_iterator)
        except StopIteration:
            # Iterator exhausted - restart it
            self._dataset_iterator = iter(self._dataset)
            self._current_entry = next(self._dataset_iterator)

        question = self._current_entry["question"]

        # Update episode state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return ReasoningGymObservation(
            question=question,
            score=None,
            correct_answer=None,
            done=False,
            reward=0.0,
            dataset_metadata=None,
        )

    def step(self, action: ReasoningGymAction) -> ReasoningGymObservation:  # type: ignore[override]
        """
        Execute a step by scoring the agent's answer.

        Args:
            action: ReasoningGymAction containing the agent's answer

        Returns:
            ReasoningGymObservation with score and correct answer, always done=True
        """
        self._state.step_count += 1

        # Validate current state
        if self._current_entry is None:
            return ReasoningGymObservation(
                question=None,
                score=None,
                correct_answer=None,
                dataset_metadata=None,
                done=True,
                reward=0.0,
            )

        # Score the answer
        answer = action.answer
        score = self._dataset.score_answer(answer, self._current_entry)
        reward = float(score)

        # Extract correct answer and metadata
        correct_answer = self._current_entry["answer"]
        metadata = self._current_entry.get("metadata", {})

        return ReasoningGymObservation(
            question=None,
            score=score,
            correct_answer=correct_answer,
            done=True,
            reward=reward,
            dataset_metadata=metadata if metadata else None,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
