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

from dataclasses import asdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

import reasoning_gym
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from reasoning_gym.arithmetic.chain_sum import ChainSumCurriculum
from reasoning_gym.coaching.base_curriculum import (
    DefaultCurriculumContext,
    RangeAttributeMode,
)
from reasoning_gym.composite import DatasetSpec

try:
    from models import ReasoningGymAction, ReasoningGymObservation
except ImportError:
    from ..models import ReasoningGymAction, ReasoningGymObservation


DEFAULT_DATASET_NAME = "leg_counting"
DEFAULT_DATASET_SIZE = 1000
DEFAULT_DATASET_SEED = 42

LEVELS = [
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 2),
    (3, 3),
    (4, 3),
    (4, 4),
    (5, 4),
]
TASKS_PER_TIER = {"train": 2000, "test": 100}
SPLIT_SEED_OFFSET = {"train": 0, "test": 500_000_000}


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
        self._final_observation: Optional[ReasoningGymObservation] = None

        # Dataset configuration tracking
        self._dataset_name: Optional[str] = None
        self._dataset_size: Optional[int] = None
        self._dataset_seed: Optional[int] = None
        self._dataset_config: Optional[Dict[str, Any]] = None
        self._dataset_specs: Optional[List[Dict[str, Any]]] = None

    def list_splits(self) -> list[str]:
        """
        List the finite splits of chain-sum tasks.

        Returns:
            `list[str]`: The train and test split names.
        """
        return list(TASKS_PER_TIER)

    def num_tasks(self, split: str) -> int:
        """
        Return a split's task count without starting an episode.

        Args:
            split (`str`):
                Split name.

        Returns:
            `int`: Number of tasks across all ten difficulty levels.

        Raises:
            `ValueError`: If the split is unknown.
        """
        if split not in TASKS_PER_TIER:
            raise ValueError(f"unknown split {split!r}")
        return len(LEVELS) * TASKS_PER_TIER[split]

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        """
        Describe a task without generating its question.

        Args:
            split (`str`):
                Split name.
            index (`int`):
                Zero-based position in the split.

        Returns:
            `dict[str, Any]`: Stable id, split, index, group and difficulty.

        Raises:
            `ValueError`: If the split is unknown.
            `IndexError`: If the index is outside the split.
        """
        n = self.num_tasks(split)
        if not 0 <= index < n:
            raise IndexError(index)
        return {
            "id": f"chain_sum-{split}-{index}",
            "split": split,
            "index": index,
            "group": "chain_sum",
            "difficulty": index % len(LEVELS),
        }

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Describe a slice of the split, with Python slice semantics.

        Args:
            split (`str`):
                Split name.
            start (`int`, *optional*):
                Inclusive start, including negative offsets from the end.
            stop (`int`, *optional*):
                Exclusive stop, including negative offsets from the end.

        Returns:
            `list[dict[str, Any]]`: Task specs in split order.

        Raises:
            `ValueError`: If the split is unknown.
        """
        return [
            self.get_task(split, index)
            for index in range(self.num_tasks(split))[slice(start, stop)]
        ]

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        """
        Describe every task in a split.

        Args:
            split (`str`):
                Split name.

        Returns:
            `list[dict[str, Any]]`: Task specs with interleaved difficulty levels.

        Raises:
            `ValueError`: If the split is unknown.
        """
        return self.get_task_range(split)

    def reset(
        self,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        dataset_specs: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        size: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: str = "train",
        index: int | None = None,
    ) -> ReasoningGymObservation:
        """
        Reset the environment and return the next question.

        Without `index`, the dataset options build a new dataset; without them, the
        current dataset is reused, or a default `leg_counting` dataset (fixed seed
        and size) is created when there is none. With `index`, the question is the
        chain-sum task at that position of `split`.

        Args:
            dataset_name (`str`, *optional*):
                Name of a reasoning_gym dataset, such as `"leg_counting"`, or
                `"composite"`.
            dataset_config (`Dict[str, Any]`, *optional*):
                Options of a dataset that is not composite.
            dataset_specs (`List[Dict[str, Any]]`, *optional*):
                The datasets of a composite dataset; required when `dataset_name` is
                `"composite"`.
            seed (`int`, *optional*):
                Seed of the new dataset; required with `dataset_name`. With `index`,
                it picks a variant of the task.
            size (`int`, *optional*):
                Number of questions of the new dataset; required with
                `dataset_name`.
            episode_id (`str`, *optional*):
                Id of the episode; a new one is generated without it.
            split (`str`, *optional*, defaults to `"train"`):
                Split of the selected chain-sum task; used only with `index`.
            index (`int`, *optional*):
                Task position. Selects a chain-sum level and question, using
                `(seed or 0) * 1_000_000_000 + split_offset + index`. Cannot be
                combined with dataset options.

        Returns:
            `ReasoningGymObservation`: The question, not done and not scored.

        Raises:
            `ValueError`: If `index` comes with dataset options or an unknown split,
                if `seed` or `size` is missing with `dataset_name`, or if a
                composite dataset has no specs.
            `IndexError`: If `index` is outside the split.
        """
        if index is not None:
            if any(
                option is not None
                for option in (dataset_name, dataset_config, dataset_specs, size)
            ):
                raise ValueError("index cannot be combined with dataset options")
            task = self.get_task(split, index)
            curriculum = ChainSumCurriculum()
            terms, digits = LEVELS[task["difficulty"]]
            curriculum.set_attr_level("num_terms", terms)
            curriculum.set_attr_level("num_digits", digits)
            config = asdict(
                curriculum.generate_configuration(
                    context=DefaultCurriculumContext(
                        mode=RangeAttributeMode.UPPER_BOUND
                    )
                )
            )
            config.pop("seed")
            config.pop("size")
            dataset_name = "chain_sum"
            dataset_config = config
            seed = (seed or 0) * 1_000_000_000 + SPLIT_SEED_OFFSET[split] + index
            size = 1

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
        self._final_observation = None

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
        # The episode is single-step: once an answer has been scored (and the
        # correct answer revealed), later steps return the same final
        # observation and are not scored again.
        if self._final_observation is not None:
            return self._final_observation.model_copy(deep=True)

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

        observation = ReasoningGymObservation(
            question=None,
            score=score,
            correct_answer=correct_answer,
            done=True,
            reward=reward,
            dataset_metadata=metadata if metadata else None,
        )
        self._final_observation = observation.model_copy(deep=True)
        return observation

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
