# SPDX-License-Identifier: BSD-3-Clause

"""Container rubrics for composing reward computations.

These containers provide common aggregation patterns for rubrics,
similar to how PyTorch provides nn.Sequential alongside nn.Module.

See RFC 004 for full design: rfcs/004-rubrics.md
"""

import asyncio
import inspect
from typing import Any, Dict, Iterator, List, Mapping, Tuple, Union

from openenv.core.rubrics.base import Rubric


def _in_async_context() -> bool:
    """Check if we're currently in an async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class Sequential(Rubric):
    """Run rubrics in order, fail-fast on zero.

    Runs child rubrics in order. If any returns 0, stops immediately
    and returns 0. This implements hierarchical gating patterns where
    syntax checks run before execution checks.

    Examples:

        ```python
        rubric = Sequential(
            Gate(Compiles()),
            Gate(PassesTests(), threshold=0.5),
            WeightedSum([PassesTests(), StyleRubric()], weights=[0.7, 0.3])
        )
        ```
    """

    def __init__(self, *rubrics: Rubric):
        """Initialize with rubrics to run in sequence.

        Args:
            *rubrics (`Rubric`):
                Rubrics to run in order. Stops and returns 0 if any child returns 0.
        """
        super().__init__()
        for i, rubric in enumerate(rubrics):
            setattr(self, f"rubric_{i}", rubric)
        self._rubric_list = list(rubrics)

    def forward(self, action: Any, observation: Any) -> float:
        """Run rubrics in order, return 0 if any returns 0. Sync version."""
        result = 1.0
        for rubric in self._rubric_list:
            score = rubric(action, observation)
            if score == 0.0:
                return 0.0
            result = score
        return result

    def __call__(self, action: Any, observation: Any):
        """Override to choose sync or async path based on children."""
        # Empty case - check if in async context
        if not self._rubric_list:
            if _in_async_context():
                return self._empty_async(action, observation)
            else:
                self._run_forward_pre_hooks(action, observation)
                return self._finish_forward(action, observation, 1.0)

        # Pre-hooks run before any child is evaluated
        ran = self._run_leading_sync_pre_hooks(action, observation)

        # Call first rubric to see if it's async
        first_result = self._rubric_list[0](action, observation)
        if inspect.iscoroutine(first_result):
            # At least one child is async, use async path
            return self._call_async_detected(action, observation, first_result, ran)
        else:
            # Continue with sync path
            if first_result == 0.0:
                self._run_forward_pre_hooks(action, observation, start=ran)
                return self._finish_forward(action, observation, 0.0)

            final_result = first_result
            for i, rubric in enumerate(self._rubric_list[1:], start=1):
                score = rubric(action, observation)
                if inspect.iscoroutine(score):
                    # Found async mid-way, switch to async
                    # We already called rubric at index i, so pass the coroutine and remaining rubrics
                    return self._call_async_mid(
                        action,
                        observation,
                        score,
                        self._rubric_list[i + 1 :],
                        ran,
                    )
                if score == 0.0:
                    self._run_forward_pre_hooks(action, observation, start=ran)
                    return self._finish_forward(action, observation, 0.0)
                final_result = score

            # All sync - check if in async context
            if _in_async_context():
                return self._wrap_sync_result(action, observation, final_result, ran)
            else:
                self._run_forward_pre_hooks(action, observation, start=ran)
                return self._finish_forward(action, observation, final_result)

    async def _empty_async(self, action, observation):
        """Async path for empty sequential."""
        await self._run_forward_pre_hooks_async(action, observation)
        return await self._finish_forward_async(action, observation, 1.0)

    async def _wrap_sync_result(self, action, observation, result, ran=0):
        """Wrap sync result for async context."""
        await self._run_forward_pre_hooks_async(action, observation, start=ran)
        return await self._finish_forward_async(action, observation, result)

    async def _call_async_detected(self, action, observation, first_coro, ran=0):
        """Async path when first child is async."""
        await self._run_forward_pre_hooks_async(action, observation, start=ran)

        result = await first_coro
        if result == 0.0:
            return await self._finish_forward_async(action, observation, result)

        for rubric in self._rubric_list[1:]:
            score = rubric(action, observation)
            if inspect.iscoroutine(score):
                score = await score
            if score == 0.0:
                return await self._finish_forward_async(action, observation, 0.0)
            result = score

        return await self._finish_forward_async(action, observation, result)

    async def _call_async_mid(
        self, action, observation, first_async_coro, remaining, ran=0
    ):
        """Async path when async detected mid-execution."""
        await self._run_forward_pre_hooks_async(action, observation, start=ran)

        # Await the first async rubric (already called)
        result = await first_async_coro
        if result == 0.0:
            return await self._finish_forward_async(action, observation, 0.0)

        # Continue with remaining rubrics
        for rubric in remaining:
            score = rubric(action, observation)
            if inspect.iscoroutine(score):
                score = await score
            if score == 0.0:
                return await self._finish_forward_async(action, observation, 0.0)
            result = score

        return await self._finish_forward_async(action, observation, result)

    def __len__(self) -> int:
        return len(self._rubric_list)

    def __getitem__(self, index: int) -> Rubric:
        return self._rubric_list[index]


class Gate(Rubric):
    """Threshold wrapper - returns 0 if child score is below threshold.

    Useful for hard constraints like "must pass 50% of tests".

    Examples:

        ```python
        rubric = Gate(PassesTests(), threshold=0.5)
        # Returns PassesTests() score if >= 0.5, else 0.0
        ```
    """

    def __init__(self, rubric: Rubric, threshold: float = 1.0):
        """Initialize with a rubric and threshold.

        Args:
            rubric (`Rubric`):
                The rubric to gate.
            threshold (`float`, *optional*, defaults to `1.0`):
                Minimum score required. If child returns less than this, Gate returns 0.
        """
        super().__init__()
        self.rubric = rubric
        self.threshold = threshold

    def forward(self, action: Any, observation: Any) -> float:
        """Return child score if >= threshold, else 0. Sync version."""
        score = self.rubric(action, observation)
        if score < self.threshold:
            return 0.0
        return score

    def __call__(self, action: Any, observation: Any):
        """Override to handle async child."""
        # Pre-hooks run before the child is evaluated
        ran = self._run_leading_sync_pre_hooks(action, observation)

        # Call child
        score = self.rubric(action, observation)

        if inspect.iscoroutine(score):
            # Child is async
            return self._call_async(action, observation, score, ran)
        else:
            # Child is sync
            self._run_forward_pre_hooks(action, observation, start=ran)
            result = 0.0 if score < self.threshold else score
            return self._finish_forward(action, observation, result)

    async def _call_async(self, action, observation, score_coro, ran=0):
        """Async path."""
        await self._run_forward_pre_hooks_async(action, observation, start=ran)

        score = await score_coro
        result = 0.0 if score < self.threshold else score
        return await self._finish_forward_async(action, observation, result)


class WeightedSum(Rubric):
    """Weighted combination of child rubrics.

    Standard aggregation pattern for multi-criteria evaluation.

    Examples:

        ```python
        rubric = WeightedSum(
            [PassesTests(), StyleRubric()],
            weights=[0.7, 0.3]
        )
        ```
    """

    def __init__(self, rubrics: List[Rubric], weights: List[float]):
        """Initialize with rubrics and weights.

        Args:
            rubrics (`list[``Rubric``]`):
                List of rubrics to combine.
            weights (`list[float]`):
                Weight for each rubric. Must sum to 1.0.

        Raises:
            ValueError: If lengths don't match or weights don't sum to 1.0.
        """
        super().__init__()
        if len(rubrics) != len(weights):
            raise ValueError(
                f"Number of rubrics ({len(rubrics)}) must match "
                f"number of weights ({len(weights)})"
            )
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        for i, rubric in enumerate(rubrics):
            setattr(self, f"rubric_{i}", rubric)
        self._rubric_list = list(rubrics)
        self._weights = list(weights)

    def forward(self, action: Any, observation: Any) -> float:
        """Return weighted sum of child scores. Sync version."""
        total = 0.0
        for rubric, weight in zip(self._rubric_list, self._weights):
            score = rubric(action, observation)
            total += score * weight
        return total

    def __call__(self, action: Any, observation: Any):
        """Override to handle async children with parallel execution."""
        # Pre-hooks run before any child is evaluated
        ran = self._run_leading_sync_pre_hooks(action, observation)

        # Call all rubrics
        results = [rubric(action, observation) for rubric in self._rubric_list]

        # Check if any are async
        has_async = any(inspect.iscoroutine(r) for r in results)

        if has_async:
            # Use async path
            return self._call_async(action, observation, results, ran)
        else:
            # Sync path
            self._run_forward_pre_hooks(action, observation, start=ran)
            total = 0.0
            for score, weight in zip(results, self._weights):
                total += score * weight
            return self._finish_forward(action, observation, total)

    async def _call_async(self, action, observation, results, ran=0):
        """Async path with parallel execution."""
        await self._run_forward_pre_hooks_async(action, observation, start=ran)

        # Separate sync and async results
        async_tasks = []
        async_indices = []
        scores = [None] * len(results)

        for i, result in enumerate(results):
            if inspect.iscoroutine(result):
                async_tasks.append(result)
                async_indices.append(i)
            else:
                scores[i] = result

        # Await all async tasks in parallel
        if async_tasks:
            async_scores = await asyncio.gather(*async_tasks)
            for i, score in zip(async_indices, async_scores):
                scores[i] = score

        # Compute weighted sum
        total = 0.0
        for score, weight in zip(scores, self._weights):
            total += score * weight

        return await self._finish_forward_async(action, observation, total)

    @property
    def weights(self) -> List[float]:
        """Get the weights (read-only copy)."""
        return list(self._weights)


class RubricList(Rubric):
    """Container for dynamic lists of rubrics.

    Analogous to nn.ModuleList. Does not define aggregation - use within
    a parent rubric that implements custom logic.

    Examples:

        ```python
        class MultiGameRubric(Rubric):
            def __init__(self, games: List[str]):
                super().__init__()
                self.games = RubricList([GameRubric(g) for g in games])

            def forward(self, action, obs) -> float:
                return self.games[obs.game_index](action, obs)
        ```
    """

    def __init__(self, rubrics: List[Rubric] = None):
        """Initialize with optional list of rubrics.

        Args:
            rubrics (`list[``Rubric``]`, *optional*):
                List of rubrics to start with.
        """
        super().__init__()
        self._rubrics: List[Rubric] = []
        if rubrics is not None:
            for i, rubric in enumerate(rubrics):
                self.append(rubric)

    def forward(self, action: Any, observation: Any) -> float:
        """RubricList does not define aggregation - override in parent."""
        raise NotImplementedError(
            "RubricList.forward() is not implemented. "
            "Use RubricList within a parent rubric that defines aggregation."
        )

    def append(self, rubric: Rubric) -> None:
        """Add a rubric to the list."""
        index = len(self._rubrics)
        setattr(self, f"rubric_{index}", rubric)
        self._rubrics.append(rubric)

    def extend(self, rubrics: List[Rubric]) -> None:
        """Add multiple rubrics to the list."""
        for rubric in rubrics:
            self.append(rubric)

    def __len__(self) -> int:
        return len(self._rubrics)

    def __getitem__(self, index: int) -> Rubric:
        return self._rubrics[index]

    def __iter__(self) -> Iterator[Rubric]:
        return iter(self._rubrics)


class RubricDict(Rubric):
    """Container for named rubrics with keyed access.

    Analogous to nn.ModuleDict. Enables keyed access for multi-task
    environments where different tasks require different rubrics.

    Examples:

        ```python
        class AtariRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.games = RubricDict({
                    "pong": PongRubric(),
                    "breakout": BreakoutRubric(),
                    "space_invaders": SpaceInvadersRubric(),
                })

            def forward(self, action, obs) -> float:
                return self.games[obs.game_id](action, obs)

        # Access: env.rubric.games["pong"]
        ```
    """

    def __init__(self, rubrics: Dict[str, Rubric] = None):
        """Initialize with optional dictionary of rubrics.

        Args:
            rubrics (`dict[str, ``Rubric``]`, *optional*):
                Dictionary mapping names to rubrics.
        """
        super().__init__()
        self._rubric_dict: Dict[str, Rubric] = {}
        if rubrics is not None:
            for name, rubric in rubrics.items():
                self[name] = rubric

    def forward(self, action: Any, observation: Any) -> float:
        """RubricDict does not define aggregation - override in parent."""
        raise NotImplementedError(
            "RubricDict.forward() is not implemented. "
            "Use RubricDict within a parent rubric that defines aggregation."
        )

    def __setitem__(self, key: str, rubric: Rubric) -> None:
        """Add a rubric with the given key."""
        setattr(self, key, rubric)
        self._rubric_dict[key] = rubric

    def __getitem__(self, key: str) -> Rubric:
        """Get rubric by key."""
        return self._rubric_dict[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._rubric_dict

    def __len__(self) -> int:
        return len(self._rubric_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._rubric_dict)

    def keys(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._rubric_dict.keys())

    def values(self) -> Iterator[Rubric]:
        """Iterate over rubrics."""
        return iter(self._rubric_dict.values())

    def items(self) -> Iterator[Tuple[str, Rubric]]:
        """Iterate over (key, rubric) pairs."""
        return iter(self._rubric_dict.items())

    def update(self, rubrics: Union[Dict[str, Rubric], Mapping[str, Rubric]]) -> None:
        """Update with rubrics from a dictionary."""
        for name, rubric in rubrics.items():
            self[name] = rubric
