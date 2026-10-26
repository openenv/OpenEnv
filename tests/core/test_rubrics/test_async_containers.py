# SPDX-License-Identifier: BSD-3-Clause

"""Tests for async container rubrics: Sequential, Gate, WeightedSum.

This test file verifies that container rubrics work with async forward():
- Sequential with async children
- Gate with async child
- WeightedSum with async children
- Parallel execution optimization
"""

import asyncio
from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import Gate, Sequential, WeightedSum


class AsyncRubric(Rubric):
    """Async rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0, delay_ms: float = 0):
        super().__init__()
        self.score = score
        self.delay_ms = delay_ms
        self.call_count = 0

    async def forward(self, action: Any, observation: Any) -> float:
        """Async forward with optional delay."""
        self.call_count += 1
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
        return self.score


class TestAsyncSequential:
    """Test async Sequential container."""

    @pytest.mark.asyncio
    async def test_empty_sequential_async(self):
        """Empty sequential returns 1.0."""
        rubric = Sequential()
        result = await rubric("action", "obs")
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_single_async_rubric(self):
        """Single async rubric returns its score."""
        rubric = Sequential(AsyncRubric(0.8))
        result = await rubric("action", "obs")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_multiple_async_rubrics_all_pass(self):
        """Multiple async rubrics return last score."""
        rubric = Sequential(
            AsyncRubric(1.0),
            AsyncRubric(0.8),
            AsyncRubric(0.9),
        )
        result = await rubric("action", "obs")
        assert result == 0.9

    @pytest.mark.asyncio
    async def test_fail_fast_on_zero_async(self):
        """Stops immediately when an async rubric returns 0."""
        r1 = AsyncRubric(1.0)
        r2 = AsyncRubric(0.0)  # Fails
        r3 = AsyncRubric(1.0)

        rubric = Sequential(r1, r2, r3)
        result = await rubric("action", "obs")

        assert result == 0.0
        assert r1.call_count == 1
        assert r2.call_count == 1
        assert r3.call_count == 0  # Should not be called

    @pytest.mark.asyncio
    async def test_sequential_awaits_each_child(self):
        """Sequential awaits each child in order."""
        call_order = []

        class OrderedAsyncRubric(Rubric):
            def __init__(self, name: str, score: float):
                super().__init__()
                self.name = name
                self.score = score

            async def forward(self, action, observation):
                call_order.append(self.name)
                await asyncio.sleep(0.001)  # Small delay
                return self.score

        rubric = Sequential(
            OrderedAsyncRubric("first", 1.0),
            OrderedAsyncRubric("second", 0.8),
            OrderedAsyncRubric("third", 0.9),
        )
        result = await rubric("action", "obs")

        assert result == 0.9
        assert call_order == ["first", "second", "third"]


class TestAsyncGate:
    """Test async Gate container."""

    @pytest.mark.asyncio
    async def test_gate_passes_above_threshold_async(self):
        """Returns child score when above threshold."""
        rubric = Gate(AsyncRubric(0.8), threshold=0.5)
        result = await rubric("action", "obs")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_gate_fails_below_threshold_async(self):
        """Returns 0 when child score is below threshold."""
        rubric = Gate(AsyncRubric(0.4), threshold=0.5)
        result = await rubric("action", "obs")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_gate_passes_at_threshold_async(self):
        """Returns score when exactly at threshold."""
        rubric = Gate(AsyncRubric(0.5), threshold=0.5)
        result = await rubric("action", "obs")
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_gate_default_threshold_async(self):
        """Default threshold is 1.0."""
        # Passes only with perfect score
        rubric = Gate(AsyncRubric(1.0))
        assert await rubric("action", "obs") == 1.0

        rubric2 = Gate(AsyncRubric(0.99))
        assert await rubric2("action", "obs") == 0.0

    @pytest.mark.asyncio
    async def test_gate_awaits_child(self):
        """Gate awaits async child."""
        rubric = Gate(AsyncRubric(0.8, delay_ms=10), threshold=0.5)
        result = await rubric("action", "obs")
        assert result == 0.8


class TestAsyncWeightedSum:
    """Test async WeightedSum container."""

    @pytest.mark.asyncio
    async def test_single_rubric_weight_one_async(self):
        """Single async rubric with weight 1.0."""
        rubric = WeightedSum([AsyncRubric(0.8)], [1.0])
        result = await rubric("action", "obs")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_two_rubrics_equal_weights_async(self):
        """Two async rubrics with equal weights."""
        rubric = WeightedSum(
            [AsyncRubric(0.6), AsyncRubric(0.8)],
            [0.5, 0.5],
        )
        result = await rubric("action", "obs")
        assert result == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_weighted_combination_async(self):
        """Weighted combination with async rubrics."""
        rubric = WeightedSum(
            [AsyncRubric(1.0), AsyncRubric(0.0)],
            [0.7, 0.3],
        )
        result = await rubric("action", "obs")
        assert result == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_weighted_sum_parallel_execution(self):
        """WeightedSum can execute children in parallel."""
        # This test verifies that children are evaluated concurrently,
        # not sequentially
        rubric = WeightedSum(
            [
                AsyncRubric(1.0, delay_ms=50),
                AsyncRubric(0.8, delay_ms=50),
                AsyncRubric(0.6, delay_ms=50),
            ],
            [0.5, 0.3, 0.2],
        )

        import time

        start = time.time()
        result = await rubric("action", "obs")
        elapsed = time.time() - start

        # If sequential, would take ~150ms. Parallel should be ~50ms
        # Allow some overhead
        assert elapsed < 0.1  # 100ms max (parallel execution)
        assert result == pytest.approx(0.86)  # 1.0*0.5 + 0.8*0.3 + 0.6*0.2

    @pytest.mark.asyncio
    async def test_weighted_sum_awaits_all_children(self):
        """WeightedSum awaits all async children."""
        r1 = AsyncRubric(1.0, delay_ms=10)
        r2 = AsyncRubric(0.5, delay_ms=10)

        rubric = WeightedSum([r1, r2], [0.6, 0.4])
        result = await rubric("action", "obs")

        assert result == pytest.approx(0.8)  # 1.0*0.6 + 0.5*0.4
        assert r1.call_count == 1
        assert r2.call_count == 1


class TestAsyncContainerComposition:
    """Test composing async containers together."""

    @pytest.mark.asyncio
    async def test_sequential_of_async_gates(self):
        """Sequential of async Gate rubrics."""
        rubric = Sequential(
            Gate(AsyncRubric(1.0)),  # Must pass completely
            Gate(AsyncRubric(0.6), threshold=0.5),  # Must be >= 0.5
            AsyncRubric(0.9),  # Final score
        )
        result = await rubric("action", "obs")
        assert result == 0.9

    @pytest.mark.asyncio
    async def test_sequential_fails_early_async(self):
        """Sequential stops when async Gate fails."""
        r3 = AsyncRubric(0.9)

        rubric = Sequential(
            Gate(AsyncRubric(0.3), threshold=0.5),  # Fails
            r3,
        )
        result = await rubric("action", "obs")

        assert result == 0.0
        assert r3.call_count == 0

    @pytest.mark.asyncio
    async def test_weighted_sum_of_async_gates(self):
        """WeightedSum with async Gate rubrics."""
        rubric = WeightedSum(
            [
                Gate(AsyncRubric(0.8), threshold=0.5),  # Passes: 0.8
                Gate(AsyncRubric(0.3), threshold=0.5),  # Fails: 0.0
            ],
            [0.6, 0.4],
        )
        result = await rubric("action", "obs")
        # 0.8 * 0.6 + 0.0 * 0.4 = 0.48
        assert result == pytest.approx(0.48)

    @pytest.mark.asyncio
    async def test_nested_async_rubrics(self):
        """Can nest async rubrics deeply."""
        inner = Sequential(
            Gate(AsyncRubric(1.0, delay_ms=5), threshold=0.5),
            AsyncRubric(0.8),
        )

        outer = WeightedSum(
            [inner, AsyncRubric(0.6)],
            [0.7, 0.3],
        )

        result = await outer("action", "obs")
        # inner returns 0.8, second child returns 0.6
        # 0.8 * 0.7 + 0.6 * 0.3 = 0.56 + 0.18 = 0.74
        assert result == pytest.approx(0.74)

    @pytest.mark.asyncio
    async def test_complex_hierarchy_with_parallel_execution(self):
        """Complex hierarchy leverages parallel execution."""

        # Create a weighted sum of two sequential chains
        # Each sequential chain has delays, but the two chains
        # should execute in parallel
        chain1 = Sequential(
            AsyncRubric(1.0, delay_ms=20),
            AsyncRubric(0.8, delay_ms=20),
        )
        chain2 = Sequential(
            AsyncRubric(0.6, delay_ms=20),
            AsyncRubric(0.5, delay_ms=20),
        )

        rubric = WeightedSum([chain1, chain2], [0.5, 0.5])

        import time

        start = time.time()
        result = await rubric("action", "obs")
        elapsed = time.time() - start

        # Sequential execution would take ~80ms total
        # Parallel execution should be ~40ms (two 20ms chains in parallel)
        assert elapsed < 0.08  # Allow overhead
        assert result == pytest.approx(0.65)  # (0.8 + 0.5) / 2


class TestAsyncBackwardCompatibility:
    """Test backward compatibility with sync rubrics in containers."""

    @pytest.mark.asyncio
    async def test_sequential_with_sync_rubrics(self):
        """Sequential works with sync rubrics."""

        class SyncRubric(Rubric):
            def __init__(self, score: float):
                super().__init__()
                self.score = score

            def forward(self, action, observation):
                return self.score

        rubric = Sequential(
            SyncRubric(1.0),
            SyncRubric(0.8),
        )
        result = await rubric("action", "obs")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_sequential_with_failing_sync_rubric(self):
        """Sequential stopped by a sync child is still awaitable."""

        class SyncRubric(Rubric):
            def __init__(self, score: float):
                super().__init__()
                self.score = score

            def forward(self, action, observation):
                return self.score

        first_fails = Sequential(SyncRubric(0.0), AsyncRubric(0.8))
        assert await first_fails("action", "obs") == 0.0
        assert first_fails.last_score == 0.0

        later_fails = Sequential(SyncRubric(1.0), SyncRubric(0.0), AsyncRubric(0.8))
        assert await later_fails("action", "obs") == 0.0
        assert later_fails.last_score == 0.0

    @pytest.mark.asyncio
    async def test_weighted_sum_mixed_sync_async(self):
        """WeightedSum works with mixed sync/async rubrics."""

        class SyncRubric(Rubric):
            def __init__(self, score: float):
                super().__init__()
                self.score = score

            def forward(self, action, observation):
                return self.score

        rubric = WeightedSum(
            [SyncRubric(0.6), AsyncRubric(0.8)],
            [0.5, 0.5],
        )
        result = await rubric("action", "obs")
        assert result == pytest.approx(0.7)
