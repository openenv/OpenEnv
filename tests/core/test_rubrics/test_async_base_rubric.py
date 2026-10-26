# SPDX-License-Identifier: BSD-3-Clause

"""Tests for async Rubric functionality.

This test file verifies that the Rubric base class supports async operations:
- async forward() method
- async __call__() with hooks
- Async hook execution
- Integration with async environments
"""

import asyncio
from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import RubricDict


class AsyncRubric(Rubric):
    """Concrete async rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score

    async def forward(self, action: Any, observation: Any) -> float:
        """Async forward implementation."""
        # Simulate async work (e.g., API call, DB query)
        return self.score


class AsyncCompositeRubric(Rubric):
    """Rubric with async child rubrics."""

    def __init__(self):
        super().__init__()
        self.child1 = AsyncRubric(0.5)
        self.child2 = AsyncRubric(0.7)

    async def forward(self, action: Any, observation: Any) -> float:
        """Async forward that awaits children."""
        score1 = await self.child1(action, observation)
        score2 = await self.child2(action, observation)
        return (score1 + score2) / 2


class TestAsyncRubricBasics:
    """Test basic async Rubric functionality."""

    @pytest.mark.asyncio
    async def test_async_forward_is_awaitable(self):
        """Async forward() can be awaited."""
        rubric = AsyncRubric(0.8)
        result = await rubric.forward("action", "observation")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_async_call_invokes_forward(self):
        """Calling an async rubric invokes async forward()."""
        rubric = AsyncRubric(0.8)
        result = await rubric("action", "observation")
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_last_score_tracked_async(self):
        """last_score is updated after async call."""
        rubric = AsyncRubric(0.6)
        assert rubric.last_score is None

        await rubric("action", "observation")
        assert rubric.last_score == 0.6

    @pytest.mark.asyncio
    async def test_async_composite_rubric(self):
        """Composite rubric with async children works."""
        rubric = AsyncCompositeRubric()
        result = await rubric("action", "observation")
        assert result == pytest.approx(0.6)  # (0.5 + 0.7) / 2


class TestAsyncRubricHooks:
    """Test async hook functionality."""

    @pytest.mark.asyncio
    async def test_forward_hook_called_async(self):
        """Forward hooks are called after async forward()."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        def hook(r, action, obs, result):
            hook_calls.append((action, obs, result))

        rubric.register_forward_hook(hook)
        await rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs", 0.9)

    @pytest.mark.asyncio
    async def test_forward_pre_hook_called_async(self):
        """Pre-forward hooks are called before async forward()."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        def pre_hook(r, action, obs):
            hook_calls.append((action, obs))

        rubric.register_forward_pre_hook(pre_hook)
        await rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs")

    @pytest.mark.asyncio
    async def test_multiple_hooks_async(self):
        """Multiple hooks work with async rubrics."""
        rubric = AsyncRubric(0.5)
        results = []

        rubric.register_forward_hook(lambda r, a, o, res: results.append(1))
        rubric.register_forward_hook(lambda r, a, o, res: results.append(2))

        await rubric("action", "obs")

        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_async_hooks(self):
        """Async hooks are supported."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        async def async_hook(r, action, obs, result):
            # Simulate async work in hook (e.g., logging to API)
            hook_calls.append(result)

        rubric.register_forward_hook(async_hook)
        await rubric("action", "obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == 0.9

    @pytest.mark.asyncio
    async def test_async_pre_hooks(self):
        """Async pre-hooks are supported."""
        rubric = AsyncRubric(0.9)
        hook_calls = []

        async def async_pre_hook(r, action, obs):
            # Simulate async pre-processing
            hook_calls.append((action, obs))

        rubric.register_forward_pre_hook(async_pre_hook)
        await rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs")


class TestAsyncChildTraversal:
    """Test async rubric child traversal works correctly."""

    @pytest.mark.asyncio
    async def test_children_still_iterable(self):
        """children() works the same for async rubrics."""
        rubric = AsyncCompositeRubric()

        children = list(rubric.children())
        assert len(children) == 2
        assert rubric.child1 in children
        assert rubric.child2 in children

    @pytest.mark.asyncio
    async def test_named_rubrics_async(self):
        """named_rubrics() works with async rubrics."""

        class NestedAsyncRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.inner = AsyncCompositeRubric()

            async def forward(self, action, observation):
                return await self.inner(action, observation)

        rubric = NestedAsyncRubric()

        paths = dict(rubric.named_rubrics())
        assert "inner" in paths
        assert "inner.child1" in paths
        assert "inner.child2" in paths

    @pytest.mark.asyncio
    async def test_get_rubric_by_path_async(self):
        """get_rubric() works with async rubrics."""

        class NestedAsyncRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.inner = AsyncCompositeRubric()

            async def forward(self, action, observation):
                return await self.inner(action, observation)

        rubric = NestedAsyncRubric()

        assert rubric.get_rubric("inner") is rubric.inner
        assert rubric.get_rubric("inner.child1") is rubric.inner.child1


class TestBackwardCompatibility:
    """Test that sync rubrics still work (backward compatibility)."""

    @pytest.mark.asyncio
    async def test_sync_rubric_still_works_sync(self):
        """Synchronous rubrics can still be called synchronously."""

        class SyncRubric(Rubric):
            def forward(self, action: Any, observation: Any) -> float:
                return 0.5

        rubric = SyncRubric()
        # Should work synchronously
        result = rubric("action", "obs")
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_sync_and_async_rubrics_mixed(self):
        """Mixing sync and async rubrics in a composite."""

        class SyncRubric(Rubric):
            def forward(self, action: Any, observation: Any) -> float:
                return 0.3

        class MixedComposite(Rubric):
            def __init__(self):
                super().__init__()
                self.sync_child = SyncRubric()
                self.async_child = AsyncRubric(0.7)

            async def forward(self, action, observation):
                # Can call sync child directly
                sync_score = self.sync_child(action, observation)
                # Must await async child
                async_score = await self.async_child(action, observation)
                return (sync_score + async_score) / 2

        rubric = MixedComposite()
        result = await rubric("action", "obs")
        assert result == pytest.approx(0.5)  # (0.3 + 0.7) / 2


class FixedSyncRubric(Rubric):
    """Concrete sync rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score

    def forward(self, action: Any, observation: Any) -> float:
        return self.score


class DispatchRubric(Rubric):
    """Sync forward() that hands off to an async child, as in the RubricDict
    dispatch pattern of the rubrics tutorial."""

    def __init__(self):
        super().__init__()
        self.games = RubricDict({"pong": AsyncRubric(0.9)})

    def forward(self, action: Any, observation: Any):
        return self.games["pong"](action, observation)


class TestSyncForwardReturningCoroutine:
    """A sync forward() may return an async child's coroutine; the parent must
    record the awaited score, not the coroutine."""

    @pytest.mark.asyncio
    async def test_last_score_is_awaited_score(self):
        rubric = DispatchRubric()

        assert await rubric("action", "obs") == 0.9
        assert rubric.last_score == 0.9

    @pytest.mark.asyncio
    async def test_hooks_see_awaited_score(self):
        rubric = DispatchRubric()
        log = []

        async def async_post_hook(r, action, obs, result):
            log.append(("async_post", result))

        rubric.register_forward_pre_hook(lambda r, a, o: log.append(("pre", None)))
        rubric.register_forward_hook(lambda r, a, o, res: log.append(("post", res)))
        rubric.register_forward_hook(async_post_hook)

        await rubric("action", "obs")

        assert log == [("pre", None), ("post", 0.9), ("async_post", 0.9)]

    def test_awaitable_from_sync_context(self):
        rubric = DispatchRubric()

        assert asyncio.run(rubric("action", "obs")) == 0.9
        assert rubric.last_score == 0.9

    def test_sync_child_still_returns_float(self):
        class SyncDispatch(Rubric):
            def __init__(self):
                super().__init__()
                self.games = RubricDict({"pong": FixedSyncRubric(0.4)})

            def forward(self, action, obs):
                return self.games["pong"](action, obs)

        rubric = SyncDispatch()
        assert rubric("action", "obs") == 0.4
        assert rubric.last_score == 0.4
