# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the base Rubric class."""

from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric


class SimpleRubric(Rubric):
    """Concrete rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score

    def forward(self, action: Any, observation: Any) -> float:
        return self.score


class CompositeRubric(Rubric):
    """Rubric with child rubrics."""

    def __init__(self):
        super().__init__()
        self.child1 = SimpleRubric(0.5)
        self.child2 = SimpleRubric(0.7)

    def forward(self, action: Any, observation: Any) -> float:
        return (self.child1(action, observation) + self.child2(action, observation)) / 2


class TestRubricBasics:
    """Test basic Rubric functionality."""

    def test_forward_is_abstract(self):
        """Cannot instantiate Rubric directly."""
        with pytest.raises(TypeError):
            Rubric()

    def test_simple_rubric_call(self):
        """Calling a rubric invokes forward()."""
        rubric = SimpleRubric(0.8)
        result = rubric("action", "observation")
        assert result == 0.8

    def test_last_score_tracked(self):
        """last_score is updated after each call."""
        rubric = SimpleRubric(0.6)
        assert rubric.last_score is None

        rubric("action", "observation")
        assert rubric.last_score == 0.6


class TestChildRegistration:
    """Test auto-registration of child rubrics."""

    def test_children_registered(self):
        """Child rubrics are registered when assigned as attributes."""
        rubric = CompositeRubric()

        children = list(rubric.children())
        assert len(children) == 2
        assert rubric.child1 in children
        assert rubric.child2 in children

    def test_named_children(self):
        """named_children returns name-rubric pairs."""
        rubric = CompositeRubric()

        named = dict(rubric.named_children())
        assert "child1" in named
        assert "child2" in named
        assert named["child1"].score == 0.5
        assert named["child2"].score == 0.7

    def test_rubrics_recursive(self):
        """rubrics() returns all descendants."""

        class NestedRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.inner = CompositeRubric()

            def forward(self, action, observation):
                return self.inner(action, observation)

        rubric = NestedRubric()

        all_rubrics = list(rubric.rubrics())
        # inner, inner.child1, inner.child2
        assert len(all_rubrics) == 3

    def test_named_rubrics_paths(self):
        """named_rubrics() returns dot-separated paths."""

        class NestedRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.inner = CompositeRubric()

            def forward(self, action, observation):
                return self.inner(action, observation)

        rubric = NestedRubric()

        paths = dict(rubric.named_rubrics())
        assert "inner" in paths
        assert "inner.child1" in paths
        assert "inner.child2" in paths

    def test_get_rubric_by_path(self):
        """get_rubric() navigates dot-separated paths."""

        class NestedRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.inner = CompositeRubric()

            def forward(self, action, observation):
                return self.inner(action, observation)

        rubric = NestedRubric()

        assert rubric.get_rubric("inner") is rubric.inner
        assert rubric.get_rubric("inner.child1") is rubric.inner.child1

    def test_get_rubric_invalid_path(self):
        """get_rubric() raises KeyError for invalid paths."""
        rubric = CompositeRubric()

        with pytest.raises(KeyError):
            rubric.get_rubric("nonexistent")


class TestHooks:
    """Test forward hook functionality."""

    def test_forward_hook_called(self):
        """Forward hooks are called after forward()."""
        rubric = SimpleRubric(0.9)
        hook_calls = []

        def hook(r, action, obs, result):
            hook_calls.append((action, obs, result))

        rubric.register_forward_hook(hook)
        rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs", 0.9)

    def test_forward_pre_hook_called(self):
        """Pre-forward hooks are called before forward()."""
        rubric = SimpleRubric(0.9)
        hook_calls = []

        def pre_hook(r, action, obs):
            hook_calls.append((action, obs))

        rubric.register_forward_pre_hook(pre_hook)
        rubric("my_action", "my_obs")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("my_action", "my_obs")

    def test_multiple_hooks(self):
        """Multiple hooks can be registered."""
        rubric = SimpleRubric(0.5)
        results = []

        rubric.register_forward_hook(lambda r, a, o, res: results.append(1))
        rubric.register_forward_hook(lambda r, a, o, res: results.append(2))

        rubric("action", "obs")

        assert results == [1, 2]


class TestReset:
    """Test reset functionality."""

    def test_default_reset_is_noop(self):
        """Default reset() does nothing (for stateless rubrics)."""
        rubric = SimpleRubric(0.5)
        rubric.reset()  # Should not raise


class TestStateDictSerialization:
    """Test state_dict serialization."""

    def test_default_state_dict_empty(self):
        """Default state_dict returns empty dict."""
        rubric = SimpleRubric(0.5)
        assert rubric.state_dict() == {}

    def test_load_state_dict_accepts_empty(self):
        """load_state_dict accepts empty dict."""
        rubric = SimpleRubric(0.5)
        rubric.load_state_dict({})  # Should not raise
