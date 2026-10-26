# SPDX-License-Identifier: BSD-3-Clause

"""Tests for container rubrics: Sequential, Gate, WeightedSum, RubricList, RubricDict."""

from typing import Any

import pytest
from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import (
    Gate,
    RubricDict,
    RubricList,
    Sequential,
    WeightedSum,
)


class FixedRubric(Rubric):
    """Concrete rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score

    def forward(self, action: Any, observation: Any) -> float:
        return self.score


class CountingRubric(Rubric):
    """Rubric that counts how many times it's called."""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score
        self.call_count = 0

    def forward(self, action: Any, observation: Any) -> float:
        self.call_count += 1
        return self.score


class TestSequential:
    """Test Sequential container."""

    def test_empty_sequential(self):
        """Empty sequential returns 1.0."""
        rubric = Sequential()
        result = rubric("action", "obs")
        assert result == 1.0

    def test_single_rubric(self):
        """Single rubric returns its score."""
        rubric = Sequential(FixedRubric(0.8))
        result = rubric("action", "obs")
        assert result == 0.8

    def test_multiple_rubrics_all_pass(self):
        """Multiple passing rubrics return last score."""
        rubric = Sequential(
            FixedRubric(1.0),
            FixedRubric(0.8),
            FixedRubric(0.9),
        )
        result = rubric("action", "obs")
        assert result == 0.9

    def test_fail_fast_on_zero(self):
        """Stops immediately when a rubric returns 0."""
        r1 = CountingRubric(1.0)
        r2 = CountingRubric(0.0)  # Fails
        r3 = CountingRubric(1.0)

        rubric = Sequential(r1, r2, r3)
        result = rubric("action", "obs")

        assert result == 0.0
        assert r1.call_count == 1
        assert r2.call_count == 1
        assert r3.call_count == 0  # Should not be called

    def test_children_registered(self):
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = Sequential(r1, r2)

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_len_and_getitem(self):
        """__len__ and __getitem__ work correctly."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = Sequential(r1, r2)

        assert len(rubric) == 2
        assert rubric[0] is r1
        assert rubric[1] is r2


class TestGate:
    """Test Gate container."""

    def test_gate_passes_above_threshold(self):
        """Returns child score when above threshold."""
        rubric = Gate(FixedRubric(0.8), threshold=0.5)
        result = rubric("action", "obs")
        assert result == 0.8

    def test_gate_fails_below_threshold(self):
        """Returns 0 when child score is below threshold."""
        rubric = Gate(FixedRubric(0.4), threshold=0.5)
        result = rubric("action", "obs")
        assert result == 0.0

    def test_gate_passes_at_threshold(self):
        """Returns score when exactly at threshold."""
        rubric = Gate(FixedRubric(0.5), threshold=0.5)
        result = rubric("action", "obs")
        assert result == 0.5

    def test_gate_default_threshold(self):
        """Default threshold is 1.0."""
        # Passes only with perfect score
        rubric = Gate(FixedRubric(1.0))
        assert rubric("action", "obs") == 1.0

        rubric2 = Gate(FixedRubric(0.99))
        assert rubric2("action", "obs") == 0.0

    def test_gate_child_registered(self):
        """Child rubric is auto-registered."""
        child = FixedRubric(0.5)
        rubric = Gate(child, threshold=0.3)

        children = list(rubric.children())
        assert len(children) == 1
        assert child in children


class TestWeightedSum:
    """Test WeightedSum container."""

    def test_single_rubric_weight_one(self):
        """Single rubric with weight 1.0."""
        rubric = WeightedSum([FixedRubric(0.8)], [1.0])
        result = rubric("action", "obs")
        assert result == 0.8

    def test_two_rubrics_equal_weights(self):
        """Two rubrics with equal weights."""
        rubric = WeightedSum(
            [FixedRubric(0.6), FixedRubric(0.8)],
            [0.5, 0.5],
        )
        result = rubric("action", "obs")
        assert result == pytest.approx(0.7)

    def test_weighted_combination(self):
        """Weighted combination with different weights."""
        rubric = WeightedSum(
            [FixedRubric(1.0), FixedRubric(0.0)],
            [0.7, 0.3],
        )
        result = rubric("action", "obs")
        assert result == pytest.approx(0.7)

    def test_weights_must_sum_to_one(self):
        """Raises error if weights don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            WeightedSum([FixedRubric(0.5), FixedRubric(0.5)], [0.5, 0.3])

    def test_lengths_must_match(self):
        """Raises error if lengths don't match."""
        with pytest.raises(ValueError, match="must match"):
            WeightedSum([FixedRubric(0.5), FixedRubric(0.5)], [1.0])

    def test_children_registered(self):
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = WeightedSum([r1, r2], [0.5, 0.5])

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_weights_property(self):
        """weights property returns copy of weights."""
        rubric = WeightedSum([FixedRubric(0.5)], [1.0])

        weights = rubric.weights
        assert weights == [1.0]

        # Modifying copy shouldn't affect internal state
        weights.append(0.5)
        assert rubric.weights == [1.0]


class TestRubricList:
    """Test RubricList container."""

    def test_empty_list(self):
        """Empty list has length 0."""
        rubric = RubricList()
        assert len(rubric) == 0

    def test_init_with_rubrics(self):
        """Initialize with list of rubrics."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricList([r1, r2])

        assert len(rubric) == 2
        assert rubric[0] is r1
        assert rubric[1] is r2

    def test_append(self):
        """Append adds rubric to list."""
        rubric = RubricList()
        r1 = FixedRubric(0.5)

        rubric.append(r1)

        assert len(rubric) == 1
        assert rubric[0] is r1

    def test_extend(self):
        """Extend adds multiple rubrics."""
        rubric = RubricList()
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric.extend([r1, r2])

        assert len(rubric) == 2

    def test_iteration(self):
        """Can iterate over rubrics."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricList([r1, r2])

        items = list(rubric)
        assert items == [r1, r2]

    def test_children_registered(self):
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricList([r1, r2])

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_forward_not_implemented(self):
        """forward() raises NotImplementedError."""
        rubric = RubricList([FixedRubric(0.5)])

        with pytest.raises(NotImplementedError):
            rubric("action", "obs")


class TestRubricDict:
    """Test RubricDict container."""

    def test_empty_dict(self):
        """Empty dict has length 0."""
        rubric = RubricDict()
        assert len(rubric) == 0

    def test_init_with_dict(self):
        """Initialize with dictionary of rubrics."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricDict({"game1": r1, "game2": r2})

        assert len(rubric) == 2
        assert rubric["game1"] is r1
        assert rubric["game2"] is r2

    def test_setitem_and_getitem(self):
        """__setitem__ and __getitem__ work."""
        rubric = RubricDict()
        r1 = FixedRubric(0.5)

        rubric["game1"] = r1

        assert rubric["game1"] is r1

    def test_contains(self):
        """__contains__ works."""
        rubric = RubricDict({"game1": FixedRubric(0.5)})

        assert "game1" in rubric
        assert "game2" not in rubric

    def test_keys_values_items(self):
        """keys(), values(), items() work."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricDict({"game1": r1, "game2": r2})

        assert set(rubric.keys()) == {"game1", "game2"}
        assert set(rubric.values()) == {r1, r2}
        assert set(rubric.items()) == {("game1", r1), ("game2", r2)}

    def test_iteration(self):
        """Can iterate over keys."""
        rubric = RubricDict({"game1": FixedRubric(0.5), "game2": FixedRubric(0.7)})

        keys = list(rubric)
        assert set(keys) == {"game1", "game2"}

    def test_update(self):
        """update() adds rubrics from dict."""
        rubric = RubricDict({"game1": FixedRubric(0.5)})
        rubric.update({"game2": FixedRubric(0.7)})

        assert len(rubric) == 2
        assert "game2" in rubric

    def test_children_registered(self):
        """Child rubrics are auto-registered."""
        r1 = FixedRubric(0.5)
        r2 = FixedRubric(0.7)

        rubric = RubricDict({"game1": r1, "game2": r2})

        children = list(rubric.children())
        assert len(children) == 2
        assert r1 in children
        assert r2 in children

    def test_forward_not_implemented(self):
        """forward() raises NotImplementedError."""
        rubric = RubricDict({"game1": FixedRubric(0.5)})

        with pytest.raises(NotImplementedError):
            rubric("action", "obs")


class TestContainerComposition:
    """Test composing containers together."""

    def test_sequential_of_gates(self):
        """Sequential of Gate rubrics for hierarchical gating."""
        rubric = Sequential(
            Gate(FixedRubric(1.0)),  # Must pass completely
            Gate(FixedRubric(0.6), threshold=0.5),  # Must be >= 0.5
            FixedRubric(0.9),  # Final score
        )
        result = rubric("action", "obs")
        assert result == 0.9

    def test_sequential_fails_early(self):
        """Sequential stops when Gate fails."""
        r3 = CountingRubric(0.9)

        rubric = Sequential(
            Gate(FixedRubric(0.3), threshold=0.5),  # Fails
            r3,
        )
        result = rubric("action", "obs")

        assert result == 0.0
        assert r3.call_count == 0

    def test_weighted_sum_of_gates(self):
        """WeightedSum with Gate rubrics."""
        rubric = WeightedSum(
            [
                Gate(FixedRubric(0.8), threshold=0.5),  # Passes: 0.8
                Gate(FixedRubric(0.3), threshold=0.5),  # Fails: 0.0
            ],
            [0.6, 0.4],
        )
        result = rubric("action", "obs")
        # 0.8 * 0.6 + 0.0 * 0.4 = 0.48
        assert result == pytest.approx(0.48)

    def test_nested_named_rubrics(self):
        """Can traverse nested rubrics with named_rubrics()."""
        inner = Sequential(
            Gate(FixedRubric(1.0), threshold=0.5),
            FixedRubric(0.8),
        )
        outer = RubricDict({"task": inner})

        paths = dict(outer.named_rubrics())

        # Should have paths for all nested rubrics
        assert "task" in paths
        assert "task.rubric_0" in paths  # Gate
        assert "task.rubric_1" in paths  # FixedRubric
        # Gate's child
        assert "task.rubric_0.rubric" in paths
