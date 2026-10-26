# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TrajectoryRubric and ExponentialDiscountingTrajectoryRubric."""

from dataclasses import dataclass
from typing import Any, List, Tuple

import pytest
from openenv.core.rubrics.trajectory import (
    ExponentialDiscountingTrajectoryRubric,
    TrajectoryRubric,
)


@dataclass
class MockObservation:
    """Mock observation for testing."""

    done: bool = False
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockAction:
    """Mock action for testing."""

    value: str = "move"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WinLossRubric(ExponentialDiscountingTrajectoryRubric):
    """Example rubric that scores 1.0 for win, 0.0 for loss, 0.5 for draw."""

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        outcome = getattr(final_obs, "metadata", {}).get("outcome")
        if outcome == "win":
            return 1.0
        elif outcome == "loss":
            return 0.0
        else:
            return 0.5


class EqualCreditRubric(TrajectoryRubric):
    """Rubric that gives equal credit to all steps."""

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        return final_obs.metadata.get("score", 0.0)

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)


class TestTrajectoryRubricBasics:
    """Test basic TrajectoryRubric functionality."""

    def test_abstract_methods_required(self):
        """Cannot instantiate TrajectoryRubric without implementing abstract methods."""
        with pytest.raises(TypeError):
            TrajectoryRubric()

    def test_returns_intermediate_until_done(self):
        """Returns intermediate_reward for non-terminal steps."""
        rubric = EqualCreditRubric(intermediate_reward=0.0)

        obs1 = MockObservation(done=False)
        result = rubric(MockAction(), obs1)

        assert result == 0.0
        assert len(rubric._trajectory) == 1

    def test_returns_score_when_done(self):
        """Returns trajectory score when done=True."""
        rubric = EqualCreditRubric(intermediate_reward=0.0)

        obs1 = MockObservation(done=False)
        obs2 = MockObservation(done=True, metadata={"score": 0.8})

        rubric(MockAction(), obs1)
        result = rubric(MockAction(), obs2)

        assert result == 0.8
        assert len(rubric._trajectory) == 2

    def test_custom_intermediate_reward(self):
        """Intermediate reward can be customized."""
        rubric = EqualCreditRubric(intermediate_reward=0.1)

        obs = MockObservation(done=False)
        result = rubric(MockAction(), obs)

        assert result == 0.1

    def test_reset_clears_trajectory(self):
        """reset() clears the accumulated trajectory."""
        rubric = EqualCreditRubric()

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        assert len(rubric._trajectory) == 2

        rubric.reset()
        assert len(rubric._trajectory) == 0

    def test_trajectory_property_returns_copy(self):
        """trajectory property returns a copy."""
        rubric = EqualCreditRubric()

        rubric(MockAction(), MockObservation(done=False))
        trajectory = rubric.trajectory

        # Modifying the copy should not affect internal state
        trajectory.clear()
        assert len(rubric._trajectory) == 1


class TestExponentialDiscounting:
    """Test ExponentialDiscountingTrajectoryRubric."""

    def test_gamma_validation(self):
        """Gamma must be in [0, 1]."""
        with pytest.raises(ValueError):
            WinLossRubric(gamma=-0.1)

        with pytest.raises(ValueError):
            WinLossRubric(gamma=1.5)

    def test_gamma_one_equal_credit(self):
        """With gamma=1.0, all steps get equal credit."""
        rubric = WinLossRubric(gamma=1.0)

        # Simulate 3-step episode with win
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert len(step_rewards) == 3
        assert step_rewards[0] == 1.0
        assert step_rewards[1] == 1.0
        assert step_rewards[2] == 1.0

    def test_gamma_zero_final_only(self):
        """With gamma=0.0, only final step gets reward."""
        rubric = WinLossRubric(gamma=0.0)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [0.0, 0.0, 1.0]

    def test_gamma_discounting_pattern(self):
        """With 0 < gamma < 1, later steps get higher reward."""
        rubric = WinLossRubric(gamma=0.5)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        # r_t = gamma^(T-1-t) * R_final, T=3, R_final=1.0
        # t=0: 0.5^2 = 0.25
        # t=1: 0.5^1 = 0.5
        # t=2: 0.5^0 = 1.0
        assert step_rewards[0] == pytest.approx(0.25)
        assert step_rewards[1] == pytest.approx(0.5)
        assert step_rewards[2] == pytest.approx(1.0)

    def test_gamma_099_standard_discounting(self):
        """With gamma=0.99, standard RL discounting pattern."""
        rubric = WinLossRubric(gamma=0.99)

        # 5-step episode with win
        for _ in range(4):
            rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        # Verify discounting order: later steps get more
        for i in range(len(step_rewards) - 1):
            assert step_rewards[i] < step_rewards[i + 1]

        # Final step gets full reward
        assert step_rewards[-1] == pytest.approx(1.0)

    def test_loss_outcome(self):
        """Loss returns 0.0 for all steps."""
        rubric = WinLossRubric(gamma=0.99)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "loss"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [0.0, 0.0]

    def test_draw_outcome(self):
        """Draw returns 0.5 for all steps (with discounting)."""
        rubric = WinLossRubric(gamma=1.0)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "draw"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [0.5, 0.5]

    def test_empty_trajectory(self):
        """compute_step_rewards() returns empty list for empty trajectory."""
        rubric = WinLossRubric(gamma=0.99)

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == []


class TestTrajectoryRubricStateSerialization:
    """Test state_dict serialization for trajectory rubrics."""

    def test_trajectory_rubric_state_dict(self):
        """TrajectoryRubric serializes intermediate_reward."""
        rubric = EqualCreditRubric(intermediate_reward=0.2)

        state = rubric.state_dict()

        assert state["intermediate_reward"] == 0.2

    def test_trajectory_rubric_load_state_dict(self):
        """TrajectoryRubric loads intermediate_reward from state."""
        rubric = EqualCreditRubric(intermediate_reward=0.0)

        rubric.load_state_dict({"intermediate_reward": 0.3})

        assert rubric.intermediate_reward == 0.3

    def test_exponential_discounting_state_dict(self):
        """ExponentialDiscountingTrajectoryRubric serializes gamma."""
        rubric = WinLossRubric(gamma=0.95, intermediate_reward=0.1)

        state = rubric.state_dict()

        assert state["gamma"] == 0.95
        assert state["intermediate_reward"] == 0.1

    def test_exponential_discounting_load_state_dict(self):
        """ExponentialDiscountingTrajectoryRubric loads gamma from state."""
        rubric = WinLossRubric(gamma=0.99)

        rubric.load_state_dict({"gamma": 0.9, "intermediate_reward": 0.2})

        assert rubric.gamma == 0.9
        assert rubric.intermediate_reward == 0.2


class TestTrajectoryRubricHooks:
    """Test that hooks work with trajectory rubrics."""

    def test_hooks_called_each_step(self):
        """Forward hooks are called on each step."""
        rubric = EqualCreditRubric()
        hook_calls = []

        def hook(r, action, obs, result):
            hook_calls.append(result)

        rubric.register_forward_hook(hook)

        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"score": 0.7}))

        assert len(hook_calls) == 2
        assert hook_calls[0] == 0.0  # intermediate
        assert hook_calls[1] == 0.7  # final


class TestTrajectoryRubricEdgeCases:
    """Test edge cases."""

    def test_single_step_episode(self):
        """Single-step episode (immediately done)."""
        rubric = WinLossRubric(gamma=0.99)

        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert step_rewards == [1.0]

    def test_very_long_episode(self):
        """Long episode (100 steps)."""
        rubric = WinLossRubric(gamma=0.99)

        for _ in range(99):
            rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))

        step_rewards = rubric.compute_step_rewards()

        assert len(step_rewards) == 100
        # First step should have gamma^99 reward
        assert step_rewards[0] == pytest.approx(0.99**99)
        # Last step should have full reward
        assert step_rewards[-1] == 1.0

    def test_observation_without_done_attribute(self):
        """Handles observations without done attribute (defaults to False)."""
        rubric = EqualCreditRubric()

        class NoDoneObs:
            pass

        obs = NoDoneObs()
        result = rubric(MockAction(), obs)

        # Should treat as not done
        assert result == 0.0
        assert len(rubric._trajectory) == 1

    def test_multiple_episodes_with_reset(self):
        """Multiple episodes with reset between them."""
        rubric = WinLossRubric(gamma=1.0)

        # Episode 1: win
        rubric(MockAction(), MockObservation(done=False))
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "win"}))
        ep1_rewards = rubric.compute_step_rewards()

        rubric.reset()

        # Episode 2: loss
        rubric(MockAction(), MockObservation(done=True, metadata={"outcome": "loss"}))
        ep2_rewards = rubric.compute_step_rewards()

        assert ep1_rewards == [1.0, 1.0]
        assert ep2_rewards == [0.0]
