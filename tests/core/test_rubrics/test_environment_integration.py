# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rubric integration with Environment base class."""

from typing import Any, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Rubric, TrajectoryRubric


# Test fixtures - using Pydantic models (not dataclasses)


class MockAction(Action):
    """Simple action for testing."""

    content: str = "test"


class MockObservation(Observation):
    """Simple observation for testing."""

    content: str = ""


class MockState(State):
    """Simple state for testing."""

    pass


class FixedRubric(Rubric):
    """Rubric that returns a fixed score."""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score

    def forward(self, action: Any, observation: Any) -> float:
        return self.score


class CountingRubric(Rubric):
    """Rubric that counts calls and returns action-dependent score."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, action: Any, observation: Any) -> float:
        self.call_count += 1
        # Return score based on action content
        if hasattr(action, "content"):
            if action.content == "good":
                return 1.0
            elif action.content == "bad":
                return 0.0
        return 0.5


class MockTrajectoryRubric(TrajectoryRubric):
    """Trajectory rubric for testing reset behavior."""

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        return 1.0 if trajectory else 0.0

    def compute_step_rewards(self) -> List[float]:
        return [1.0] * len(self._trajectory)


class SimpleEnvironment(Environment[MockAction, MockObservation, MockState]):
    """Minimal environment implementation for testing."""

    def __init__(self, rubric: Optional[Rubric] = None):
        super().__init__(rubric=rubric)
        self._state = MockState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MockObservation:
        self._reset_rubric()  # Reset rubric state
        self._state = MockState(episode_id=episode_id)
        return MockObservation(content="initial")

    def step(
        self,
        action: MockAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MockObservation:
        obs = MockObservation(content=f"response to {action.content}")
        obs.reward = self._apply_rubric(action, obs)
        return obs

    @property
    def state(self) -> MockState:
        return self._state


class TestEnvironmentRubricIntegration:
    """Test rubric integration with Environment base class."""

    def test_environment_without_rubric(self):
        """Environment works without a rubric."""
        env = SimpleEnvironment()
        assert env.rubric is None

        obs = env.reset()
        assert obs.content == "initial"

        obs = env.step(MockAction(content="test"))
        assert obs.reward == 0.0  # Default when no rubric

    def test_environment_with_rubric(self):
        """Environment uses rubric for reward computation."""
        rubric = FixedRubric(0.75)
        env = SimpleEnvironment(rubric=rubric)

        assert env.rubric is rubric

        env.reset()
        obs = env.step(MockAction(content="test"))

        assert obs.reward == 0.75

    def test_rubric_called_each_step(self):
        """Rubric is called on each step."""
        rubric = CountingRubric()
        env = SimpleEnvironment(rubric=rubric)

        env.reset()
        assert rubric.call_count == 0

        env.step(MockAction(content="a"))
        assert rubric.call_count == 1

        env.step(MockAction(content="b"))
        assert rubric.call_count == 2

    def test_rubric_receives_action_and_observation(self):
        """Rubric receives both action and observation."""
        rubric = CountingRubric()
        env = SimpleEnvironment(rubric=rubric)

        env.reset()

        obs = env.step(MockAction(content="good"))
        assert obs.reward == 1.0

        obs = env.step(MockAction(content="bad"))
        assert obs.reward == 0.0

    def test_rubric_reset_on_env_reset(self):
        """Rubric state is reset when environment resets."""
        rubric = MockTrajectoryRubric()
        env = SimpleEnvironment(rubric=rubric)

        env.reset()
        env.step(MockAction(content="a"))
        env.step(MockAction(content="b"))

        assert len(rubric._trajectory) == 2

        env.reset()
        assert len(rubric._trajectory) == 0  # Reset clears trajectory

    def test_rubric_introspection(self):
        """Can introspect rubric from environment."""

        class CompositeRubric(Rubric):
            def __init__(self):
                super().__init__()
                self.child1 = FixedRubric(0.5)
                self.child2 = FixedRubric(0.8)

            def forward(self, action, obs):
                return (self.child1(action, obs) + self.child2(action, obs)) / 2

        rubric = CompositeRubric()
        env = SimpleEnvironment(rubric=rubric)

        env.reset()
        env.step(MockAction(content="test"))

        # Can introspect child scores
        assert env.rubric is not None
        named = dict(env.rubric.named_children())
        assert "child1" in named
        assert "child2" in named
        assert named["child1"].last_score == 0.5
        assert named["child2"].last_score == 0.8

    def test_apply_rubric_without_rubric(self):
        """_apply_rubric returns 0.0 when no rubric is set."""
        env = SimpleEnvironment()
        action = MockAction(content="test")
        obs = MockObservation(content="result")

        result = env._apply_rubric(action, obs)
        assert result == 0.0

    def test_reset_rubric_without_rubric(self):
        """_reset_rubric is safe when no rubric is set."""
        env = SimpleEnvironment()
        env._reset_rubric()  # Should not raise


class TestEnvironmentRubricLifecycle:
    """Test rubric lifecycle with multiple episodes."""

    def test_multiple_episodes(self):
        """Rubric handles multiple episodes correctly."""
        rubric = MockTrajectoryRubric()
        env = SimpleEnvironment(rubric=rubric)

        # Episode 1
        env.reset()
        env.step(MockAction(content="a"))
        env.step(MockAction(content="b"))
        ep1_len = len(rubric._trajectory)

        # Episode 2
        env.reset()
        env.step(MockAction(content="c"))
        ep2_len = len(rubric._trajectory)

        assert ep1_len == 2
        assert ep2_len == 1  # Reset cleared previous episode

    def test_rubric_hooks_work(self):
        """Rubric hooks work through environment."""
        rubric = FixedRubric(0.9)
        env = SimpleEnvironment(rubric=rubric)

        hook_calls = []

        def hook(r, action, obs, result):
            hook_calls.append(result)

        rubric.register_forward_hook(hook)

        env.reset()
        env.step(MockAction(content="a"))
        env.step(MockAction(content="b"))

        assert len(hook_calls) == 2
        assert hook_calls == [0.9, 0.9]
