# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Reasoning Gym Environment."""

import pytest

pytest.importorskip("reasoning_gym", reason="reasoning_gym is not installed")

from reasoning_gym_env.models import ReasoningGymAction, ReasoningGymObservation
from reasoning_gym_env.server.reasoning_gym_environment import ReasoningGymEnvironment


class TestReasoningGymEnvironment:
    """Tests for the ReasoningGymEnvironment class."""

    def test_reset_with_simple_dataset(self):
        """Test reset with a simple dataset configuration."""
        env = ReasoningGymEnvironment()
        obs = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        assert isinstance(obs, ReasoningGymObservation)
        assert obs.question is not None
        assert isinstance(obs.question, str)
        assert obs.score is None
        assert obs.correct_answer is None
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_with_dataset_config(self):
        """Test reset with dataset config parameters."""
        env = ReasoningGymEnvironment()
        obs = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 3, "max_animals": 5},
            seed=42,
            size=10,
        )

        assert isinstance(obs, ReasoningGymObservation)
        assert obs.question is not None

    def test_reset_with_composite_dataset(self):
        """Test reset with a composite dataset."""
        env = ReasoningGymEnvironment()
        obs = env.reset(
            dataset_name="composite",
            dataset_specs=[
                {"name": "leg_counting", "weight": 1, "config": {}},
                {"name": "word_sorting", "weight": 1, "config": {}},
            ],
            seed=42,
            size=10,
        )

        assert isinstance(obs, ReasoningGymObservation)
        assert obs.question is not None

    def test_reset_reuses_dataset(self):
        """Test that reset without parameters reuses existing dataset."""
        env = ReasoningGymEnvironment()

        # Create initial dataset
        obs1 = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )
        question1 = obs1.question

        # Reset without params should get next question from same dataset
        obs2 = env.reset()
        question2 = obs2.question

        assert question1 != question2
        assert obs2.question is not None

    def test_reset_without_params_creates_default_dataset(self):
        """Test that reset without parameters creates default dataset."""
        env = ReasoningGymEnvironment()

        obs = env.reset()

        assert isinstance(obs, ReasoningGymObservation)
        assert obs.question is not None
        assert obs.done is False
        assert env._dataset_name == "leg_counting"
        assert env._dataset_seed == 42
        assert env._dataset_size == 1000

    def test_reset_default_can_be_overridden(self):
        """Test that default dataset can be overridden."""
        env = ReasoningGymEnvironment()

        # Create default dataset
        env.reset()

        # Override with explicit params
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 10, "max_animals": 20},
            seed=99,
            size=5,
        )

        assert env._dataset_seed == 99
        assert env._dataset_size == 5

    def test_reset_missing_seed_raises_error(self):
        """Test that reset without seed raises ValueError."""
        env = ReasoningGymEnvironment()

        with pytest.raises(ValueError, match="seed and size must be provided"):
            env.reset(dataset_name="leg_counting", size=10)

    def test_reset_missing_size_raises_error(self):
        """Test that reset without size raises ValueError."""
        env = ReasoningGymEnvironment()

        with pytest.raises(ValueError, match="seed and size must be provided"):
            env.reset(dataset_name="leg_counting", seed=42)

    def test_reset_composite_missing_specs_raises_error(self):
        """Test that composite dataset without specs raises ValueError."""
        env = ReasoningGymEnvironment()

        with pytest.raises(ValueError, match="dataset_specs must be provided"):
            env.reset(dataset_name="composite", seed=42, size=10)

    def test_reset_composite_empty_specs_raises_error(self):
        """Test that composite dataset with empty specs raises ValueError."""
        env = ReasoningGymEnvironment()

        with pytest.raises(ValueError, match="dataset_specs cannot be empty"):
            env.reset(dataset_name="composite", dataset_specs=[], seed=42, size=10)

    def test_step_scores_answer(self):
        """Test step with an answer and check scoring."""
        env = ReasoningGymEnvironment()
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        obs = env.step(ReasoningGymAction(answer="4"))

        assert isinstance(obs, ReasoningGymObservation)
        assert obs.question is None
        assert obs.score is not None
        assert isinstance(obs.score, (int, float))
        assert 0.0 <= obs.score <= 1.0
        assert obs.correct_answer is not None
        assert obs.done is True
        assert obs.reward == obs.score

    def test_step_increments_state(self):
        """Test that step increments step count."""
        env = ReasoningGymEnvironment()
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        assert env.state.step_count == 0

        env.step(ReasoningGymAction(answer="test"))

        assert env.state.step_count == 1

    def test_step_without_current_entry(self):
        """Test step when no current entry is set."""
        env = ReasoningGymEnvironment()
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        # Manually clear current entry
        env._current_entry = None

        obs = env.step(ReasoningGymAction(answer="test"))

        assert obs.done is True
        assert obs.score is None
        assert obs.correct_answer is None
        assert obs.reward == 0.0

    def test_dataset_iterator_wraps_around(self):
        """Test that dataset iterator restarts when exhausted."""
        env = ReasoningGymEnvironment()

        # Create small dataset
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=2,
        )

        # Get all questions
        questions = []
        for _ in range(3):  # More than dataset size
            obs = env.reset()
            questions.append(obs.question)

        # First question should repeat after wrapping
        assert questions[0] == questions[2]

    def test_state_property(self):
        """Test state property returns current state."""
        env = ReasoningGymEnvironment()

        obs = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
            episode_id="test-episode",
        )

        state = env.state
        assert state.episode_id == "test-episode"
        assert state.step_count == 0
        assert obs is not None

    def test_episode_id_generation(self):
        """Test that episode_id is auto-generated when not provided."""
        env = ReasoningGymEnvironment()

        obs = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        state = env.state
        assert state.episode_id is not None
        assert len(state.episode_id) > 0
        assert obs is not None

    def test_dataset_metadata_in_observation(self):
        """Test that dataset metadata is included in observation."""
        env = ReasoningGymEnvironment()
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        obs = env.step(ReasoningGymAction(answer="4"))

        # Metadata might be None if not provided by dataset
        assert obs.dataset_metadata is None or isinstance(obs.dataset_metadata, dict)

    def test_supports_concurrent_sessions(self):
        """Test that environment declares concurrent session support."""
        assert ReasoningGymEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True


class TestReasoningGymModels:
    """Tests for the data models."""

    def test_reasoning_gym_action(self):
        """Test ReasoningGymAction model."""
        action = ReasoningGymAction(answer="42")

        assert action.answer == "42"
        assert isinstance(action.answer, str)

    def test_reasoning_gym_observation_defaults(self):
        """Test ReasoningGymObservation default values."""
        obs = ReasoningGymObservation(
            done=False,
            reward=0.0,
        )

        assert obs.question is None
        assert obs.score is None
        assert obs.correct_answer is None
        assert obs.dataset_metadata is None
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reasoning_gym_observation_full(self):
        """Test ReasoningGymObservation with all fields."""
        obs = ReasoningGymObservation(
            question="What is 2+2?",
            score=1.0,
            correct_answer="4",
            done=True,
            reward=1.0,
            dataset_metadata={"difficulty": "easy"},
        )

        assert obs.question == "What is 2+2?"
        assert obs.score == 1.0
        assert obs.correct_answer == "4"
        assert obs.done is True
        assert obs.reward == 1.0
        assert obs.dataset_metadata == {"difficulty": "easy"}


class TestReasoningGymEnvClient:
    """Tests for the ReasoningGymEnv client."""

    def test_step_payload_conversion(self):
        """Test _step_payload converts action to dict."""
        from reasoning_gym_env import ReasoningGymEnv

        env = ReasoningGymEnv(base_url="http://localhost:8000")
        action = ReasoningGymAction(answer="test answer")

        payload = env._step_payload(action)

        assert isinstance(payload, dict)
        assert payload["answer"] == "test answer"

    def test_parse_result(self):
        """Test _parse_result parses server response."""
        from openenv.core.client_types import StepResult
        from reasoning_gym_env import ReasoningGymEnv

        env = ReasoningGymEnv(base_url="http://localhost:8000")

        payload = {
            "observation": {
                "question": "Test question?",
                "score": 0.8,
                "correct_answer": "correct",
                "metadata": {},
                "dataset_metadata": {"key": "value"},
            },
            "reward": 0.8,
            "done": True,
        }

        result = env._parse_result(payload)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, ReasoningGymObservation)
        assert result.observation.question == "Test question?"
        assert result.observation.score == 0.8
        assert result.observation.correct_answer == "correct"
        assert result.observation.done is True
        assert result.reward == 0.8
        assert result.done is True

    def test_parse_state(self):
        """Test _parse_state parses state response."""
        from openenv.core.env_server.types import State
        from reasoning_gym_env import ReasoningGymEnv

        env = ReasoningGymEnv(base_url="http://localhost:8000")

        payload = {
            "episode_id": "test-episode",
            "step_count": 5,
        }

        state = env._parse_state(payload)

        assert isinstance(state, State)
        assert state.episode_id == "test-episode"
        assert state.step_count == 5


class TestReasoningGymIntegration:
    """Integration tests for complete workflows."""

    def test_complete_episode_workflow(self):
        """Test a complete episode from reset to step."""
        env = ReasoningGymEnvironment()

        # Reset with dataset
        obs = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        assert obs.question is not None
        assert not obs.done
        episode_id = env.state.episode_id

        # Step with answer
        obs = env.step(ReasoningGymAction(answer="4"))

        assert obs.score is not None
        assert obs.correct_answer is not None
        assert obs.done is True
        assert env.state.step_count == 1

        # Episode ID should persist
        assert env.state.episode_id == episode_id

    def test_multiple_episodes_with_dataset_reuse(self):
        """Test multiple episodes reusing the same dataset."""
        env = ReasoningGymEnvironment()

        # Create dataset
        obs1 = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=10,
        )
        question1 = obs1.question

        # Complete first episode
        env.step(ReasoningGymAction(answer="4"))

        # Start second episode (reuse dataset)
        obs2 = env.reset()
        question2 = obs2.question

        assert question1 != question2
        assert env.state.step_count == 0  # Reset for new episode

    def test_dataset_recreation_with_new_params(self):
        """Test that providing new params recreates dataset."""
        env = ReasoningGymEnvironment()

        # Create first dataset
        env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=42,
            size=5,
        )

        # Create new dataset with different seed
        obs = env.reset(
            dataset_name="leg_counting",
            dataset_config={"min_animals": 5, "max_animals": 15},
            seed=99,
            size=5,
        )

        assert obs.question is not None
        assert env.state.step_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
