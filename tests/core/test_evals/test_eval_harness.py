# SPDX-License-Identifier: BSD-3-Clause

"""Tests for EvalHarness ABC."""

from typing import Any

import pytest
from openenv.core.evals import EvalConfig, EvalHarness, EvalResult


class ConcreteEvalHarness(EvalHarness):
    """Concrete implementation of EvalHarness for testing."""

    def __init__(self, return_scores: dict[str, Any] | None = None):
        self.return_scores = (
            return_scores if return_scores is not None else {"acc": 0.85}
        )
        self.run_called = False
        self.last_config = None

    def run(
        self,
        harness_version: str,
        library_versions: dict[str, str],
        dataset: str,
        eval_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the evaluation and return scores."""
        self.run_called = True
        self.last_config = {
            "harness_version": harness_version,
            "library_versions": library_versions,
            "dataset": dataset,
            "eval_parameters": eval_parameters,
        }
        return self.return_scores


class TestEvalHarnessABC:
    """Tests for EvalHarness ABC."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that EvalHarness cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EvalHarness()

    def test_concrete_implementation_works(self):
        """Test that concrete implementations work."""
        harness = ConcreteEvalHarness(return_scores={"acc": 0.9})
        result = harness.run(
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        assert result == {"acc": 0.9}
        assert harness.run_called

    def test_run_method_signature(self):
        """Test that run() accepts the correct parameters."""
        harness = ConcreteEvalHarness()
        scores = harness.run(
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0", "torch": "2.1.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5, "limit": 100},
        )
        assert isinstance(scores, dict)
        assert harness.last_config["harness_version"] == "0.4.0"
        assert harness.last_config["dataset"] == "hellaswag"


class TestEvalHarnessIntegration:
    """Tests for EvalHarness integration with EvalConfig and EvalResult."""

    def test_run_from_config_method(self):
        """Test run_from_config() method creates EvalResult from EvalConfig."""
        harness = ConcreteEvalHarness(return_scores={"acc": 0.85, "acc_stderr": 0.02})
        config = EvalConfig(
            harness_name="test_harness",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )

        result = harness.run_from_config(config)

        assert isinstance(result, EvalResult)
        assert result.config == config
        assert result.scores["acc"] == 0.85
        assert result.scores["acc_stderr"] == 0.02

    def test_run_from_config_passes_parameters_correctly(self):
        """Test that run_from_config extracts and passes config fields to run()."""
        harness = ConcreteEvalHarness()
        config = EvalConfig(
            harness_name="test_harness",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )

        harness.run_from_config(config)

        assert harness.last_config["harness_version"] == "0.4.0"
        assert harness.last_config["library_versions"] == {"transformers": "4.36.0"}
        assert harness.last_config["dataset"] == "hellaswag"
        assert harness.last_config["eval_parameters"] == {"num_fewshot": 5}

    def test_run_from_config_preserves_config_in_result(self):
        """Test that run_from_config preserves the original config in result."""
        harness = ConcreteEvalHarness(return_scores={"acc": 0.9})
        config = EvalConfig(
            harness_name="test_harness",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )

        result = harness.run_from_config(config)

        # Result should contain the exact same config object
        assert result.config is config


class TestEvalHarnessErrorHandling:
    """Tests for error handling in EvalHarness."""

    def test_run_with_empty_library_versions(self):
        """Test run() works with empty library_versions dict."""
        harness = ConcreteEvalHarness()
        scores = harness.run(
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        assert isinstance(scores, dict)

    def test_run_with_empty_eval_parameters(self):
        """Test run() works with empty eval_parameters dict."""
        harness = ConcreteEvalHarness()
        scores = harness.run(
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={},
        )
        assert isinstance(scores, dict)

    def test_run_returns_empty_scores(self):
        """Test that run() can return empty scores dict."""
        harness = ConcreteEvalHarness(return_scores={})
        scores = harness.run(
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        assert scores == {}


class TestEvalHarnessName:
    """Tests for EvalHarness name property."""

    def test_name_property_returns_class_name(self):
        """Test that name property returns the class name."""
        harness = ConcreteEvalHarness()
        assert harness.name == "ConcreteEvalHarness"

    def test_name_property_for_custom_harness(self):
        """Test that name property works for any subclass."""

        class CustomHarness(EvalHarness):
            def run(self, harness_version, library_versions, dataset, eval_parameters):
                return {"acc": 1.0}

        harness = CustomHarness()
        assert harness.name == "CustomHarness"


class TestEvalHarnessReproducibility:
    """Tests for reproducibility verification."""

    def test_run_with_same_config_should_be_reproducible(self):
        """Test that running with identical config params should be deterministic."""
        harness = ConcreteEvalHarness(return_scores={"acc": 0.85})

        scores1 = harness.run(
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5, "seed": 42},
        )

        scores2 = harness.run(
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5, "seed": 42},
        )

        # Same config should produce same scores
        assert scores1 == scores2

    def test_config_captures_all_reproducibility_parameters(self):
        """Test that EvalConfig captures all parameters needed for reproducibility."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0", "torch": "2.1.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5, "seed": 42, "limit": 100},
        )

        # All parameters that affect results should be in config
        assert config.harness_version == "0.4.0"
        assert "transformers" in config.library_versions
        assert "torch" in config.library_versions
        assert config.eval_parameters["seed"] == 42
        assert config.eval_parameters["num_fewshot"] == 5
