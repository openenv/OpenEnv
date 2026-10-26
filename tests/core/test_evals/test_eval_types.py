# SPDX-License-Identifier: BSD-3-Clause

"""Tests for EvalConfig and EvalResult Pydantic models."""

import pytest
from openenv.core.evals import EvalConfig, EvalResult
from pydantic import ValidationError


class TestEvalConfig:
    """Tests for EvalConfig model."""

    def test_eval_config_creation(self):
        """Test creating a valid EvalConfig."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0", "torch": "2.1.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5, "limit": 100},
        )
        assert config.harness_name == "lm_eval"
        assert config.harness_version == "0.4.0"
        assert config.dataset == "hellaswag"
        assert config.eval_parameters["num_fewshot"] == 5

    def test_eval_config_requires_all_fields(self):
        """Test that EvalConfig requires all mandatory fields."""
        with pytest.raises(ValidationError):
            EvalConfig()  # Missing all required fields

    def test_eval_config_rejects_extra_fields(self):
        """Test that EvalConfig forbids unknown fields."""
        with pytest.raises(ValidationError):
            EvalConfig(
                harness_name="lm_eval",
                harness_version="0.4.0",
                library_versions={},
                dataset="hellaswag",
                eval_parameters={},
                unknown_field="should_fail",  # Should be rejected
            )

    def test_eval_config_library_versions_dict(self):
        """Test that library_versions must be a dict."""
        with pytest.raises(ValidationError):
            EvalConfig(
                harness_name="lm_eval",
                harness_version="0.4.0",
                library_versions="invalid",  # Must be dict
                dataset="hellaswag",
                eval_parameters={},
            )

    def test_eval_config_eval_parameters_dict(self):
        """Test that eval_parameters must be a dict."""
        with pytest.raises(ValidationError):
            EvalConfig(
                harness_name="lm_eval",
                harness_version="0.4.0",
                library_versions={},
                dataset="hellaswag",
                eval_parameters="invalid",  # Must be dict
            )

    def test_eval_config_serialization(self):
        """Test EvalConfig can be serialized to dict."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        data = config.model_dump()
        assert data["harness_name"] == "lm_eval"
        assert data["harness_version"] == "0.4.0"
        assert data["library_versions"]["transformers"] == "4.36.0"
        assert data["eval_parameters"]["num_fewshot"] == 5

    def test_eval_config_deserialization(self):
        """Test EvalConfig can be created from dict."""
        data = {
            "harness_name": "lm_eval",
            "harness_version": "0.4.0",
            "library_versions": {"transformers": "4.36.0"},
            "dataset": "hellaswag",
            "eval_parameters": {"num_fewshot": 5},
        }
        config = EvalConfig(**data)
        assert config.harness_name == "lm_eval"
        assert config.library_versions["transformers"] == "4.36.0"

    def test_eval_config_empty_dicts_allowed(self):
        """Test that empty library_versions and eval_parameters are allowed."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        assert config.library_versions == {}
        assert config.eval_parameters == {}

    def test_eval_config_nested_eval_parameters(self):
        """Test that eval_parameters can contain nested structures."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={
                "model_args": {"device": "cuda", "batch_size": 16},
                "limit": 100,
            },
        )
        assert config.eval_parameters["model_args"]["device"] == "cuda"
        assert config.eval_parameters["limit"] == 100


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_eval_result_creation(self):
        """Test creating a valid EvalResult."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        result = EvalResult(
            config=config,
            scores={"acc": 0.85, "acc_stderr": 0.02},
        )
        assert result.config.harness_name == "lm_eval"
        assert result.scores["acc"] == 0.85
        assert result.scores["acc_stderr"] == 0.02

    def test_eval_result_requires_config(self):
        """Test that EvalResult requires config field."""
        with pytest.raises(ValidationError):
            EvalResult(scores={"acc": 0.85})  # Missing config

    def test_eval_result_requires_scores(self):
        """Test that EvalResult requires scores field."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        with pytest.raises(ValidationError):
            EvalResult(config=config)  # Missing scores

    def test_eval_result_rejects_extra_fields(self):
        """Test that EvalResult forbids unknown fields."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        with pytest.raises(ValidationError):
            EvalResult(
                config=config,
                scores={"acc": 0.85},
                unknown_field="should_fail",  # Should be rejected
            )

    def test_eval_result_scores_dict(self):
        """Test that scores must be a dict."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        with pytest.raises(ValidationError):
            EvalResult(
                config=config,
                scores="invalid",  # Must be dict
            )

    def test_eval_result_serialization(self):
        """Test EvalResult can be serialized to dict."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        result = EvalResult(
            config=config,
            scores={"acc": 0.85, "acc_stderr": 0.02},
        )
        data = result.model_dump()
        assert data["config"]["harness_name"] == "lm_eval"
        assert data["scores"]["acc"] == 0.85

    def test_eval_result_deserialization(self):
        """Test EvalResult can be created from dict."""
        data = {
            "config": {
                "harness_name": "lm_eval",
                "harness_version": "0.4.0",
                "library_versions": {"transformers": "4.36.0"},
                "dataset": "hellaswag",
                "eval_parameters": {"num_fewshot": 5},
            },
            "scores": {"acc": 0.85, "acc_stderr": 0.02},
        }
        result = EvalResult(**data)
        assert result.config.harness_name == "lm_eval"
        assert result.scores["acc"] == 0.85

    def test_eval_result_scores_supports_various_types(self):
        """Test that scores can contain int, float, bool, None values."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        result = EvalResult(
            config=config,
            scores={
                "acc": 0.85,
                "num_samples": 1000,
                "passed": True,
                "error": None,
            },
        )
        assert result.scores["acc"] == 0.85
        assert result.scores["num_samples"] == 1000
        assert result.scores["passed"] is True
        assert result.scores["error"] is None

    def test_eval_result_empty_scores_allowed(self):
        """Test that empty scores dict is allowed."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        result = EvalResult(config=config, scores={})
        assert result.scores == {}

    def test_eval_result_nested_scores(self):
        """Test that scores can contain nested structures."""
        config = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        result = EvalResult(
            config=config,
            scores={
                "overall": {"acc": 0.85, "acc_stderr": 0.02},
                "per_task": {"task1": 0.9, "task2": 0.8},
            },
        )
        assert result.scores["overall"]["acc"] == 0.85
        assert result.scores["per_task"]["task1"] == 0.9


class TestEvalConfigEqualityAndHashing:
    """Test EvalConfig equality for reproducibility checks."""

    def test_equal_configs_are_equal(self):
        """Test that identical configs are equal."""
        config1 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        config2 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        assert config1 == config2

    def test_different_harness_version_not_equal(self):
        """Test that configs with different harness versions are not equal."""
        config1 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        config2 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.1",  # Different version
            library_versions={},
            dataset="hellaswag",
            eval_parameters={},
        )
        assert config1 != config2

    def test_different_library_versions_not_equal(self):
        """Test that configs with different library versions are not equal."""
        config1 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.36.0"},
            dataset="hellaswag",
            eval_parameters={},
        )
        config2 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={"transformers": "4.37.0"},  # Different version
            dataset="hellaswag",
            eval_parameters={},
        )
        assert config1 != config2

    def test_different_eval_parameters_not_equal(self):
        """Test that configs with different eval parameters are not equal."""
        config1 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 5},
        )
        config2 = EvalConfig(
            harness_name="lm_eval",
            harness_version="0.4.0",
            library_versions={},
            dataset="hellaswag",
            eval_parameters={"num_fewshot": 10},  # Different parameter
        )
        assert config1 != config2
