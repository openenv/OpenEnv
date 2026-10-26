# SPDX-License-Identifier: BSD-3-Clause

"""Tests for InspectAIHarness with mocked inspect_ai dependencies."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.evals import EvalConfig, EvalResult
from openenv.core.evals.inspect_harness import InspectAIHarness


# ---------------------------------------------------------------------------
# Helpers to build mock inspect_ai modules
# ---------------------------------------------------------------------------


def _make_mock_metric(name, value):
    """Build a mock EvalMetric with name and value attributes."""
    metric = MagicMock()
    metric.name = name
    metric.value = value
    return metric


def _make_mock_eval_score(metrics):
    """Build a mock EvalScore with a metrics dict.

    Args:
        metrics: List of (name, value) tuples.

    Returns:
        Mock EvalScore with metrics as ``dict[str, EvalMetric]``.
    """
    score = MagicMock()
    score.metrics = {name: _make_mock_metric(name, val) for name, val in metrics}
    return score


def _make_mock_eval_log(*, status="success", metrics=None, results=None):
    """Build a mock EvalLog object.

    Args:
        status: Log status string ("success" or "error").
        metrics: List of (name, value) tuples for a single scorer.
        results: Override results object (if None, built from metrics).
    """
    log = MagicMock()
    log.status = status

    if results is not None:
        log.results = results
    elif metrics is not None:
        mock_results = MagicMock()
        mock_results.scores = [_make_mock_eval_score(metrics)]
        log.results = mock_results
    else:
        log.results = None

    return log


def _make_mock_inspect_modules(*, eval_return=None):
    """Build a dict of mock modules that simulate inspect_ai's structure.

    Args:
        eval_return: Return value for inspect_ai.eval(). Defaults to a
            single successful log with {"accuracy": 0.85}.
    """
    if eval_return is None:
        eval_return = [_make_mock_eval_log(metrics=[("accuracy", 0.85)])]

    # Top-level
    inspect_ai_mod = ModuleType("inspect_ai")
    mock_eval = MagicMock(name="eval", return_value=eval_return)
    inspect_ai_mod.eval = mock_eval

    return {
        "inspect_ai": inspect_ai_mod,
    }, mock_eval


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestInspectAIHarnessConstruction:
    """Test instantiation and default values."""

    def test_default_construction(self):
        harness = InspectAIHarness()
        assert harness.log_dir is None

    def test_custom_construction(self):
        harness = InspectAIHarness(log_dir="/tmp/logs")
        assert harness.log_dir == "/tmp/logs"

    def test_name_property(self):
        harness = InspectAIHarness()
        assert harness.name == "InspectAIHarness"

    def test_is_eval_harness_subclass(self):
        from openenv.core.evals.base import EvalHarness

        assert issubclass(InspectAIHarness, EvalHarness)


class TestInspectAIHarnessImportGuard:
    """Test that run() raises a clear ImportError when inspect-ai is missing."""

    def test_import_error_message(self):
        harness = InspectAIHarness()
        with patch.dict(sys.modules, {"inspect_ai": None}):
            with pytest.raises(ImportError, match="inspect-ai is required"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={"model": "openai/gpt-4o"},
                )


class TestInspectAIHarnessRun:
    """Test the run() method with mocked inspect_ai."""

    def _run_harness(
        self, eval_parameters, dataset="mmlu", eval_return=None, **init_kwargs
    ):
        """Helper to run the harness with mocked inspect_ai modules."""
        mock_modules, mock_eval = _make_mock_inspect_modules(
            eval_return=eval_return,
        )

        harness = InspectAIHarness(**init_kwargs)
        with patch.dict(sys.modules, mock_modules):
            scores = harness.run(
                harness_version="0.3.0",
                library_versions={"openai": "1.0.0"},
                dataset=dataset,
                eval_parameters=eval_parameters,
            )

        return scores, mock_eval

    def test_basic_run_returns_scores(self):
        scores, _ = self._run_harness({"model": "openai/gpt-4o"})
        assert scores == {"accuracy": 0.85}

    def test_eval_called_with_correct_task_from_dataset(self):
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o"},
            dataset="hellaswag",
        )
        args, kwargs = mock_eval.call_args
        assert args[0] == "hellaswag"
        assert kwargs["model"] == "openai/gpt-4o"

    def test_task_parameter_overrides_dataset(self):
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "task": "gsm8k"},
            dataset="hellaswag",
        )
        args, _ = mock_eval.call_args
        assert args[0] == "gsm8k"

    def test_missing_model_raises_value_error(self):
        harness = InspectAIHarness()
        mock_modules, _ = _make_mock_inspect_modules()
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(ValueError, match="model"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={},
                )

    def test_optional_kwargs_passed_through(self):
        _, mock_eval = self._run_harness(
            {
                "model": "openai/gpt-4o",
                "max_samples": 100,
                "max_connections": 4,
                "temperature": 0.5,
                "max_tokens": 256,
                "epochs": 3,
                "model_base_url": "https://example.test/v1",
            }
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["max_samples"] == 100
        assert kwargs["max_connections"] == 4
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256
        assert kwargs["epochs"] == 3
        assert kwargs["model_base_url"] == "https://example.test/v1"

    def test_none_optional_kwargs_omitted(self):
        _, mock_eval = self._run_harness({"model": "openai/gpt-4o"})
        _, kwargs = mock_eval.call_args
        assert "max_samples" not in kwargs
        assert "max_connections" not in kwargs
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert "epochs" not in kwargs
        assert "model_base_url" not in kwargs

    def test_task_args_passed_through(self):
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "task_args": {"num_fewshot": 5}},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["task_args"] == {"num_fewshot": 5}

    def test_model_args_passed_through(self):
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "model_args": {"api_key": "test"}},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["model_args"] == {"api_key": "test"}

    def test_solver_passed_through(self):
        solver = ["chain_of_thought", "generate"]
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "solver": solver},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["solver"] == solver

    def test_scorer_passed_through(self):
        scorer = ["exact"]
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o", "scorer": scorer},
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["scorer"] == scorer

    def test_log_dir_passed_through(self):
        _, mock_eval = self._run_harness(
            {"model": "openai/gpt-4o"},
            log_dir="/tmp/logs",
        )
        _, kwargs = mock_eval.call_args
        assert kwargs["log_dir"] == "/tmp/logs"

    def test_error_status_raises_runtime_error(self):
        error_log = _make_mock_eval_log(status="error")
        harness = InspectAIHarness()
        mock_modules, _ = _make_mock_inspect_modules(eval_return=[error_log])
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(RuntimeError, match="failed with status"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={"model": "openai/gpt-4o"},
                )

    def test_empty_logs_raises_runtime_error(self):
        harness = InspectAIHarness()
        mock_modules, _ = _make_mock_inspect_modules(eval_return=[])
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(RuntimeError, match="returned no logs"):
                harness.run(
                    harness_version="0.3.0",
                    library_versions={},
                    dataset="mmlu",
                    eval_parameters={"model": "openai/gpt-4o"},
                )


class TestInspectAIHarnessScoreExtraction:
    """Test _extract_scores() parses EvalLog.results."""

    def test_extracts_single_metric(self):
        harness = InspectAIHarness()
        log = _make_mock_eval_log(metrics=[("accuracy", 0.92)])
        scores = harness._extract_scores(log)
        assert scores == {"accuracy": 0.92}

    def test_extracts_multiple_metrics(self):
        harness = InspectAIHarness()
        log = _make_mock_eval_log(
            metrics=[("accuracy", 0.85), ("f1", 0.88), ("stderr", 0.02)]
        )
        scores = harness._extract_scores(log)
        assert scores == {"accuracy": 0.85, "f1": 0.88, "stderr": 0.02}

    def test_returns_empty_dict_when_results_none(self):
        harness = InspectAIHarness()
        log = _make_mock_eval_log()
        assert log.results is None
        scores = harness._extract_scores(log)
        assert scores == {}

    def test_returns_empty_dict_when_no_metrics(self):
        harness = InspectAIHarness()
        # An EvalScore with an empty metrics dict
        log = _make_mock_eval_log(metrics=[])
        scores = harness._extract_scores(log)
        assert scores == {}


class TestInspectAIHarnessIntegration:
    """Test run_from_config produces correct EvalResult."""

    def test_run_from_config_returns_eval_result(self):
        eval_return = [
            _make_mock_eval_log(metrics=[("accuracy", 0.85), ("stderr", 0.02)])
        ]
        mock_modules, _ = _make_mock_inspect_modules(eval_return=eval_return)

        harness = InspectAIHarness()
        config = EvalConfig(
            harness_name="InspectAIHarness",
            harness_version="0.3.0",
            library_versions={"openai": "1.0.0"},
            dataset="mmlu",
            eval_parameters={"model": "openai/gpt-4o"},
        )

        with patch.dict(sys.modules, mock_modules):
            result = harness.run_from_config(config)

        assert isinstance(result, EvalResult)
        assert result.config is config
        assert result.scores["accuracy"] == 0.85
        assert result.scores["stderr"] == 0.02
