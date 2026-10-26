# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the QED Math Environment.

Unit tests (no network/Docker required):
  - parse_schema, length_penalty, apply_score_threshold
  - MathProofRubric: normalize_reward, _parse_response, _build_prompt,
    grade() with mocked LLM, empty-proof fast path, retry/failure paths
  - QEDMathEnvironment: reset, get_problem_payload, _verify_math,
    _grade_answer_submission, _apply_reward_shaping, _strip_reasoning,
    submit_proof_payload (mocked grader), multi-step problems,
    original_problem field, verifier metrics, dataset loading
  - MCP tool registration (get_problem, submit_proof, get_grading_guidelines)
  - Models: ProblemObservation, ProofSubmissionObservation

Integration tests (require running server, marked @pytest.mark.integration):
  - Full HTTP/WebSocket stack smoke test

Run from repo root:
    PYTHONPATH=src:envs uv run pytest tests/envs/test_qed_math_environment.py -v
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Path setup — ensure src/ and envs/ are importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "envs"))


# Autouse fixture: patch openai.AsyncOpenAI so tests that instantiate
# QEDMathEnvironment (whose MathProofRubric builds an AsyncOpenAI client on first
# use) can grade without a real OPENAI_API_KEY in the environment.
@pytest.fixture(autouse=True)
def _patch_openai_client(monkeypatch):
    """Replace openai.AsyncOpenAI with a MagicMock for the duration of each test."""
    mock_client = MagicMock()
    # Ensure async methods on the mock return coroutines
    mock_client.responses.create = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: mock_client)
    yield mock_client


# Guard: skip entire module if required packages are missing
pytest.importorskip("math_verify", reason="math_verify is not installed")
pytest.importorskip("fastmcp", reason="fastmcp is not installed")

from openenv.core.env_server.mcp_environment import get_server_tools  # noqa: E402
from openenv.core.env_server.types import State  # noqa: E402
from qed_math_env.models import (  # noqa: E402
    ProblemObservation,
    ProofSubmissionObservation,
)
from qed_math_env.server.math_verify_service import (  # noqa: E402
    _verify_answer_worker,
    MathVerifierService,
    VerifyRequest,
    VerifyResponse,
)
from qed_math_env.server.qed_math_environment import (  # noqa: E402
    _normalize_problem,
    answer_reward_for_status,
    load_evaluator_prompt,
    load_problems,
    QEDMathConfig,
    QEDMathEnvironment,
    remove_reasoning,
)
from qed_math_env.server.rubric import (  # noqa: E402
    apply_score_threshold,
    GradingResult,
    length_penalty,
    MathProofRubric,
    MAX_SCORE,
    parse_schema,
)


# Helpers


BOOTSTRAP_PROBLEM = {
    "problem": "Prove that the sum of two even integers is even.",
    "reference_solution": "Let a=2m and b=2n. Then a+b=2(m+n), so it is even.",
    "grading_guidelines": "Award full credit for a correct parity argument.",
    "problem_id": "bootstrap_000001",
    "dataset_source": "bootstrap",
    "problem_type": "proof",
    "max_attempts": 1,
}

ANSWER_PROBLEM = {
    "problem": "What is 2+2?",
    "reference_solution": "4",
    "grading_guidelines": "",
    "problem_id": "answer_001",
    "dataset_source": "test",
    "problem_type": "answer",
    "evaluation_mode": "answer",
    "max_attempts": 1,
}

MULTI_STEP_PROBLEM = {
    "problem": "Prove Fermat's Last Theorem.",
    "reference_solution": "Wiles 1995.",
    "grading_guidelines": "",
    "problem_id": "multi_001",
    "dataset_source": "test",
    "problem_type": "multi_step",
    "max_attempts": 3,
}


def _make_env(**kwargs) -> QEDMathEnvironment:
    """Create an environment with bootstrap problems (no dataset needed)."""
    return QEDMathEnvironment(**kwargs)


def _make_env_with_problem(raw_problem: dict) -> QEDMathEnvironment:
    """Create an environment pre-loaded with a single synthetic problem."""
    env = _make_env()
    normalized = _normalize_problem(
        raw_problem, 0, raw_problem.get("dataset_source", "test")
    )
    env._problems = [normalized]
    env._gold_cache_problem_count = len(env._problems)
    env._build_gold_answer_cache()
    return env


# parse_schema


class TestParseSchema:
    def test_string_passthrough(self):
        assert parse_schema("plain text") == "plain text"

    def test_empty_string(self):
        assert parse_schema("") == ""

    def test_list_of_criteria(self):
        schema = [
            {"title": "Correctness", "points": 4, "desc": "Is it correct?"},
            {"title": "Clarity", "points": 3, "description": "Is it clear?"},
        ]
        result = parse_schema(schema)
        assert "# Correctness (4 points)" in result
        assert "# Clarity (3 points)" in result
        assert "Is it correct?" in result
        assert "Is it clear?" in result

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            parse_schema(42)

    def test_list_missing_keys_raises(self):
        with pytest.raises(ValueError):
            parse_schema([{"title": "X"}])  # missing points/desc

    def test_list_non_dict_entry_raises(self):
        with pytest.raises(ValueError):
            parse_schema(["not a dict"])


# length_penalty


class TestLengthPenalty:
    def test_zero_buffer_always_zero(self):
        assert length_penalty(1000, 999, 0) == 0.0
        assert length_penalty(1000, 1, 0) == 0.0

    def test_inside_buffer_zone(self):
        # buffer=100, max=1000 → penalty zone is 900-1000
        penalty = length_penalty(1000, 950, 100)
        assert penalty < 0.0

    def test_outside_buffer_zone(self):
        # sequence_length <= max_length - buffer_tokens → no penalty
        assert length_penalty(1000, 800, 100) == 0.0

    def test_at_boundary(self):
        # sequence_length == max_length - buffer_tokens → no penalty (boundary inclusive at 0)
        assert length_penalty(1000, 900, 100) == 0.0

    def test_penalty_increases_toward_max(self):
        p1 = length_penalty(1000, 910, 100)
        p2 = length_penalty(1000, 960, 100)
        assert p2 < p1  # closer to max → more negative


# apply_score_threshold


class TestApplyScoreThreshold:
    def test_zero(self):
        assert apply_score_threshold(0) == 0.0

    def test_partial_credit_collapses_to_one(self):
        for s in [1, 2, 3, 4, 5]:
            assert apply_score_threshold(float(s)) == 1.0

    def test_high_scores_pass_through(self):
        assert apply_score_threshold(6.0) == 6.0
        assert apply_score_threshold(7.0) == 7.0


# MathProofRubric unit tests


class TestMathProofRubricNormalizeReward:
    def test_score_zero(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        assert rubric.normalize_reward(0) == pytest.approx(0.0)

    def test_score_max(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        assert rubric.normalize_reward(MAX_SCORE) == pytest.approx(1.0)

    def test_score_mid(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        assert rubric.normalize_reward(3) == pytest.approx(3 / 7)

    def test_custom_threshold_collapses_partial(self):
        rubric = MathProofRubric(
            grader_model="test", api_key="x", custom_threshold=True
        )
        # scores 1-5 collapse to 1, normalized to 1/7
        for score in [1, 2, 3, 4, 5]:
            assert rubric.normalize_reward(score) == pytest.approx(1 / 7)

    def test_custom_threshold_preserves_high(self):
        rubric = MathProofRubric(
            grader_model="test", api_key="x", custom_threshold=True
        )
        assert rubric.normalize_reward(6) == pytest.approx(6 / 7)
        assert rubric.normalize_reward(7) == pytest.approx(1.0)


def test_grader_client_created_on_first_use(monkeypatch, _patch_openai_client):
    """The environment starts without grader credentials, the client is built lazily."""

    def missing_credentials(**kwargs):
        raise RuntimeError("Missing credentials")

    monkeypatch.setattr("openai.AsyncOpenAI", missing_credentials)
    env = _make_env()
    env.reset()

    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: _patch_openai_client)
    assert env._rubric._client is _patch_openai_client


class TestMathProofRubricParseResponse:
    def _rubric(self):
        return MathProofRubric(grader_model="test", api_key="x")

    def test_parses_score_tag(self):
        rubric = self._rubric()
        score, feedback = rubric._parse_response("Good proof. <score>5</score>")
        assert score == 5
        assert "<score>5</score>" in feedback

    def test_clamps_above_max(self):
        rubric = self._rubric()
        score, _ = rubric._parse_response("<score>99</score>")
        assert score == MAX_SCORE

    def test_clamps_below_zero(self):
        rubric = self._rubric()
        # Negative values are clamped by max(0, ...) — regex only matches digits
        # so a missing tag falls through to 0.
        score, _ = rubric._parse_response("<score>0</score>")
        assert score == 0

    def test_missing_tag_returns_zero(self):
        rubric = self._rubric()
        score, feedback = rubric._parse_response("No tag here.")
        assert score == 0
        assert feedback == "No tag here."


class TestMathProofRubricBuildPrompt:
    def _rubric(self, template: str = "") -> MathProofRubric:
        return MathProofRubric(
            grader_model="test", api_key="x", prompt_template=template
        )

    def test_fallback_prompt_contains_proof(self):
        rubric = self._rubric()
        prompt = rubric._build_prompt("My proof", "Problem X", "Ref sol", "Rubric")
        assert "My proof" in prompt
        assert "Problem X" in prompt

    def test_template_substitution(self):
        template = "P:{problem} S:{solution} R:{marking_scheme} H:{human_solution}"
        rubric = self._rubric(template)
        prompt = rubric._build_prompt("my_proof", "my_problem", "my_ref", "my_rubric")
        assert "my_proof" in prompt
        assert "my_problem" in prompt
        assert "my_rubric" in prompt
        assert "my_ref" in prompt


class TestMathProofRubricGrade:
    """Tests for MathProofRubric.grade() — LLM calls are mocked."""

    def _rubric(self, **kwargs) -> MathProofRubric:
        return MathProofRubric(grader_model="test", api_key="x", **kwargs)

    @pytest.mark.asyncio
    async def test_successful_grade(self):
        rubric = self._rubric()
        response_text = "The proof is correct. <score>7</score>"
        with patch.object(
            rubric, "_call_llm", new=AsyncMock(return_value=response_text)
        ):
            result = await rubric.grade("Good proof.", "Problem", "Ref", "Guidelines")
        assert result.score == 7
        assert result.reward == pytest.approx(1.0)
        assert result.metrics["verifier/rollouts/success"] == 1

    @pytest.mark.asyncio
    async def test_score_normalized(self):
        rubric = self._rubric()
        response_text = "<score>3</score>"
        with patch.object(
            rubric, "_call_llm", new=AsyncMock(return_value=response_text)
        ):
            result = await rubric.grade("Partial proof.", "P", "R", "G")
        assert result.score == 3
        assert result.reward == pytest.approx(3 / 7)

    @pytest.mark.asyncio
    async def test_empty_proof_fast_path(self):
        rubric = self._rubric()
        result = await rubric.grade("   ", "Problem", "Ref", "Guidelines")
        assert result.score == 0
        assert result.reward == 0.0
        assert result.metrics["verifier/rollouts/failure"] == 1
        assert result.metrics["verifier/failures/no_input"] == 1

    @pytest.mark.asyncio
    async def test_missing_score_tag_records_metric(self):
        rubric = self._rubric()
        response_text = "Looks good but I forgot the tag."
        with patch.object(
            rubric, "_call_llm", new=AsyncMock(return_value=response_text)
        ):
            result = await rubric.grade("Some proof.", "P", "R", "G")
        assert result.metrics["verifier/failures/no_score_tag"] == 1
        assert result.metrics["verifier/rollouts/failure"] == 1

    @pytest.mark.asyncio
    async def test_all_attempts_exhausted(self):
        import openai as _openai

        rubric = self._rubric(max_retries=2, retry_backoff=[0, 0])
        with patch.object(
            rubric,
            "_call_llm",
            new=AsyncMock(
                side_effect=_openai.RateLimitError(
                    "rate", response=MagicMock(), body={}
                )
            ),
        ):
            result = await rubric.grade("My proof.", "P", "R", "G")
        assert result.score == 0
        assert result.reward == 0.0
        assert result.metrics["verifier/failures/all_attempts_failed"] == 1
        assert result.metrics["verifier/failures/rate_limit"] >= 1

    @pytest.mark.asyncio
    async def test_custom_threshold_applied(self):
        rubric = self._rubric(custom_threshold=True)
        response_text = "<score>4</score>"
        with patch.object(
            rubric, "_call_llm", new=AsyncMock(return_value=response_text)
        ):
            result = await rubric.grade("Partial proof.", "P", "R", "G")
        # Score 4 → thresholded to 1 → reward 1/7
        assert result.reward == pytest.approx(1 / 7)

    @pytest.mark.asyncio
    async def test_verifier_metrics_present(self):
        rubric = self._rubric()
        with patch.object(
            rubric, "_call_llm", new=AsyncMock(return_value="<score>5</score>")
        ):
            result = await rubric.grade("proof", "P", "R", "G")
        expected_keys = [
            "verifier/rollouts/success",
            "verifier/rollouts/failure",
            "verifier/failures/timeout",
            "verifier/failures/rate_limit",
            "verifier/failures/no_score_tag",
            "verifier/failures/num_retries",
            "verifier/runtime/latency_per_request",
            "verifier/runtime/input_tokens",
            "verifier/runtime/output_tokens",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing metric: {key}"


# remove_reasoning


class TestRemoveReasoning:
    def test_no_delimiter_returns_original(self):
        assert remove_reasoning("hello world") == "hello world"

    def test_empty_delimiters_returns_original(self):
        assert remove_reasoning("hello", []) == "hello"

    def test_delimiter_present_strips_before(self):
        text = "<think>step1 step2</think>Final answer."
        result = remove_reasoning(text, ["</think>"])
        assert result == "Final answer."

    def test_delimiter_absent_returns_empty_string(self):
        # When delimiters are configured but none present → empty (signal to grade full text)
        result = remove_reasoning("No think tag here.", ["</think>"])
        assert result == ""

    def test_multiple_delimiters_first_match_wins(self):
        text = "<think>reasoning</think>answer"
        result = remove_reasoning(text, ["</think>", "</reason>"])
        assert result == "answer"


# QEDMathEnvironment: dataset loading


class TestLoadProblems:
    def test_bootstrap_when_no_dataset(self):
        problems = load_problems(None)
        assert len(problems) >= 1
        assert all("problem" in p for p in problems)

    def test_load_local_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "problems.jsonl"
        rows = [
            {
                "problem": "Prove 1+1=2.",
                "reference_solution": "By definition.",
                "problem_id": "p001",
            },
            {
                "problem": "Prove 2+2=4.",
                "reference_solution": "By arithmetic.",
                "problem_id": "p002",
            },
        ]
        with jsonl_file.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        problems = load_problems(str(jsonl_file))
        assert len(problems) == 2
        assert problems[0]["problem"] == "Prove 1+1=2."
        assert problems[1]["problem_id"] == "p002"

    def test_load_local_json_list(self, tmp_path):
        json_file = tmp_path / "problems.json"
        rows = [
            {"problem": "P1", "reference_solution": "S1"},
            {"problem": "P2", "reference_solution": "S2"},
        ]
        json_file.write_text(json.dumps(rows))
        problems = load_problems(str(json_file))
        assert len(problems) == 2

    def test_answer_mode_boxed_wrapping(self, tmp_path):
        jsonl_file = tmp_path / "answers.jsonl"
        row = {
            "problem": "2+2=?",
            "reference_solution": "4",
            "evaluation_mode": "answer",
        }
        jsonl_file.write_text(json.dumps(row) + "\n")
        problems = load_problems(str(jsonl_file))
        # Answer-mode reference solution should be wrapped in \boxed{}
        assert "\\boxed{4}" in problems[0]["reference_solution"]

    def test_select_by_problem_id(self, tmp_path):
        jsonl_file = tmp_path / "probs.jsonl"
        rows = [
            {"problem": "P1", "problem_id": "id_alpha"},
            {"problem": "P2", "problem_id": "id_beta"},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in rows))
        env = QEDMathEnvironment(dataset_path=str(jsonl_file))
        obs = env.reset(problem_id="id_beta")
        assert isinstance(obs, ProblemObservation)
        assert obs.problem_id == "id_beta"
        assert obs.problem == "P2"


# QEDMathEnvironment: reset


class TestQEDMathEnvironmentReset:
    def test_reset_returns_problem_observation(self):
        env = _make_env()
        obs = env.reset()
        assert isinstance(obs, ProblemObservation)
        assert obs.problem != ""
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_fields_populated(self):
        env = _make_env()
        obs = env.reset()
        assert obs.problem_id != ""
        assert obs.dataset_source != ""
        assert obs.max_attempts >= 1

    def test_reset_increments_counter(self):
        env = _make_env()
        env.reset()
        env.reset()
        assert env._reset_count == 2

    def test_reset_clears_attempt_count(self):
        env = _make_env()
        env.reset()
        env._attempt_count = 99
        env.reset()
        assert env._attempt_count == 0


# QEDMathEnvironment: get_problem_payload / get_grading_guidelines_payload


class TestGetProblemPayload:
    def test_without_reset_returns_error(self):
        env = _make_env()
        payload = env.get_problem_payload()
        assert "error" in payload

    def test_after_reset_returns_problem(self):
        env = _make_env()
        env.reset()
        payload = env.get_problem_payload()
        assert "problem" in payload
        assert "error" not in payload
        assert "problem_id" in payload
        assert "reference_solution" not in payload

    def test_answer_mode_includes_reference_solution(self):
        env = _make_env_with_problem(ANSWER_PROBLEM)
        env.reset()
        payload = env.get_problem_payload()
        assert payload["reference_solution"] == r"\boxed{4}"

    def test_grading_guidelines_payload_without_reset(self):
        env = _make_env()
        payload = env.get_grading_guidelines_payload()
        assert "error" in payload

    def test_grading_guidelines_payload_after_reset(self):
        env = _make_env()
        env.reset()
        payload = env.get_grading_guidelines_payload()
        assert "grading_guidelines" in payload
        assert "error" not in payload


# QEDMathEnvironment: _verify_math (answer-mode verification)


class TestVerifyMath:
    def test_correct_answer(self):
        result = QEDMathEnvironment._verify_math(r"\boxed{4}", r"\boxed{4}")
        assert result == "correct"

    def test_wrong_answer(self):
        result = QEDMathEnvironment._verify_math(r"\boxed{5}", r"\boxed{4}")
        assert result == "wrong"

    def test_no_boxed_returns_no_answer(self):
        result = QEDMathEnvironment._verify_math("The answer is 4.", r"\boxed{4}")
        assert result == "no_answer"

    def test_empty_boxed_returns_unparsable(self):
        result = QEDMathEnvironment._verify_math(r"\boxed{}", r"\boxed{4}")
        assert result == "unparsable"


# QEDMathEnvironment: _grade_answer_submission


class TestGradeAnswerSubmission:
    @pytest.mark.asyncio
    async def test_correct_gives_max_reward(self):
        env = _make_env()
        try:
            result = await env._grade_answer_submission(r"\boxed{4}", r"\boxed{4}")
            assert result.score == 7
            assert result.reward == pytest.approx(1.0)
            assert result.metrics["verifier/rollouts/success"] == 1
        finally:
            await env._verifier_service.stop()

    @pytest.mark.asyncio
    async def test_wrong_gives_zero(self):
        env = _make_env()
        try:
            result = await env._grade_answer_submission(r"\boxed{5}", r"\boxed{4}")
            assert result.score == 0
            assert result.reward == pytest.approx(0.0)
            assert result.metrics["verifier/rollouts/failure"] == 1
        finally:
            await env._verifier_service.stop()

    @pytest.mark.asyncio
    async def test_missing_boxed_gives_zero(self):
        env = _make_env()
        try:
            result = await env._grade_answer_submission(
                "The answer is 4.", r"\boxed{4}"
            )
            assert result.score == 0
            assert result.reward == pytest.approx(0.0)
        finally:
            await env._verifier_service.stop()

    @pytest.mark.asyncio
    async def test_answer_grading_uses_verifier_service(self):
        env = _make_env()
        try:
            result = await env._grade_answer_submission(r"\boxed{4}", r"\boxed{4}")
            assert result.score == 7
            assert result.reward == pytest.approx(1.0)
            assert env._verifier_service._executor is not None
        finally:
            await env._verifier_service.stop()

    @pytest.mark.asyncio
    async def test_answer_grading_wrong_answer_via_service(self):
        env = _make_env()
        try:
            result = await env._grade_answer_submission(r"\boxed{5}", r"\boxed{4}")
            assert result.score == 0
            assert result.reward == pytest.approx(0.0)
        finally:
            await env._verifier_service.stop()

    @pytest.mark.asyncio
    async def test_timeout_metrics_from_service(self):
        env = _make_env()
        try:
            timeout_response = VerifyResponse(
                request_id="req-timeout",
                status="timeout",
                elapsed_ms=12.5,
                retry_count=1,
                error_type="ClientTimeout",
                error_message="simulated timeout",
            )
            with patch.object(
                env._verifier_service,
                "verify_answer",
                new=AsyncMock(return_value=timeout_response),
            ):
                result = await env._grade_answer_submission(r"\boxed{4}", r"\boxed{4}")
                assert result.score == 0
                assert result.metrics["verifier/failures/timeout"] == 1
                assert result.metrics["verifier/failures/num_retries"] == 1
                assert result.metrics[
                    "verifier/runtime/latency_per_request"
                ] == pytest.approx(12.5)
                assert "verifier/workers/restart_count" in result.metrics
                assert "verifier/workers/worker_restarted" in result.metrics
                assert "verifier/queue/depth" in result.metrics
        finally:
            await env._verifier_service.stop()

    @pytest.mark.asyncio
    async def test_shutdown_verifier_service_hook(self):
        env = _make_env()
        await env._grade_answer_submission(r"\boxed{4}", r"\boxed{4}")
        assert env._verifier_service._executor is not None
        await env.shutdown_verifier_service()
        assert env._verifier_service._executor is None


# QEDMathEnvironment: reward shaping


class TestRewardShaping:
    def test_no_tokens_no_shaping(self):
        env = _make_env()
        env._discount_factor = 0.99
        assert env._apply_reward_shaping(1.0, 0) == pytest.approx(1.0)

    def test_discount_factor_applied(self):
        env = _make_env()
        env._discount_factor = 0.99
        shaped = env._apply_reward_shaping(1.0, 10)
        assert shaped == pytest.approx(0.99**10)

    def test_discount_decreases_reward_with_tokens(self):
        env = _make_env()
        env._discount_factor = 0.999
        r1 = env._apply_reward_shaping(1.0, 100)
        r2 = env._apply_reward_shaping(1.0, 500)
        assert r2 < r1

    def test_length_penalty_applied_inside_buffer(self):
        env = _make_env()
        env._discount_factor = 1.0
        env._max_tokens = 1000
        env._buffer_tokens = 100
        # Token count inside penalty zone (950 > 900)
        shaped = env._apply_reward_shaping(1.0, 950)
        assert shaped < 1.0

    def test_length_penalty_zero_outside_buffer(self):
        env = _make_env()
        env._discount_factor = 1.0
        env._max_tokens = 1000
        env._buffer_tokens = 100
        # Token count outside penalty zone (800 <= 900)
        shaped = env._apply_reward_shaping(1.0, 800)
        assert shaped == pytest.approx(1.0)

    def test_score_thresholding_via_rubric(self):
        env = _make_env()
        # Mimic custom_threshold=True: scores 1-5 → reward = 1/7
        rubric = MathProofRubric(
            grader_model="test", api_key="x", custom_threshold=True
        )
        env._rubric = rubric
        for score in [1, 2, 3, 4, 5]:
            assert rubric.normalize_reward(score) == pytest.approx(1 / 7)


# QEDMathEnvironment: reasoning stripping


class TestReasoningStripping:
    def test_strip_think_tag(self):
        env = _make_env()
        env._reasoning_delimiters = ["</think>"]
        result = env._strip_reasoning("<think>step by step</think>Final answer.")
        assert result == "Final answer."

    def test_no_delimiter_configured_passthrough(self):
        env = _make_env()
        env._reasoning_delimiters = None
        result = env._strip_reasoning("Some text.")
        assert result == "Some text."

    def test_delimiter_absent_returns_empty(self):
        env = _make_env()
        env._reasoning_delimiters = ["</think>"]
        result = env._strip_reasoning("No think tag here.")
        assert result == ""

    @pytest.mark.asyncio
    async def test_grading_falls_back_to_full_text_when_strip_empty(self):
        """When stripping produces empty output, the full proof text is graded."""
        env = _make_env_with_problem(BOOTSTRAP_PROBLEM)
        env.reset()
        env._reasoning_delimiters = ["</think>"]

        graded_input: list[str] = []

        async def mock_grade(proof, *args, **kwargs):
            graded_input.append(proof)
            return GradingResult(score=7, feedback="ok", reward=1.0)

        with patch.object(env._rubric, "grade", new=mock_grade):
            await env._grade_submission("proof without think tag")

        # Stripping produced "" → falls back to original text
        assert graded_input[0] == "proof without think tag"


# QEDMathEnvironment: submit_proof_payload (mocked grader)


class TestSubmitProofPayload:
    @pytest.mark.asyncio
    async def test_no_active_problem(self):
        env = _make_env()
        payload = await env.submit_proof_payload("Some proof.")
        assert payload["score"] == 0
        assert payload["done"] is True
        assert "error" in payload["metadata"]

    @pytest.mark.asyncio
    async def test_successful_submission(self):
        env = _make_env_with_problem(BOOTSTRAP_PROBLEM)
        env.reset()

        mock_result = GradingResult(score=7, feedback="Correct.", reward=1.0)
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            payload = await env.submit_proof_payload("Let a=2m, b=2n, a+b=2(m+n).")

        assert payload["score"] == 7
        assert payload["reward"] == pytest.approx(1.0)
        assert payload["done"] is True
        assert payload["is_correct"] is True

    @pytest.mark.asyncio
    async def test_empty_proof_returns_zero(self):
        env = _make_env_with_problem(BOOTSTRAP_PROBLEM)
        env.reset()
        payload = await env.submit_proof_payload("   ")
        assert payload["score"] == 0
        assert payload["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_attempt_number_increments(self):
        env = _make_env_with_problem(MULTI_STEP_PROBLEM)
        env.reset()

        mock_result = GradingResult(score=3, feedback="Partial.", reward=3 / 7)
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            p1 = await env.submit_proof_payload("attempt 1")
            p2 = await env.submit_proof_payload("attempt 2")

        assert p1["attempt_number"] == 1
        assert p2["attempt_number"] == 2

    @pytest.mark.asyncio
    async def test_verifier_metrics_in_metadata(self):
        env = _make_env_with_problem(BOOTSTRAP_PROBLEM)
        env.reset()

        metrics = {
            "verifier/rollouts/success": 1,
            "verifier/rollouts/failure": 0,
            "verifier/failures/timeout": 0,
            "verifier/failures/rate_limit": 0,
            "verifier/failures/no_input": 0,
            "verifier/failures/no_score_tag": 0,
            "verifier/failures/all_attempts_failed": 0,
            "verifier/failures/num_retries": 0,
            "verifier/runtime/latency_per_request": 0.01,
            "verifier/runtime/input_tokens": 50,
            "verifier/runtime/output_tokens": 20,
        }
        mock_result = GradingResult(
            score=6, feedback="Good.", reward=6 / 7, metrics=metrics
        )
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            payload = await env.submit_proof_payload("Good proof here.")

        vm = payload["metadata"]["verifier_metrics"]
        assert vm["verifier/rollouts/success"] == 1
        assert "verifier/runtime/latency_per_request" in vm
        assert "verifier/runtime/input_tokens" in vm
        assert "verifier/runtime/output_tokens" in vm


# Multi-step problems


class TestMultiStepProblems:
    @pytest.mark.asyncio
    async def test_first_attempt_not_done(self):
        env = _make_env_with_problem(MULTI_STEP_PROBLEM)
        env.reset()

        # Score below success threshold → not done yet
        mock_result = GradingResult(score=3, feedback="Keep trying.", reward=3 / 7)
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            payload = await env.submit_proof_payload("First attempt.")

        assert payload["done"] is False
        assert payload["attempts_remaining"] > 0

    @pytest.mark.asyncio
    async def test_exhausting_attempts_sets_done(self):
        env = _make_env_with_problem(MULTI_STEP_PROBLEM)
        env.reset()

        mock_result = GradingResult(score=1, feedback="Not enough.", reward=1 / 7)
        # 3 attempts configured
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            await env.submit_proof_payload("attempt 1")
            await env.submit_proof_payload("attempt 2")
            payload = await env.submit_proof_payload("attempt 3")

        assert payload["done"] is True
        assert payload["attempts_remaining"] == 0

    @pytest.mark.asyncio
    async def test_correct_submission_ends_episode(self):
        env = _make_env_with_problem(MULTI_STEP_PROBLEM)
        env.reset()

        # Score == 7 (success_score_threshold default) → done immediately
        mock_result = GradingResult(score=7, feedback="Correct!", reward=1.0)
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            payload = await env.submit_proof_payload("Full proof.")

        assert payload["done"] is True
        assert payload["is_correct"] is True


# Original problem field (QED-Nano RC stream compatibility)


class TestOriginalProblemField:
    @pytest.mark.asyncio
    async def test_original_problem_used_for_grading(self):
        """When original_problem is set it should be passed to the rubric."""
        raw = dict(BOOTSTRAP_PROBLEM)
        raw["original_problem"] = "The ORIGINAL phrasing of the problem."
        env = _make_env_with_problem(raw)
        env.reset()

        graded_problems: list[str] = []

        async def mock_grade(proof, problem, *args, **kwargs):
            graded_problems.append(problem)
            return GradingResult(score=5, feedback="ok", reward=5 / 7)

        with patch.object(env._rubric, "grade", new=mock_grade):
            await env.submit_proof_payload("Some proof.")

        assert graded_problems[0] == "The ORIGINAL phrasing of the problem."

    def test_normalize_problem_preserves_original_problem(self):
        raw = dict(BOOTSTRAP_PROBLEM)
        raw["original_problem"] = "Original phrasing."
        normalized = _normalize_problem(raw, 0, "test")
        assert normalized["original_problem"] == "Original phrasing."

    def test_normalize_problem_original_problem_none_when_absent(self):
        normalized = _normalize_problem(BOOTSTRAP_PROBLEM, 0, "test")
        assert normalized["original_problem"] is None


# Models


class TestModels:
    def test_problem_observation_defaults(self):
        obs = ProblemObservation()
        assert obs.problem == ""
        assert obs.problem_type == "proof"
        assert obs.max_attempts == 1

    def test_proof_submission_observation_defaults(self):
        obs = ProofSubmissionObservation()
        assert obs.score == 0
        assert obs.reward == 0.0
        assert obs.done is True
        assert obs.is_correct is False

    def test_problem_observation_round_trips(self):
        obs = ProblemObservation(
            problem="P",
            reference_solution="R",
            problem_id="x",
            problem_type="answer",
            max_attempts=2,
        )
        data = obs.model_dump()
        restored = ProblemObservation(**data)
        assert restored.problem == "P"
        assert restored.problem_type == "answer"
        assert restored.max_attempts == 2

    def test_proof_submission_observation_round_trips(self):
        obs = ProofSubmissionObservation(
            proof="proof text",
            score=5,
            feedback="nice",
            reward=5 / 7,
            done=False,
            attempts_remaining=2,
            is_correct=False,
        )
        data = obs.model_dump()
        restored = ProofSubmissionObservation(**data)
        assert restored.score == 5
        assert restored.attempts_remaining == 2


# MCP tool registration


class TestMCPToolRegistration:
    def test_tools_registered_on_init(self):
        env = _make_env()
        tools = get_server_tools(env.mcp_server)
        assert "get_problem" in tools
        assert "submit_proof" in tools
        assert "get_grading_guidelines" in tools

    def test_get_problem_tool_callable_after_reset(self):
        env = _make_env()
        env.reset()
        tools = get_server_tools(env.mcp_server)
        assert "get_problem" in tools

    def test_get_grading_guidelines_tool_callable_after_reset(self):
        env = _make_env()
        env.reset()
        payload = env.get_grading_guidelines_payload()
        assert "grading_guidelines" in payload


# MathVerifierService


class TestVerifyAnswerWorker:
    """Tests for the _verify_answer_worker function (process worker entry point)."""

    def test_correct_answer(self):
        request = VerifyRequest(
            request_id="req-1",
            prediction=r"\boxed{4}",
            gold="4",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        assert response.status == "correct"
        assert response.request_id == "req-1"
        assert response.elapsed_ms >= 0

    def test_wrong_answer(self):
        request = VerifyRequest(
            request_id="req-2",
            prediction=r"\boxed{5}",
            gold="4",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        assert response.status == "wrong"

    def test_no_boxed_answer(self):
        request = VerifyRequest(
            request_id="req-3",
            prediction="The answer is 4",
            gold="4",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        assert response.status == "no_answer"

    def test_empty_boxed(self):
        request = VerifyRequest(
            request_id="req-4",
            prediction=r"\boxed{}",
            gold="4",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        assert response.status == "unparsable"

    def test_unparsable_prediction(self):
        request = VerifyRequest(
            request_id="req-5",
            prediction=r"\boxed{@#$%^&*()}",
            gold="4",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        # Should be unparsable or internal_error depending on what math_verify does
        assert response.status in {"unparsable", "internal_error"}

    def test_unparsable_gold(self):
        request = VerifyRequest(
            request_id="req-6",
            prediction=r"\boxed{4}",
            gold="@#$%^&*()",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        assert response.status == "unparsable"

    def test_prediction_too_long(self):
        request = VerifyRequest(
            request_id="req-7",
            prediction=r"\boxed{" + "x" * 2000 + "}",
            gold="4",
            strict=True,
            timeout_seconds=1,
            max_prediction_length=1000,
        )
        response = _verify_answer_worker(request)
        assert response.status == "unparsable"

    def test_nested_braces(self):
        """Test that nested braces in \\boxed{...} are handled correctly."""
        request = VerifyRequest(
            request_id="req-8",
            prediction=r"\boxed{\frac{1}{2}}",
            gold=r"\frac{1}{2}",
            strict=True,
            timeout_seconds=1,
        )
        response = _verify_answer_worker(request)
        # Should parse and verify correctly
        assert response.status in {"correct", "wrong"}


class TestMathVerifierService:
    """Tests for the MathVerifierService async interface."""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        service = MathVerifierService(
            max_workers=2,
            queue_size=50,
            request_timeout_seconds=3.0,
        )
        assert service.max_workers == 2
        assert service.queue_size == 50
        assert service.request_timeout_seconds == 3.0

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        service = MathVerifierService(max_workers=1)
        # Start should be idempotent
        await service.start()
        assert service._executor is not None
        await service.start()  # Should not fail if already started
        await service.stop()
        assert service._executor is None

    @pytest.mark.asyncio
    async def test_verify_answer_without_start(self):
        """verify_answer should auto-start the process pool when needed."""
        service = MathVerifierService(max_workers=1)
        response = await service.verify_answer(
            prediction=r"\boxed{4}",
            gold="4",
            strict=True,
        )
        assert response.status == "correct"
        assert service._executor is not None
        await service.stop()

    @pytest.mark.asyncio
    async def test_verify_answer_correct(self):
        service = MathVerifierService(max_workers=1)
        await service.start()
        try:
            response = await service.verify_answer(
                prediction=r"\boxed{4}",
                gold="4",
                strict=True,
            )
            assert response.status == "correct"
            assert response.request_id.startswith("req-")
            assert response.elapsed_ms >= 0
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_verify_answer_wrong(self):
        service = MathVerifierService(max_workers=1)
        await service.start()
        try:
            response = await service.verify_answer(
                prediction=r"\boxed{5}",
                gold="4",
                strict=True,
            )
            assert response.status == "wrong"
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_verify_answer_no_answer(self):
        service = MathVerifierService(max_workers=1)
        await service.start()
        try:
            response = await service.verify_answer(
                prediction="The answer is 4",
                gold="4",
                strict=True,
            )
            assert response.status == "no_answer"
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_verify_answer_timeout(self):
        """Test that client-side timeout is triggered."""
        service = MathVerifierService(
            max_workers=1,
            request_timeout_seconds=0.001,  # Very short timeout to force timeout
        )
        await service.start()
        try:
            response = await service.verify_answer(
                prediction=r"\boxed{4}",
                gold="4",
                strict=True,
            )
            # Should either timeout or succeed quickly; mostly just testing the path
            assert response.request_id.startswith("req-")
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test multiple concurrent verification requests."""
        import asyncio

        service = MathVerifierService(max_workers=2)
        await service.start()
        try:
            # Fire multiple requests concurrently
            tasks = [
                service.verify_answer(r"\boxed{1}", "1"),
                service.verify_answer(r"\boxed{2}", "2"),
                service.verify_answer(r"\boxed{3}", "3"),
                service.verify_answer(r"\boxed{4}", "4"),
            ]
            responses = await asyncio.gather(*tasks)
            assert len(responses) == 4
            assert all(r.status == "correct" for r in responses)
            assert len(set(r.request_id for r in responses)) == 4  # All unique IDs
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_default_strict_mode(self):
        """Test that default strict mode is applied."""
        service = MathVerifierService(max_workers=1, strict=True)
        assert service.strict is True
        await service.start()
        try:
            response = await service.verify_answer(
                prediction=r"\boxed{4.0}",
                gold="4",
                strict=True,  # Explicitly strict
            )
            # Should handle the request
            assert response.request_id.startswith("req-")
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_queue_backpressure_rejects_when_saturated(self):
        import asyncio

        service = MathVerifierService(max_workers=1, queue_size=1)
        await service.start()

        gate_started = asyncio.Event()
        gate_release = asyncio.Event()

        async def slow_once(request):
            gate_started.set()
            await gate_release.wait()
            return _verify_answer_worker(request)

        try:
            with patch.object(service, "_run_request_once", side_effect=slow_once):
                task1 = asyncio.create_task(service.verify_answer(r"\boxed{1}", "1"))
                await gate_started.wait()

                response2 = await service.verify_answer(r"\boxed{2}", "2")
                assert response2.status == "internal_error"
                assert response2.error_type == "QueueFull"

                gate_release.set()
                response1 = await task1
                assert response1.status in {"correct", "wrong"}
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_retry_policy_retries_transient_timeout(self):
        service = MathVerifierService(max_workers=1, max_retries=1)
        await service.start()
        call_count = 0

        async def flaky_once(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return VerifyResponse(
                    request_id=request.request_id,
                    status="timeout",
                    elapsed_ms=1.0,
                    retry_count=0,
                    worker_id=None,
                    worker_restarted=False,
                    error_type="ClientTimeout",
                    error_message="simulated timeout",
                )
            return _verify_answer_worker(request)

        try:
            with patch.object(service, "_run_request_once", side_effect=flaky_once):
                response = await service.verify_answer(r"\boxed{4}", "4")
            assert response.status == "correct"
            assert response.retry_count == 1
            assert call_count == 2
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_worker_crash_restarts_pool_and_redispatches(self):
        service = MathVerifierService(max_workers=1, max_retries=1)
        await service.start()
        call_count = 0
        restart_calls = {"count": 0}
        original_restart = service._restart_pool

        async def crashing_once(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return VerifyResponse(
                    request_id=request.request_id,
                    status="internal_error",
                    elapsed_ms=1.0,
                    error_type="BrokenProcessPool",
                    error_message="broken process pool simulated",
                )
            return _verify_answer_worker(request)

        async def restart_wrapper():
            restart_calls["count"] += 1
            await original_restart()

        try:
            with patch.object(service, "_run_request_once", side_effect=crashing_once):
                with patch.object(
                    service, "_restart_pool", side_effect=restart_wrapper
                ):
                    response = await service.verify_answer(r"\boxed{4}", "4")

            assert response.status == "correct"
            assert response.retry_count == 1
            assert response.worker_restarted is True
            assert restart_calls["count"] == 1

            health = await service.health_probe()
            assert health["restart_count"] >= 1
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_no_retry_for_unparsable_outputs(self):
        service = MathVerifierService(max_workers=1, max_retries=3)
        await service.start()
        call_count = 0

        async def unparsable_once(request):
            nonlocal call_count
            call_count += 1
            return VerifyResponse(
                request_id=request.request_id,
                status="unparsable",
                elapsed_ms=1.0,
                error_type=None,
                error_message=None,
            )

        try:
            with patch.object(
                service, "_run_request_once", side_effect=unparsable_once
            ):
                response = await service.verify_answer("not boxed", "4")

            assert response.status == "unparsable"
            assert response.retry_count == 0
            assert call_count == 1
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_health_probe_reports_runtime_state(self):
        service = MathVerifierService(max_workers=1, queue_size=3)
        before_start = await service.health_probe()
        assert before_start["status"] == "stopped"
        assert before_start["executor_running"] is False

        await service.start()
        try:
            after_start = await service.health_probe()
            assert after_start["status"] == "healthy"
            assert after_start["executor_running"] is True
            assert after_start["queue_size"] == 3
            assert after_start["max_workers"] == 1
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_metrics_snapshot_tracks_rollout_counters(self):
        service = MathVerifierService(max_workers=1, queue_size=4)
        await service.start()
        try:
            await service.verify_answer(r"\boxed{4}", "4")
            await service.verify_answer(r"\boxed{5}", "4")
            await service.verify_answer("no boxed answer", "4")

            metrics = await service.metrics_snapshot()
            assert metrics["verifier/requests/count"] >= 3
            assert metrics["verifier/requests/latency_ms"] >= 0
            assert metrics["verifier/requests/timeout_count"] >= 0
            assert metrics["verifier/requests/error_count"] >= 0
            assert metrics["verifier/workers/heartbeat_lag_ms"] >= 0
        finally:
            await service.stop()


class TestGoldCache:
    @pytest.mark.asyncio
    async def test_answer_mode_grading_uses_cached_gold(self):
        env = _make_env_with_problem(ANSWER_PROBLEM)
        env.reset()

        problem_id = env._current_problem["problem_id"]
        env._gold_answer_cache[env._gold_cache_key(problem_id)] = r"\boxed{42}"

        captured_expected: dict[str, str] = {}

        async def fake_grade_answer_submission(
            submission: str,
            expected_answer: str,
            problem_id: str = "",
        ):
            captured_expected["value"] = expected_answer
            return GradingResult(score=7, feedback="ok", reward=1.0)

        with patch.object(
            env,
            "_grade_answer_submission",
            new=fake_grade_answer_submission,
        ):
            await env._grade_submission(r"\boxed{42}")

        assert captured_expected["value"] == r"\boxed{42}"
        await env._verifier_service.stop()

    def test_gold_cache_invalidation_on_config_change(self):
        env = _make_env_with_problem(ANSWER_PROBLEM)
        old_signature = env._gold_cache_signature

        env._config.verifier_numeric_precision += 1
        env._refresh_gold_cache_if_needed()

        assert env._gold_cache_signature != old_signature
        env.reset()
        problem_id = env._current_problem["problem_id"]
        assert env._gold_cache_key(problem_id) in env._gold_answer_cache

    @pytest.mark.asyncio
    async def test_cache_hit_rate_surfaces_in_verifier_metrics(self):
        env = _make_env_with_problem(ANSWER_PROBLEM)
        env.reset()

        await env._grade_submission(r"\boxed{4}")
        payload = await env.submit_proof_payload(r"\boxed{4}")
        verifier_metrics = payload["metadata"]["verifier_metrics"]

        assert "verifier/cache/hit_rate" in verifier_metrics
        assert verifier_metrics["verifier/cache/hit_rate"] >= 0.0
        assert verifier_metrics["verifier/cache/hit_rate"] <= 1.0

        await env._verifier_service.stop()


# Config defaults aligned with QED-Nano


class TestConfigDefaults:
    def test_default_prompt_is_v1(self):
        assert QEDMathConfig().prompt_name == "v1"

    def test_default_reasoning_delimiters(self):
        assert QEDMathConfig().reasoning_delimiters == ["</think>"]

    def test_default_answer_reward_preset(self):
        assert QEDMathConfig().answer_reward_preset == "pure_success"

    def test_grader_sampling_defaults(self):
        cfg = QEDMathConfig()
        assert cfg.grader_temperature == 1.0
        assert cfg.grader_max_output_tokens is None

    def test_env_strips_think_by_default(self):
        env = _make_env()
        assert env._reasoning_delimiters == ["</think>"]
        assert env._strip_reasoning("<think>reasoning</think>Answer.") == "Answer."

    def test_v1_prompt_loads_from_disk(self):
        prompt = load_evaluator_prompt("v1")
        assert "{problem}" in prompt
        assert "{solution}" in prompt
        assert "{marking_scheme}" in prompt
        assert "grader" in prompt.lower()

    def test_v0_prompt_loads_from_disk(self):
        prompt = load_evaluator_prompt("v0")
        assert "{human_solution}" in prompt


# Problem-type auto-detection (QED-Nano `if "schema" in problem`)


class TestProblemTypeDetection:
    def test_rubric_list_present_is_proof(self):
        row = {"problem": "P", "rubrics": [{"title": "t", "points": 1, "desc": "d"}]}
        assert _normalize_problem(row, 0, "test")["problem_type"] == "proof"

    def test_string_guidelines_is_proof(self):
        row = {"problem": "P", "grading_guidelines": "Award credit for rigor."}
        assert _normalize_problem(row, 0, "test")["problem_type"] == "proof"

    def test_no_schema_is_answer(self):
        row = {"problem": "P", "reference_solution": "4"}
        norm = _normalize_problem(row, 0, "test")
        assert norm["problem_type"] == "answer"
        assert norm["evaluation_mode"] == "answer"
        # Answer-mode gold gets boxed-wrapped.
        assert norm["reference_solution"] == r"\boxed{4}"

    def test_empty_guidelines_is_answer(self):
        row = {"problem": "P", "grading_guidelines": "", "reference_solution": "4"}
        assert _normalize_problem(row, 0, "test")["problem_type"] == "answer"

    def test_explicit_proof_type_overrides_missing_schema(self):
        row = {"problem": "P", "problem_type": "proof", "reference_solution": "4"}
        assert _normalize_problem(row, 0, "test")["problem_type"] == "proof"


# Answer-mode reward presets


class TestAnswerRewardPresets:
    def test_pure_success_table(self):
        assert answer_reward_for_status("correct", "pure_success") == 1.0
        assert answer_reward_for_status("wrong", "pure_success") == 0.0
        assert answer_reward_for_status("no_answer", "pure_success") == 0.0

    def test_base_table_penalties(self):
        assert answer_reward_for_status("correct", "base") == 1.0
        assert answer_reward_for_status("wrong", "base") == -0.5
        assert answer_reward_for_status("no_answer", "base") == -1.0
        assert answer_reward_for_status("unparsable", "base") == -1.0
        # Infra failures stay neutral.
        assert answer_reward_for_status("timeout", "base") == 0.0
        assert answer_reward_for_status("internal_error", "base") == 0.0

    def test_unknown_preset_falls_back_to_pure_success(self):
        assert answer_reward_for_status("wrong", "does_not_exist") == 0.0

    @pytest.mark.asyncio
    async def test_base_preset_negative_reward_for_wrong(self):
        env = QEDMathEnvironment(config=QEDMathConfig(answer_reward_preset="base"))
        normalized = _normalize_problem(ANSWER_PROBLEM, 0, "test")
        env._problems = [normalized]
        env._build_gold_answer_cache()
        try:
            result = await env._grade_answer_submission(r"\boxed{5}", r"\boxed{4}")
            assert result.score == 0
            assert result.reward == pytest.approx(-0.5)
        finally:
            await env._verifier_service.stop()


# Success threshold aligned to upstream (score == 7)


class TestSuccessThreshold:
    def test_default_threshold_is_seven(self):
        norm = _normalize_problem(BOOTSTRAP_PROBLEM, 0, "test")
        assert norm["success_score_threshold"] == 7

    @pytest.mark.asyncio
    async def test_score_six_is_not_correct(self):
        env = _make_env_with_problem(BOOTSTRAP_PROBLEM)
        env.reset()
        mock_result = GradingResult(score=6, feedback="Almost.", reward=6 / 7)
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            payload = await env.submit_proof_payload("proof")
        assert payload["is_correct"] is False

    @pytest.mark.asyncio
    async def test_score_seven_is_correct(self):
        env = _make_env_with_problem(BOOTSTRAP_PROBLEM)
        env.reset()
        mock_result = GradingResult(score=7, feedback="Perfect.", reward=1.0)
        with patch.object(
            env._rubric, "grade", new=AsyncMock(return_value=mock_result)
        ):
            payload = await env.submit_proof_payload("proof")
        assert payload["is_correct"] is True


# Grader sampling params + real token usage


class TestGraderSamplingParams:
    def test_rubric_defaults(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        assert rubric.temperature == 1.0
        assert rubric.max_output_tokens is None

    @pytest.mark.asyncio
    async def test_call_llm_forwards_temperature(self):
        rubric = MathProofRubric(grader_model="test", api_key="x", temperature=0.3)
        await rubric._call_llm("prompt text")
        kwargs = rubric._client.responses.create.call_args.kwargs
        assert kwargs["temperature"] == 0.3
        assert kwargs["model"] == "test"

    @pytest.mark.asyncio
    async def test_call_llm_forwards_max_output_tokens(self):
        rubric = MathProofRubric(
            grader_model="test", api_key="x", max_output_tokens=256
        )
        await rubric._call_llm("prompt")
        kwargs = rubric._client.responses.create.call_args.kwargs
        assert kwargs["max_output_tokens"] == 256

    @pytest.mark.asyncio
    async def test_call_llm_omits_max_output_tokens_when_none(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        await rubric._call_llm("prompt")
        kwargs = rubric._client.responses.create.call_args.kwargs
        assert "max_output_tokens" not in kwargs

    def test_capture_usage_responses_api(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        usage = MagicMock()
        usage.input_tokens = 11
        usage.output_tokens = 22
        response = MagicMock()
        response.usage = usage
        rubric._capture_usage(response, responses_api=True)
        assert rubric._last_input_tokens == 11
        assert rubric._last_output_tokens == 22

    def test_capture_usage_chat_completions(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")
        usage = MagicMock()
        usage.prompt_tokens = 7
        usage.completion_tokens = 9
        response = MagicMock()
        response.usage = usage
        rubric._capture_usage(response, responses_api=False)
        assert rubric._last_input_tokens == 7
        assert rubric._last_output_tokens == 9

    @pytest.mark.asyncio
    async def test_grade_uses_real_token_usage_when_available(self):
        rubric = MathProofRubric(grader_model="test", api_key="x")

        async def fake_call(prompt):
            rubric._last_input_tokens = 111
            rubric._last_output_tokens = 222
            return "<score>7</score>"

        with patch.object(rubric, "_call_llm", new=fake_call):
            result = await rubric.grade("proof", "P", "R", "G")
        assert result.metrics["verifier/runtime/input_tokens"] == 111
        assert result.metrics["verifier/runtime/output_tokens"] == 222


class TestQEDMathEnvClient:
    def test_default_message_timeout_matches_submit_proof_timeout(self):
        from qed_math_env.client import QEDMathEnv

        client = QEDMathEnv(base_url="http://localhost:8000")

        assert client._message_timeout == pytest.approx(600.0)

    def test_reset_uses_dispatcher_not_async_only_method(self):
        from qed_math_env.client import QEDMathEnv

        assert not inspect.iscoroutinefunction(QEDMathEnv.reset)

    @pytest.mark.asyncio
    async def test_reset_normalizes_problem_observation(self):
        from qed_math_env.client import QEDMathEnv

        client = QEDMathEnv(base_url="http://localhost:8000")
        client._send_and_receive = AsyncMock(
            return_value={
                "data": {
                    "observation": {
                        "problem": "What is 2+2?",
                        "reference_solution": "4",
                        "grading_guidelines": "",
                        "problem_id": "answer_001",
                        "dataset_source": "test",
                        "problem_type": "answer",
                        "max_attempts": 1,
                    },
                    "reward": None,
                    "done": False,
                    "metadata": {"source": "unit"},
                }
            }
        )

        result = await client.reset(problem_id="answer_001")

        assert isinstance(result.observation, ProblemObservation)
        assert result.observation.problem_id == "answer_001"
        assert result.metadata == {"source": "unit"}
        client._send_and_receive.assert_awaited_once_with(
            {"type": "reset", "data": {"problem_id": "answer_001"}}
        )

    def test_get_state_sync_uses_current_session(self, monkeypatch):
        from qed_math_env.client import QEDMathEnv

        client = QEDMathEnv(base_url="http://localhost:8000")
        expected_state = State(episode_id="episode-1", step_count=3)
        state_mock = MagicMock(return_value=expected_state)
        sync_mock = MagicMock(side_effect=AssertionError("opened a new session"))
        monkeypatch.setattr(client, "state", state_mock)
        monkeypatch.setattr(client, "sync", sync_mock)

        assert client.get_state_sync() is expected_state
        state_mock.assert_called_once_with()
        sync_mock.assert_not_called()


# Integration tests (require running server)


@pytest.mark.integration
class TestQEDMathServerIntegration:
    """Full HTTP/WebSocket stack smoke test.

    Requires a running QED Math server at QED_MATH_URL (default: http://localhost:8000).
    Start it with:

        docker run -p 8000:8000 \\
          -e JUDGE_API_BASE_URL=... \\
          -e JUDGE_API_KEY=... \\
          qed-math-env:latest
    """

    BASE_URL = "http://localhost:8000"

    @pytest.fixture
    def env(self):
        from qed_math_env.client import QEDMathEnv

        with QEDMathEnv(base_url=self.BASE_URL).sync() as e:
            yield e

    def test_reset_returns_problem(self, env):
        obs = env.reset()
        assert obs is not None

    def test_list_tools_returns_three_tools(self, env):
        tools = env.list_tools()
        names = {t.name for t in tools}
        assert {"get_problem", "submit_proof", "get_grading_guidelines"}.issubset(names)

    def test_get_problem_tool(self, env):
        env.reset()
        result = env.call_tool("get_problem")
        assert result is not None

    def test_get_grading_guidelines_tool(self, env):
        env.reset()
        result = env.call_tool("get_grading_guidelines")
        assert result is not None

    def test_submit_proof_tool_returns_observation(self, env):
        env.reset()
        result = env.call_tool(
            "submit_proof", proof="Let a=2m and b=2n, so a+b=2(m+n)."
        )
        assert result is not None
