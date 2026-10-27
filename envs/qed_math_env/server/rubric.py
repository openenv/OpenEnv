# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rubric implementation for the QED Math Environment.

Grades mathematical proofs on a 0-7 scale using an LLM judge and
normalizes the score to a [0, 1] reward signal.

The grader is prompted to produce a ``<score>N</score>`` tag; that tag is
parsed and the integer score is normalized to a reward. Optional
``custom_threshold`` collapses partial-credit scores (1-5) to 1, which
mirrors the thresholding in QED-Nano rollouts.

References:
    QED-Nano: training/pipelinerl/domains/math/verifier_api.py (verify_proof)
    QED-Nano: training/pipelinerl/domains/math/rollouts.py (apply_score_threshold)
    src/openenv/core/rubrics/base.py - Rubric base class
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Union

import openai
from openenv.core.rubrics.base import Rubric


def parse_schema(schema: Any) -> str:
    """Normalize a schema payload into the Markdown string used as {marking_scheme}.

    Ported from QED-Nano: training/pipelinerl/domains/math/verifier_api.py.

    Accepts either a plain string (returned as-is) or a list of dicts with
    the keys ``"title"``, ``"points"``, and ``"desc"`` / ``"description"``
    that is converted to a Markdown representation.

    Args:
        schema: A string marking scheme or a list of rubric-criterion dicts.

    Returns:
        Markdown-formatted marking scheme string.

    Raises:
        TypeError: If *schema* is not a string or list.
        ValueError: If a list entry is not a dict or is missing required keys.
    """
    if isinstance(schema, str):
        return schema
    if isinstance(schema, dict):
        return str(schema)
    if not isinstance(schema, list):
        raise TypeError(
            f"parse_schema expects a string or list of dicts, got {type(schema).__name__}"
        )

    sections: list[str] = []
    for idx, entry in enumerate(schema):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Schema entry at index {idx} must be a dict, got {type(entry).__name__}"
            )
        title = entry.get("title")
        points = entry.get("points")
        description = entry.get("desc") or entry.get("description")
        if title is None or points is None or description is None:
            raise ValueError(
                f"Schema entry at index {idx} is missing 'title', 'points', or 'desc'/'description'"
            )
        sections.append(
            f"# {title} ({points} points)\nDescription: {description}".strip()
        )

    return "\n\n".join(sections)


# Maximum score on the 0-7 rubric scale.
MAX_SCORE = 7

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = [15, 30, 60]

# Format variables expected by evaluator prompt templates that come from
# QED-Nano's prompts/evaluator_prompts/*.md files.
_TEMPLATE_VARS = ("{problem}", "{human_solution}", "{marking_scheme}", "{solution}")


@dataclass
class GradingResult:
    """Structured output from a single MathProofRubric grading call."""

    score: int  # Raw integer score in [0, MAX_SCORE]
    feedback: str  # Full grader response text (may contain reasoning)
    reward: float  # Normalized reward in [0, 1]
    metrics: dict[str, float | int] = field(default_factory=dict)
    # Verifier metrics following QED-Nano conventions:
    #   verifier/rollouts/success, verifier/rollouts/failure,
    #   verifier/failures/timeout, verifier/failures/rate_limit,
    #   verifier/failures/no_score_tag, verifier/failures/num_retries,
    #   verifier/runtime/latency_per_request,
    #   verifier/runtime/input_tokens, verifier/runtime/output_tokens


def apply_score_threshold(score: float) -> float:
    """Apply reward thresholding based on verification score.

    Ported from QED-Nano: training/pipelinerl/domains/math/rollouts.py.

    Proofs with partial credit (score 1–5) are collapsed to 1 so that the
    training signal distinguishes only between "no progress" (0), "any
    progress" (1), and "near-complete / correct" (6–7).

    Args:
        score: Raw verification score (0–7).

    Returns:
        Thresholded score.
    """
    if score < 1.0:
        return score
    if score < 6.0:
        return 1.0
    return score


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """Compute an overlong penalty for sequences approaching *max_length*.

    When ``buffer_tokens`` is 0 the penalty is always 0 (disabled).  Otherwise
    sequences that exceed ``max_length - buffer_tokens`` but stay within
    ``max_length`` receive a linearly increasing negative penalty.

    Args:
        max_length: Maximum allowed token count.
        sequence_length: Actual token count of the generation.
        buffer_tokens: Width of the soft penalty zone before *max_length*.
            0 disables the penalty.

    Returns:
        A non-positive float: 0.0 when outside the penalty zone, negative
        when inside.
    """
    if buffer_tokens <= 0:
        return 0.0
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.0


class MathProofRubric(Rubric):
    """LLM-based rubric for grading mathematical proofs on a 0-7 scale.

    Uses OpenAI-compatible endpoints to evaluate a
    submitted proof against the problem statement, reference solution, and
    grading guidelines.  The grader is expected to emit a
    ``<score>N</score>`` tag; any text outside that tag is treated as
    reasoning / feedback.

    Args:
        grader_model: Model identifier forwarded to the API (e.g.
            ``"gemini-2.0-flash"`` or ``"gpt-4o"``).
        prompt_template: Evaluator prompt template string.  When the
            template contains the variables ``{problem}``,
            ``{human_solution}``, ``{marking_scheme}``, and ``{solution}``,
            they are substituted before the call.  If none of those
            variables are present the rubric constructs a minimal prompt
            automatically.
        custom_threshold: When *True*, apply :func:`apply_score_threshold`
            before normalizing, collapsing partial-credit scores 1-5 to 1.
        api_base_url: Override the OpenAI base URL.  Falls back to the
            ``OPENAI_BASE_URL`` environment variable or the OpenAI default.
        api_key: Override the API key.  Falls back to ``OPENAI_API_KEY``.
        max_retries: Number of LLM call attempts before giving up.
        retry_backoff: Sleep durations (seconds) between attempts.
        timeout_seconds: Per-attempt timeout for the LLM call.
        temperature: Sampling temperature forwarded to the grader (QED-Nano
            uses ``1.0``).
        max_output_tokens: Optional output-token cap forwarded to the grader.
            ``None`` omits it so the provider default applies (avoids
            truncating the trailing ``<score>`` tag on verbose graders).
    """

    def __init__(
        self,
        grader_model: str = "gemini-2.0-flash",
        prompt_template: str = "",
        custom_threshold: bool = False,
        api_base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: list[int] | None = None,
        timeout_seconds: int = 900,
        temperature: float = 1.0,
        max_output_tokens: int | None = None,
    ):
        super().__init__()
        self.grader_model = grader_model
        self.prompt_template = prompt_template
        self.custom_threshold = custom_threshold
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff or list(_DEFAULT_RETRY_BACKOFF)
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # Real token usage captured from the last successful LLM call (None when
        # unavailable, e.g. providers that omit ``usage`` or mocked clients).
        self._last_input_tokens: int | None = None
        self._last_output_tokens: int | None = None

        self._client_kwargs: dict[str, Any] = {}
        if api_base_url is not None:
            self._client_kwargs["base_url"] = api_base_url
        if api_key is not None:
            self._client_kwargs["api_key"] = api_key

    @cached_property
    def _client(self) -> openai.AsyncOpenAI:
        """Grader client, created on first use so the server starts without credentials."""
        return openai.AsyncOpenAI(**self._client_kwargs)

    # Rubric base-class interface

    async def forward(self, action: Any, observation: Any) -> float:
        """Evaluate a proof submission and return the normalized reward.

        Expected types (duck-typed):
            action:      SubmitProof — provides ``.proof``
            observation: ProblemObservation — provides ``.problem``,
                         ``.reference_solution``, ``.grading_guidelines``

        Returns:
            Normalized reward in [0, 1].
        """
        proof = getattr(action, "proof", str(action))
        problem = getattr(observation, "problem", "")
        reference_solution = getattr(observation, "reference_solution", "")
        grading_guidelines = getattr(observation, "grading_guidelines", "")
        result = await self.grade(
            proof, problem, reference_solution, grading_guidelines
        )
        return result.reward

    # Extended grading interface (used directly by the environment)

    async def grade(
        self,
        proof: str,
        problem: str,
        reference_solution: str,
        grading_guidelines: Union[str, list, None] = "",
    ) -> GradingResult:
        """Grade a proof and return a :class:`GradingResult`.

        Builds the evaluator prompt, calls the LLM, parses the
        ``<score>N</score>`` tag, applies optional thresholding, and
        returns the normalized reward.  Retries transient errors up to
        ``max_retries`` times with configurable back-off.

        Also collects verifier metrics (success/failure, failure causes,
        retry count, latency, tokens) mirroring QED-Nano's
        ``_build_rollout_metrics`` pattern.

        Args:
            proof: Agent's proof text.
            problem: Problem statement.
            reference_solution: Ground-truth / reference solution.
            grading_guidelines: Marking scheme / rubric text (optional).
                May be a plain string or a structured schema list (see
                :func:`parse_schema`); ``None`` is treated as empty.

        Returns:
            GradingResult with ``score`` (0-7), ``feedback``,
            ``reward`` (0.0-1.0), and ``metrics`` dict.
        """
        metrics: dict[str, float | int] = {
            "verifier/rollouts/success": 0,
            "verifier/rollouts/failure": 0,
            "verifier/failures/timeout": 0,
            "verifier/failures/rate_limit": 0,
            "verifier/failures/no_input": 0,
            "verifier/failures/no_score_tag": 0,
            "verifier/failures/all_attempts_failed": 0,
            "verifier/failures/num_retries": 0,
            "verifier/runtime/latency_per_request": 0.0,
            "verifier/runtime/input_tokens": 0,
            "verifier/runtime/output_tokens": 0,
        }

        if not proof.strip():
            metrics["verifier/rollouts/failure"] = 1
            metrics["verifier/failures/no_input"] = 1
            return GradingResult(
                score=0,
                feedback="Empty proof submission.",
                reward=0.0,
                metrics=metrics,
            )

        # Normalize structured schema to Markdown before building the prompt.
        guidelines_str: str
        if grading_guidelines is None:
            guidelines_str = ""
        elif isinstance(grading_guidelines, str):
            guidelines_str = grading_guidelines
        else:
            guidelines_str = parse_schema(grading_guidelines)

        prompt = self._build_prompt(proof, problem, reference_solution, guidelines_str)

        attempt_causes: list[str] = []
        t0 = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=self.timeout_seconds,
                )
                score, feedback = self._parse_response(response_text)

                elapsed = time.perf_counter() - t0
                metrics["verifier/runtime/latency_per_request"] = round(elapsed, 4)
                metrics["verifier/failures/num_retries"] = attempt - 1

                # Prefer real token usage from the provider; fall back to a
                # rough heuristic (~4 chars per token) when usage is absent.
                metrics["verifier/runtime/input_tokens"] = (
                    self._last_input_tokens
                    if self._last_input_tokens is not None
                    else max(1, len(prompt) // 4)
                )
                metrics["verifier/runtime/output_tokens"] = (
                    self._last_output_tokens
                    if self._last_output_tokens is not None
                    else max(1, len(response_text) // 4)
                )

                # Check for missing score tag (score defaults to 0).
                if not re.search(r"<score>\d+</score>", response_text):
                    metrics["verifier/failures/no_score_tag"] = 1
                    metrics["verifier/rollouts/failure"] = 1
                else:
                    metrics["verifier/rollouts/success"] = 1

                reward = self.normalize_reward(score)
                return GradingResult(
                    score=score,
                    feedback=feedback,
                    reward=reward,
                    metrics=metrics,
                )

            except openai.RateLimitError:
                attempt_causes.append("rate_limit")
                metrics["verifier/failures/rate_limit"] += 1
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except asyncio.TimeoutError:
                attempt_causes.append("timeout")
                metrics["verifier/failures/timeout"] += 1
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

            except Exception as exc:  # noqa: BLE001
                attempt_causes.append(f"error:{exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))

        # All attempts exhausted.
        elapsed = time.perf_counter() - t0
        metrics["verifier/runtime/latency_per_request"] = round(elapsed, 4)
        metrics["verifier/failures/num_retries"] = self.max_retries
        metrics["verifier/failures/all_attempts_failed"] = 1
        metrics["verifier/rollouts/failure"] = 1

        return GradingResult(
            score=0,
            feedback=(
                f"Grading failed after {self.max_retries} attempt(s): "
                + "; ".join(attempt_causes)
            ),
            reward=0.0,
            metrics=metrics,
        )

    # Helpers

    def normalize_reward(self, score: int) -> float:
        """Normalize a 0-7 score to a [0, 1] reward.

        When ``custom_threshold`` is *True*, :func:`apply_score_threshold`
        is applied first (collapses partial credit 1-5 → 1).

        Args:
            score: Raw integer score (0–7).

        Returns:
            Normalized reward in [0, 1].
        """
        effective = (
            apply_score_threshold(float(score))
            if self.custom_threshold
            else float(score)
        )
        return effective / MAX_SCORE

    def _build_prompt(
        self,
        proof: str,
        problem: str,
        reference_solution: str,
        grading_guidelines: str,
    ) -> str:
        """Format the evaluator prompt template with grading variables.

        Variable mapping (QED-Nano verifier_api.py convention):
            ``{problem}``          → problem statement
            ``{human_solution}``   → reference / ground-truth solution
            ``{marking_scheme}``   → grading rubric / guidelines
            ``{solution}``         → agent's submitted proof

        Falls back to a minimal constructed prompt when the template does
        not contain any of the expected format variables.
        """
        if self.prompt_template and any(
            v in self.prompt_template for v in _TEMPLATE_VARS
        ):
            return self.prompt_template.format(
                problem=problem,
                human_solution=reference_solution,
                marking_scheme=grading_guidelines,
                solution=proof,
            )

        # Fallback: build a minimal prompt from available context.
        parts: list[str] = [
            self.prompt_template
            or (
                "You are a strict math proof grader. "
                "Score the submission from 0 to 7 based on mathematical "
                "correctness, completeness, and logical rigor."
            )
        ]
        if problem:
            parts.append(f"\n\nProblem:\n{problem}")
        if reference_solution:
            parts.append(f"\n\nReference Solution:\n{reference_solution}")
        if grading_guidelines:
            parts.append(f"\n\nGrading Guidelines:\n{grading_guidelines}")
        parts.append(f"\n\nSubmitted Proof:\n{proof}")
        parts.append(
            "\n\nProvide your score using exactly this format: <score>N</score> "
            "where N is an integer from 0 to 7."
        )
        return "".join(parts)

    async def _call_llm(self, prompt: str) -> str:
        """Send prompt to the model and return response text.

        Tries the Responses API first to align with QED-Nano behavior, then
        falls back to Chat Completions for providers that do not support
        Responses. Sampling parameters (``temperature``, ``max_output_tokens``)
        are forwarded and real token usage is captured when the provider
        returns it.
        """
        self._last_input_tokens = None
        self._last_output_tokens = None
        try:
            responses_kwargs: dict[str, Any] = {
                "model": self.grader_model,
                "input": prompt,
                "temperature": self.temperature,
            }
            if self.max_output_tokens is not None:
                responses_kwargs["max_output_tokens"] = self.max_output_tokens
            response = await self._client.responses.create(**responses_kwargs)
            self._capture_usage(response, responses_api=True)
            text = getattr(response, "output_text", None)
            if isinstance(text, str) and text:
                return text
            return self._extract_response_text(response)
        except Exception:  # noqa: BLE001
            # Compatibility fallback for endpoints that only expose
            # chat/completions semantics.
            chat_kwargs: dict[str, Any] = {
                "model": self.grader_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if self.max_output_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_output_tokens
            response = await self._client.chat.completions.create(**chat_kwargs)
            self._capture_usage(response, responses_api=False)
            return response.choices[0].message.content or ""

    def _capture_usage(self, response: Any, responses_api: bool) -> None:
        """Record token usage from a grader response when available.

        Reads the Responses-API (``input_tokens``/``output_tokens``) or
        Chat-Completions (``prompt_tokens``/``completion_tokens``) usage shape.
        Leaves the cached counts as ``None`` when no usage is reported.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        if responses_api:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
        else:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(input_tokens, int):
            self._last_input_tokens = input_tokens
        if isinstance(output_tokens, int):
            self._last_output_tokens = output_tokens

    def _extract_response_text(self, response: Any) -> str:
        """Best-effort extraction of plain text from a Responses API payload."""
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return ""

        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = getattr(part, "type", None)
                if part_type == "output_text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        chunks.append(text)

        return "\n".join(chunks)

    def _parse_response(self, text: str) -> tuple[int, str]:
        """Parse ``<score>N</score>`` from a grader response.

        Args:
            text: Raw LLM response text.

        Returns:
            ``(score, feedback)`` where *score* is clamped to [0, MAX_SCORE]
            and *feedback* is the full response text.
        """
        match = re.search(r"<score>(\d+)</score>", text)
        if match:
            score = max(0, min(MAX_SCORE, int(match.group(1))))
            return score, text
        return 0, text

    def _backoff(self, attempt: int) -> int:
        """Return the sleep duration (seconds) for a given attempt number."""
        idx = min(attempt - 1, len(self.retry_backoff) - 1)
        return self.retry_backoff[idx]
