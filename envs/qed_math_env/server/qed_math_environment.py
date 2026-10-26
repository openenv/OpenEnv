# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
QED Math Environment Implementation.

A math proof environment that presents problems to agents and evaluates
submitted proofs using LLM-based rubric grading (0-7 scale).
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from datasets import load_dataset

import random
from uuid import uuid4
from typing import Any, Optional

import math_verify

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
)
from openenv.core.env_server.types import Action, Observation, State

try:
    from ..models import ProblemObservation, ProofSubmissionObservation
except ImportError:
    from models import ProblemObservation, ProofSubmissionObservation

from .mcp_server import register_mcp_tools
from .rubric import GradingResult, MathProofRubric, length_penalty, parse_schema
from .math_verify_service import (
    EmptyBoxedException,
    MathVerifierService,
    NoAnswerException,
    UnparsableException,
    _parse_math_verify_expression,
)


DEFAULT_EVALUATOR_PROMPT = (
    "You are a strict math proof grader. Score the submission from 0 to 7 based on "
    "mathematical correctness, completeness, and logical rigor."
)

logger = logging.getLogger(__name__)

DatasetSource = str | dict[str, Any] | list[str | dict[str, Any]] | None


def _default_verifier_workers() -> int:
    """Choose a conservative worker default for shared training hosts."""
    cpu_count = os.cpu_count() or 2
    return max(2, min(8, cpu_count // 2 or 1))


def _default_verifier_queue_size() -> int:
    """Default queue size proportional to worker count for burst tolerance."""
    return _default_verifier_workers() * 32


# Answer-mode reward presets keyed on the verifier ``status`` value, mirroring
# QED-Nano's ``RewardTable`` (training/conf/rewards/*.yaml). ``pure_success`` is
# the shipped default (correct -> 1, everything else -> 0); ``base`` adds the
# negative penalties from QED-Nano's ``base`` reward config. ``timeout`` and
# ``internal_error`` are infrastructure failures and stay neutral so transient
# verifier issues never inject a spurious negative training signal. The
# ``finished`` (truncation) dimension of the upstream table is not modelled —
# single-turn ``submit_proof`` has no truncation signal.
ANSWER_REWARD_PRESETS: dict[str, dict[str, float]] = {
    "pure_success": {
        "correct": 1.0,
        "wrong": 0.0,
        "no_answer": 0.0,
        "unparsable": 0.0,
        "timeout": 0.0,
        "internal_error": 0.0,
    },
    "base": {
        "correct": 1.0,
        "wrong": -0.5,
        "no_answer": -1.0,
        "unparsable": -1.0,
        "timeout": 0.0,
        "internal_error": 0.0,
    },
}


def answer_reward_for_status(status: str, preset: str = "pure_success") -> float:
    """Return the answer-mode reward for a verifier *status* under *preset*.

    Args:
        status (`str`):
            Verifier status (``correct``, ``wrong``, ``no_answer``,
            ``unparsable``, ``timeout``, or ``internal_error``).
        preset (`str`, *optional*, defaults to `"pure_success"`):
            Reward-preset name. Unknown presets fall back to ``pure_success``.

    Returns:
        `float`: Reward for the given status (may be negative for ``base``).
    """
    table = ANSWER_REWARD_PRESETS.get(preset, ANSWER_REWARD_PRESETS["pure_success"])
    return table.get(status, 0.0)


def remove_reasoning(
    completion: str,
    reasoning_delimiters: list[str] | None = None,
) -> str:
    """Strip reasoning traces from model output before verification.

    When *reasoning_delimiters* is provided, splits on each delimiter in
    order and keeps only the text after the last occurrence.  This handles
    outputs that include ``<think>...</think>`` or similar chain-of-thought
    wrapping.

    Args:
        completion: Raw model output.
        reasoning_delimiters: Ordered list of delimiter strings to split on.
            ``None`` or empty list disables stripping.

    Returns:
        The final-answer portion of the completion, or an empty string if
        no delimiter was found (indicating the model never produced a
        final-answer section).
    """
    if not reasoning_delimiters:
        return completion
    for delim in reasoning_delimiters:
        if delim in completion:
            completion = completion.split(delim)[-1]
            return completion.strip()
    return ""


@dataclass
class QEDMathConfig:
    """Configuration for QEDMathEnvironment rubric and dataset behavior."""

    dataset_path: DatasetSource = None
    grader_model: str = "gemini-3-pro"
    prompt_name: str = "v1"
    custom_reward_threshold: bool = False
    max_attempts: int = 1
    discount_factor: float = 1.0
    buffer_tokens: int = 0
    max_tokens: int = 0
    reasoning_delimiters: list[str] | None = field(default_factory=lambda: ["</think>"])
    grader_temperature: float = 1.0
    grader_max_output_tokens: int | None = None
    answer_reward_preset: str = "pure_success"
    verifier_workers: int = field(default_factory=_default_verifier_workers)
    verifier_queue_size: int = field(default_factory=_default_verifier_queue_size)
    verifier_request_timeout_seconds: float = 5.0
    verifier_max_retries: int = 1
    verifier_strict: bool = True
    verifier_numeric_precision: int = 5
    verifier_float_rounding: int = 10


def load_evaluator_prompt(prompt_name: str) -> str:
    """Load an evaluator prompt template by name.

    Prompt lookup order:
    1) env local: envs/qed_math_env/prompts/evaluator_prompts/{prompt_name}.md
    2) fallback to default in-code template
    """
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "evaluator_prompts"
        / f"{prompt_name}.md"
    )
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return DEFAULT_EVALUATOR_PROMPT


def _bootstrap_problems() -> list[dict]:
    return [
        {
            "problem": "Prove that the sum of two even integers is even.",
            "reference_solution": "Let a=2m and b=2n. Then a+b=2(m+n), so it is even.",
            "grading_guidelines": "Award full credit for a correct parity argument.",
            "problem_id": "bootstrap_000001",
            "dataset_source": "bootstrap",
            "problem_type": "proof",
            "max_attempts": 1,
        }
    ]


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _canonical_problem_type(raw_problem: dict[str, Any]) -> str:
    explicit_type = _first_present_value(
        raw_problem,
        ("problem_type", "type", "problem_kind", "mode"),
        None,
    )
    if isinstance(explicit_type, str):
        normalized = explicit_type.strip().lower()
        if normalized in {"proof", "answer", "multi_step"}:
            return normalized

    if bool(raw_problem.get("multi_step")):
        return "multi_step"

    evaluation_mode = _first_present_value(raw_problem, ("evaluation_mode",), None)
    if isinstance(evaluation_mode, str) and evaluation_mode.strip().lower() == "answer":
        return "answer"

    # Auto-detect mirroring QED-Nano's ``if "schema" in problem`` routing: the
    # presence of a grading rubric/schema implies LLM proof grading, while its
    # absence implies a boxed-answer problem verified with ``math_verify``.
    grading_schema = _first_present_value(
        raw_problem,
        (
            "grading_guidelines",
            "rubrics",
            "schema",
            "schema_0",
            "Grading guidelines",
            "details",
        ),
        None,
    )
    if grading_schema is None:
        return "answer"
    if isinstance(grading_schema, str) and not grading_schema.strip():
        return "answer"
    if isinstance(grading_schema, (list, dict)) and not grading_schema:
        return "answer"
    return "proof"


def _coerce_dataset_specs(dataset_path: DatasetSource) -> list[str | dict[str, Any]]:
    if dataset_path is None:
        return []
    if isinstance(dataset_path, (str, dict)):
        return [dataset_path]
    if isinstance(dataset_path, list):
        return dataset_path
    raise TypeError(
        "dataset_path must be None, a string path or hub id, a dataset spec dict, or a list of specs."
    )


def _read_local_problem_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    elif suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            rows = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict) and isinstance(parsed.get("problems"), list):
            rows = [item for item in parsed["problems"] if isinstance(item, dict)]
        else:
            raise ValueError(
                "JSON dataset must be a list of problem objects or contain 'problems'."
            )
    else:
        raise ValueError("Unsupported dataset format. Expected .jsonl or .json.")

    return rows


def _read_hub_problem_rows(
    spec: str | dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    if isinstance(spec, str):
        hub_id = spec
        config = None
        split = "train"
        trust_remote_code = True
    else:
        hub_id = str(spec.get("hub_id") or spec.get("dataset") or "").strip()
        config = spec.get("config")
        split = spec.get("split", "train")
        trust_remote_code = spec.get("trust_remote_code", True)

    if not hub_id:
        raise ValueError("Hub dataset specs must include 'hub_id' or 'dataset'.")

    load_args: tuple[Any, ...] = (hub_id,)
    if config is not None:
        load_args += (config,)

    dataset = load_dataset(
        *load_args,
        split=split,
        trust_remote_code=trust_remote_code,
    )
    rows = [dict(row) for row in dataset]
    logger.info(
        "Loaded QED math hub dataset %s%s split=%s with %d rows",
        hub_id,
        f"/{config}" if config else "",
        split,
        len(rows),
    )
    return rows, hub_id


def _first_present_value(
    raw_problem: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Return the first non-None value for the provided keys.

    This uses key presence rather than truthiness so explicit empty strings in
    canonical fields are preserved instead of silently falling through to alias
    fields from other dataset formats.
    """
    for key in keys:
        if key in raw_problem and raw_problem[key] is not None:
            return raw_problem[key]
    return default


def _normalize_problem(
    raw_problem: dict[str, Any], index: int, dataset_source: str
) -> dict:
    """Map heterogeneous dataset rows to canonical QED-Math fields.

    Canonical environment fields:
    - problem <- problem | task | Problem
    - reference_solution <- reference_solution | solution | answer | Solution
    - grading_guidelines <- grading_guidelines | rubrics | schema | schema_0 |
      Grading guidelines | details
    - dataset_source <- dataset_source | dataset | data_source | loader default
    """
    problem = _first_present_value(raw_problem, ("problem", "task", "Problem"))
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Dataset row is missing a non-empty problem statement.")

    reference_solution = _first_present_value(
        raw_problem,
        ("reference_solution", "solution", "answer", "Solution"),
        "",
    )
    grading_guidelines = _first_present_value(
        raw_problem,
        (
            "grading_guidelines",
            "rubrics",
            "schema",
            "schema_0",
            "Grading guidelines",
            "details",
        ),
        "",
    )
    problem_id = _first_present_value(
        raw_problem,
        ("problem_id", "id"),
        f"problem_{index:06d}",
    )
    resolved_dataset_source = _first_present_value(
        raw_problem,
        ("dataset_source", "dataset", "data_source"),
        dataset_source,
    )
    problem_type = _canonical_problem_type(raw_problem)
    default_max_attempts = 1 if problem_type != "multi_step" else 3
    max_attempts = _coerce_positive_int(
        _first_present_value(
            raw_problem, ("max_attempts", "attempts", "num_attempts"), None
        ),
        default=default_max_attempts,
    )
    success_score_threshold = _coerce_positive_int(
        _first_present_value(raw_problem, ("success_score_threshold",), None),
        default=7,
    )
    evaluation_mode = _first_present_value(raw_problem, ("evaluation_mode",), None)
    if isinstance(evaluation_mode, str):
        evaluation_mode = evaluation_mode.strip().lower()
    else:
        evaluation_mode = "answer" if problem_type == "answer" else "proof"
    if evaluation_mode not in {"proof", "answer"}:
        evaluation_mode = "proof"

    # Match QED-Nano process_math semantics: answer datasets use boxed gold answers.
    if (
        evaluation_mode == "answer"
        and isinstance(reference_solution, str)
        and "\\boxed{" not in reference_solution
    ):
        reference_solution = f"\\boxed{{{reference_solution}}}"

    # QED-Nano RC stream: when the prompt seen by the actor differs from the
    # original problem (e.g. after reasoning-cache summarization), the dataset
    # row carries an ``original_problem`` field that must be used for grading.
    original_problem = _first_present_value(raw_problem, ("original_problem",), None)

    return {
        "problem": problem,
        "original_problem": original_problem,
        "reference_solution": str(reference_solution),
        "grading_guidelines": grading_guidelines,
        "problem_id": str(problem_id),
        "dataset_source": str(resolved_dataset_source),
        "problem_type": problem_type,
        "max_attempts": max_attempts,
        "success_score_threshold": success_score_threshold,
        "evaluation_mode": evaluation_mode,
    }


def _load_problems_from_spec(
    spec: str | dict[str, Any],
    start_index: int,
) -> list[dict[str, Any]]:
    if isinstance(spec, dict):
        if "path" in spec:
            path = Path(spec["path"])
            rows = _read_local_problem_rows(path)
            dataset_source = str(
                spec.get("dataset_source") or spec.get("dataset") or path.stem
            )
        elif "hub_id" in spec or "dataset" in spec:
            rows, dataset_source = _read_hub_problem_rows(spec)
        else:
            raise ValueError(
                "Dataset spec dict must include either 'path' for local files or 'hub_id'/'dataset' for hub datasets."
            )
    else:
        candidate_path = Path(spec)
        if candidate_path.exists() or candidate_path.suffix.lower() in {
            ".json",
            ".jsonl",
        }:
            rows = _read_local_problem_rows(candidate_path)
            dataset_source = candidate_path.stem
        elif "/" in spec:
            rows, dataset_source = _read_hub_problem_rows(spec)
        else:
            raise ValueError(
                "Dataset source must be a local .json/.jsonl path or a Hugging Face dataset id like 'owner/name'."
            )

    problems: list[dict[str, Any]] = []
    for offset, row in enumerate(rows, start=1):
        try:
            problems.append(
                _normalize_problem(row, start_index + offset, dataset_source)
            )
        except ValueError:
            logger.warning(
                "Skipping invalid QED math dataset row from %s at offset %d",
                dataset_source,
                offset,
            )

    return problems


def load_problems(dataset_path: DatasetSource) -> list[dict]:
    """Load problems from local JSON/JSONL files or Hugging Face datasets.

    Supported sources:
    - Local JSONL: one problem object per line
    - Local JSON: list[object] or {"problems": list[object]}
    - Hugging Face hub id: "owner/name"
    - Hugging Face spec dict: {"hub_id": ..., "config": ..., "split": ...}
    - Lists mixing the above forms
    """
    dataset_specs = _coerce_dataset_specs(dataset_path)
    if not dataset_specs:
        return _bootstrap_problems()

    problems: list[dict[str, Any]] = []
    for spec in dataset_specs:
        problems.extend(_load_problems_from_spec(spec, start_index=len(problems)))

    if not problems:
        raise ValueError(
            "No valid QED math problems were loaded from the configured dataset source."
        )

    return problems


class QEDMathEnvironment(MCPEnvironment):
    """
    Math proof environment with MCP tools and rubric-based grading.

    This environment provides MCP tools for:
    - get_problem(): Return current problem statement, reference solution,
      and grading guidelines.
    - submit_proof(proof): Grade a proof via MathProofRubric and return
      score (0-7) and normalized reward.
    - get_grading_guidelines(): Return the rubric for the current problem.

    The full implementation includes:
    - Dataset loading from QED-Nano format
    - LLM-based proof grading via rubric
    - Reward normalization to [0, 1]

    Reference:
        - envs/echo_env/server/echo_environment.py - MCP environment pattern
        - QED-Nano: training/pipelinerl/domains/math/rollouts.py - Rollout logic
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        grader_model: str | None = None,
        prompt_name: str | None = None,
        custom_reward_threshold: bool | None = None,
        max_attempts: int | None = None,
        config: QEDMathConfig | None = None,
    ):
        """
        Initialize the QED Math environment.

        Args:
            dataset_path: Optional path to problems dataset.
            grader_model: LLM model to use for grading.
            prompt_name: Grading prompt version.
            custom_reward_threshold: Apply QED-Nano-style score thresholding.
            max_attempts: Default maximum number of attempts when the
                dataset row does not provide max_attempts.
            config: Optional consolidated configuration object.
        """
        mcp = FastMCP("qed_math_env")

        register_mcp_tools(mcp, self)

        super().__init__(mcp)

        # Prefer explicit constructor args when provided; otherwise use config.
        base_config = config or QEDMathConfig()
        self._config = QEDMathConfig(
            dataset_path=(
                dataset_path if dataset_path is not None else base_config.dataset_path
            ),
            grader_model=(
                grader_model if grader_model is not None else base_config.grader_model
            ),
            prompt_name=(
                prompt_name if prompt_name is not None else base_config.prompt_name
            ),
            custom_reward_threshold=(
                custom_reward_threshold
                if custom_reward_threshold is not None
                else base_config.custom_reward_threshold
            ),
            max_attempts=(
                max_attempts if max_attempts is not None else base_config.max_attempts
            ),
            discount_factor=base_config.discount_factor,
            buffer_tokens=base_config.buffer_tokens,
            max_tokens=base_config.max_tokens,
            reasoning_delimiters=base_config.reasoning_delimiters,
            grader_temperature=base_config.grader_temperature,
            grader_max_output_tokens=base_config.grader_max_output_tokens,
            answer_reward_preset=base_config.answer_reward_preset,
            verifier_workers=base_config.verifier_workers,
            verifier_queue_size=base_config.verifier_queue_size,
            verifier_request_timeout_seconds=base_config.verifier_request_timeout_seconds,
            verifier_max_retries=base_config.verifier_max_retries,
            verifier_strict=base_config.verifier_strict,
            verifier_numeric_precision=base_config.verifier_numeric_precision,
            verifier_float_rounding=base_config.verifier_float_rounding,
        )

        self._dataset_path = self._config.dataset_path
        self._grader_model = self._config.grader_model
        self._prompt_name = self._config.prompt_name
        self._prompt_template = load_evaluator_prompt(self._config.prompt_name)
        self._problems: list[dict] = load_problems(self._config.dataset_path)
        self._current_problem: dict | None = None
        self._gold_cache_signature = self._build_gold_cache_signature()
        self._gold_cache_problem_count = len(self._problems)
        self._gold_answer_cache: dict[tuple[str, tuple[bool, int, int]], str] = {}
        self._gold_cache_hits = 0
        self._gold_cache_misses = 0
        self._build_gold_answer_cache()
        # Read judge credentials from JUDGE_* env vars (set in Docker) with
        # fallbacks to the standard OPENAI_* names for local development.
        judge_api_base_url = os.environ.get("JUDGE_API_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL"
        )
        judge_api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        judge_model = os.environ.get("JUDGE_MODEL") or self._config.grader_model
        self._rubric = MathProofRubric(
            grader_model=judge_model,
            prompt_template=self._prompt_template,
            custom_threshold=self._config.custom_reward_threshold,
            api_base_url=judge_api_base_url,
            api_key=judge_api_key,
            temperature=self._config.grader_temperature,
            max_output_tokens=self._config.grader_max_output_tokens,
        )
        # Initialize verifier service for answer-mode grading.
        # The service is started lazily on first verify_answer call or explicit start.
        self._verifier_service = MathVerifierService(
            max_workers=self._config.verifier_workers,
            queue_size=self._config.verifier_queue_size,
            request_timeout_seconds=self._config.verifier_request_timeout_seconds,
            max_retries=self._config.verifier_max_retries,
            strict=self._config.verifier_strict,
            numeric_precision=self._config.verifier_numeric_precision,
            float_rounding=self._config.verifier_float_rounding,
        )
        self._discount_factor = self._config.discount_factor
        self._buffer_tokens = self._config.buffer_tokens
        self._max_tokens = self._config.max_tokens
        self._reasoning_delimiters = self._config.reasoning_delimiters
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._attempt_count = 0
        self._current_max_attempts = max(1, int(self._config.max_attempts))
        # Token count injected by the training harness via the HTTP step request
        # Set in step_async before the MCP dispatch
        # consumed and cleared in submit_proof_payload.
        self._pending_output_length_tokens: int = 0

    def _build_gold_cache_signature(self) -> tuple[bool, int, int]:
        """Return cache settings that affect answer-verifier behavior."""
        return (
            bool(self._config.verifier_strict),
            int(self._config.verifier_numeric_precision),
            int(self._config.verifier_float_rounding),
        )

    def _gold_cache_key(self, problem_id: str) -> tuple[str, tuple[bool, int, int]]:
        """Build a stable answer-cache key for a specific problem id."""
        return (problem_id, self._gold_cache_signature)

    def _build_gold_answer_cache(self) -> None:
        """Build cache of answer-mode gold references keyed by problem id/settings."""
        self._gold_answer_cache = {}
        for problem in self._problems:
            evaluation_mode = problem.get("evaluation_mode", "proof")
            if evaluation_mode != "answer":
                continue

            problem_id = str(problem.get("problem_id", "")).strip()
            if not problem_id:
                continue

            reference_solution = str(problem.get("reference_solution", ""))
            # Validate parseability once at cache-build time for earlier detection,
            # but retain original text for worker-process parsing/verification.
            try:
                _parse_math_verify_expression(reference_solution)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to pre-parse answer-mode gold for problem_id=%s; deferring to runtime verifier.",
                    problem_id,
                )
            self._gold_answer_cache[self._gold_cache_key(problem_id)] = (
                reference_solution
            )

    def _refresh_gold_cache_if_needed(self) -> None:
        """Refresh answer cache when verifier settings or problem set changes."""
        current_signature = self._build_gold_cache_signature()
        current_problem_count = len(self._problems)
        if (
            current_signature != self._gold_cache_signature
            or current_problem_count != self._gold_cache_problem_count
        ):
            self._gold_cache_signature = current_signature
            self._gold_cache_problem_count = current_problem_count
            self._build_gold_answer_cache()

    def _get_cached_gold_answer(self, problem_id: str, fallback: str) -> str:
        """Get cached gold reference for answer-mode grading."""
        self._refresh_gold_cache_if_needed()
        key = self._gold_cache_key(problem_id)
        if key in self._gold_answer_cache:
            self._gold_cache_hits += 1
            return self._gold_answer_cache[key]

        self._gold_cache_misses += 1
        return fallback

    def _gold_cache_hit_rate(self) -> float:
        """Return cache hit rate for answer-mode gold lookup."""
        total = self._gold_cache_hits + self._gold_cache_misses
        if total == 0:
            return 0.0
        return float(self._gold_cache_hits) / float(total)

    async def step_async(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Async step override — fixes the run_async_safely event-loop deadlock
        described in https://github.com/huggingface/OpenEnv/issues/455.

        The WebSocket server detects this override and calls it directly on the
        outer event loop instead of dispatching step() into a thread executor,
        which would cause asyncio.run() inside run_async_safely to conflict with
        the FastMCP server's anyio transport.
        """
        self._state.step_count += 1

        from openenv.core.env_server.mcp_types import (
            CallToolAction,
            CallToolObservation,
            ListToolsAction,
            ListToolsObservation,
            Tool,
            ToolError,
            ToolErrorType,
        )

        # LLM judge calls (submit_proof) can take several minutes; use a generous
        # ceiling that matches MathProofRubric.timeout_seconds.
        MCP_TIMEOUT = 600.0

        if isinstance(action, ListToolsAction):
            try:
                tools_result = await self._async_list_tools()
                tools = [
                    Tool(
                        name=t.name,
                        description=t.description or "",
                        input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
                    )
                    for t in tools_result
                ]
                return ListToolsObservation(tools=tools)
            except Exception as exc:
                return ListToolsObservation(
                    tools=[],
                    metadata={"error": str(exc), "error_type": "list_tools_failed"},
                )

        if isinstance(action, CallToolAction):
            # Capture harness-supplied token count before MCP dispatch so
            # submit_proof_payload can use it without the agent touching it.
            if action.tool_name == "submit_proof":
                raw = kwargs.get("output_length_tokens", 0)
                try:
                    self._pending_output_length_tokens = max(0, int(raw))
                except (TypeError, ValueError):
                    self._pending_output_length_tokens = 0
            timeout = timeout_s if timeout_s is not None else MCP_TIMEOUT
            try:
                result = await asyncio.wait_for(
                    self._async_call_tool(action.tool_name, action.arguments),
                    timeout=timeout,
                )
                obs = CallToolObservation(tool_name=action.tool_name, result=result)
                return self._enrich_submit_proof_obs(action, obs)
            except asyncio.TimeoutError:
                self._pending_output_length_tokens = 0
                return CallToolObservation(
                    tool_name=action.tool_name,
                    result=None,
                    error=ToolError(
                        error_type=ToolErrorType.TIMEOUT,
                        message=f"Tool '{action.tool_name}' timed out after {timeout}s",
                    ),
                )
            except Exception as exc:
                self._pending_output_length_tokens = 0
                return CallToolObservation(
                    tool_name=action.tool_name,
                    result=None,
                    error=ToolError(
                        error_type=ToolErrorType.EXECUTION_ERROR,
                        message=str(exc),
                    ),
                )

        # Non-MCP actions fall through to the sync _step_impl
        return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment and load a new problem.

        Args:
            seed: Optional random seed for problem selection.
            episode_id: Optional episode identifier.
            problem_id: Optional specific problem ID to load (via kwargs).
            **kwargs: Additional reset parameters.

        Returns:
            ProblemObservation with problem details.

        Loads a problem from the dataset and returns a ProblemObservation
        with problem, reference_solution, grading_guidelines, problem_id,
        and dataset_source.
        """
        selected_problem_id = kwargs.pop("problem_id", None)
        self._refresh_gold_cache_if_needed()

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._attempt_count = 0
        self._reset_count += 1

        # Reset rubric state if the rubric exposes a reset hook.
        rubric_reset = getattr(self._rubric, "reset", None)
        if callable(rubric_reset):
            rubric_reset()

        if self._problems:
            if selected_problem_id is not None:
                selected = next(
                    (
                        problem
                        for problem in self._problems
                        if problem.get("problem_id") == selected_problem_id
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError(
                        f"problem_id {selected_problem_id!r} not found in the loaded dataset."
                    )
                self._current_problem = selected
            elif seed is not None:
                rng = random.Random(seed)
                self._current_problem = rng.choice(self._problems)
            else:
                self._current_problem = random.choice(self._problems)
        else:
            self._current_problem = None

        if self._current_problem is None:
            return Observation(
                done=False,
                reward=0.0,
                metadata={
                    "error": "No problems loaded. Provide a valid dataset_path.",
                    "status": "empty",
                },
            )

        self._current_max_attempts = _coerce_positive_int(
            self._current_problem.get("max_attempts"),
            default=max(1, int(self._config.max_attempts)),
        )

        return ProblemObservation(
            problem=self._current_problem.get("problem", ""),
            reference_solution=(
                self._current_problem.get("reference_solution", "")
                if self._current_problem.get("evaluation_mode", "proof") == "answer"
                else ""
            ),
            grading_guidelines=parse_schema(
                self._current_problem.get("grading_guidelines", "") or ""
            ),
            problem_id=self._current_problem.get("problem_id", ""),
            dataset_source=self._current_problem.get("dataset_source", ""),
            problem_type=self._current_problem.get("problem_type", "proof"),
            max_attempts=self._current_max_attempts,
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "reset_count": self._reset_count,
                "step_count": self._state.step_count,
                "attempt_count": self._attempt_count,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.

        Args:
            action: The action to execute.
            timeout_s: Optional timeout.
            **kwargs: Additional arguments.

        Returns:
            Observation with error message.
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use MCP tools (CallToolAction) for interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Args:
            action: The action to execute.
            timeout_s: Optional timeout for the action.
            **kwargs: Additional arguments.

        Returns:
            Observation from the action execution.
        """
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        return self._enrich_submit_proof_obs(action, obs)

    def _enrich_submit_proof_obs(self, action: Any, obs: Any) -> Any:
        """Attach reward/done from submit_proof payload onto the CallToolObservation."""
        if (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_proof"
            and isinstance(obs, CallToolObservation)
            and obs.error is None
        ):
            payload = self._extract_tool_payload(obs.result)
            if isinstance(payload, dict):
                proof_obs = ProofSubmissionObservation(
                    proof=str(payload.get("proof", "")),
                    score=int(payload.get("score", 0)),
                    feedback=str(payload.get("feedback", "")),
                    done=bool(payload.get("done", True)),
                    reward=float(payload.get("reward", 0.0)),
                    problem_type=str(payload.get("problem_type", "proof")),
                    attempt_number=int(payload.get("attempt_number", 1)),
                    attempts_remaining=int(payload.get("attempts_remaining", 0)),
                    is_correct=bool(payload.get("is_correct", False)),
                    metadata=dict(payload.get("metadata", {})),
                )
                metadata = dict(obs.metadata)
                metadata["proof_submission"] = proof_obs.model_dump()
                return CallToolObservation(
                    tool_name=obs.tool_name,
                    result=obs.result,
                    error=obs.error,
                    done=proof_obs.done,
                    reward=proof_obs.reward,
                    metadata=metadata,
                )
        return obs

    @staticmethod
    def _extract_tool_payload(tool_result: Any) -> Any:
        """Extract a structured tool payload from FastMCP CallToolResult-like values."""
        if tool_result is None:
            return None

        data = getattr(tool_result, "data", None)
        if data is not None:
            return data

        structured_content = getattr(tool_result, "structured_content", None)
        if structured_content is not None:
            return structured_content

        if isinstance(tool_result, dict):
            return tool_result

        return None

    @staticmethod
    def _utc_now_iso() -> str:
        """Return a UTC timestamp string for progress events."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _chunk_feedback(feedback: str, chunk_size: int = 280) -> list[str]:
        """Split grader feedback into chunks for client-side streamed rendering."""
        if not feedback:
            return []
        return [
            feedback[i : i + chunk_size] for i in range(0, len(feedback), chunk_size)
        ]

    def _build_grading_progress(
        self,
        proof: str,
        feedback: str,
        is_correct: bool,
        done: bool,
    ) -> dict[str, Any]:
        """Build WebSocket-friendly progress and streaming feedback metadata."""
        feedback_chunks = self._chunk_feedback(feedback)
        return {
            "status": "completed" if done else "in_progress",
            "progress": 1.0,
            "events": [
                {
                    "stage": "submission_received",
                    "progress": 0.2,
                    "message": "Proof submission received.",
                    "timestamp": self._utc_now_iso(),
                },
                {
                    "stage": "grading_started",
                    "progress": 0.6,
                    "message": "Grading started.",
                    "timestamp": self._utc_now_iso(),
                },
                {
                    "stage": "grading_completed",
                    "progress": 0.9,
                    "message": "Grading completed.",
                    "timestamp": self._utc_now_iso(),
                },
                {
                    "stage": "result_ready",
                    "progress": 1.0,
                    "message": "Result ready for client consumption.",
                    "timestamp": self._utc_now_iso(),
                },
            ],
            "realtime": {
                "websocket_supported": True,
                "submission_type": "proof" if proof.strip() else "empty",
            },
            "streaming_feedback": {
                "chunks": feedback_chunks,
                "chunk_count": len(feedback_chunks),
                "is_final": True,
            },
            "is_correct": is_correct,
        }

    def get_problem_payload(self) -> dict:
        """Payload for MCP tool get_problem."""
        if self._current_problem is None:
            return {
                "error": "No active problem. Call reset() first.",
                "done": False,
                "reward": 0.0,
            }

        problem_type = self._current_problem.get("problem_type", "proof")
        evaluation_mode = self._current_problem.get("evaluation_mode")
        if isinstance(evaluation_mode, str):
            evaluation_mode = evaluation_mode.strip().lower()
        else:
            evaluation_mode = "answer" if problem_type == "answer" else "proof"
        if evaluation_mode not in {"proof", "answer"}:
            evaluation_mode = "proof"

        payload = {
            "problem": self._current_problem.get("problem", ""),
            "grading_guidelines": self._current_grading_guidelines_text(),
            "problem_id": self._current_problem.get("problem_id", ""),
            "dataset_source": self._current_problem.get("dataset_source", ""),
            "problem_type": problem_type,
            "max_attempts": self._current_max_attempts,
            "attempt_count": self._attempt_count,
            "done": False,
            "reward": 0.0,
        }

        # Keep answer-mode parity with QED-Nano while preventing proof-mode
        # agents from seeing the grading key via MCP.
        if evaluation_mode == "answer":
            payload["reference_solution"] = self._current_problem.get(
                "reference_solution", ""
            )

        return payload

    @staticmethod
    def _verify_math(
        prediction: str,
        gold: str,
        strict: bool = True,
        max_prediction_length: int = 1000,
    ) -> str:
        """QED-Nano-style math verification using math_verify.

        Returns one of: correct, wrong, no_answer, unparsable.
        """
        try:
            if not isinstance(prediction, str) or not isinstance(gold, str):
                raise ValueError("Prediction and gold must be strings")

            boxed_start = prediction.rfind("\\boxed{")
            if boxed_start < 0:
                raise NoAnswerException()

            boxed_prediction = prediction[boxed_start:]
            if "\\boxed{}" in boxed_prediction:
                raise EmptyBoxedException()

            if len(boxed_prediction) > max_prediction_length:
                raise UnparsableException()

            gold_parsed = _parse_math_verify_expression(gold)
            boxed_prediction_parsed = _parse_math_verify_expression(boxed_prediction)
            if not gold_parsed:
                raise ValueError("Failed to parse gold answer.")
            if not boxed_prediction_parsed:
                raise ValueError("Failed to parse prediction.")

            equivalent = math_verify.verify(
                gold_parsed,
                boxed_prediction_parsed,
                strict=strict,
                timeout_seconds=1,
            )
            return "correct" if equivalent else "wrong"

        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, NoAnswerException):
                return "no_answer"
            return "unparsable"

    async def _grade_answer_submission(
        self,
        submission: str,
        expected_answer: str,
        problem_id: str = "",
    ) -> GradingResult:
        """Grade an answer submission using process-based verifier workers.

        Args:
            submission: The submitted answer (may contain \\boxed{...}).
            expected_answer: The reference answer.

        Returns:
            GradingResult with score (0 or 7), feedback, and metrics.
        """
        response = await self._verifier_service.verify_answer(
            prediction=submission,
            gold=expected_answer,
            strict=self._config.verifier_strict,
            max_prediction_length=1000,
            numeric_precision=self._config.verifier_numeric_precision,
            float_rounding=self._config.verifier_float_rounding,
        )
        answer_status = response.status
        verifier_health = await self._verifier_service.health_probe()
        verifier_service_metrics = await self._verifier_service.metrics_snapshot()

        logger.info(
            "qed_math_verifier_result request_id=%s problem_id=%s evaluation_mode=answer status=%s elapsed_ms=%.3f retry_count=%d",
            response.request_id,
            problem_id,
            answer_status,
            float(response.elapsed_ms),
            int(response.retry_count),
        )

        score = 7 if answer_status == "correct" else 0
        reward = answer_reward_for_status(
            answer_status, self._config.answer_reward_preset
        )
        feedback = f"answer_status={answer_status}"

        return GradingResult(
            score=score,
            feedback=feedback,
            reward=reward,
            metrics={
                "verifier/rollouts/success": int(answer_status == "correct"),
                "verifier/rollouts/failure": int(answer_status != "correct"),
                "verifier/failures/timeout": int(answer_status == "timeout"),
                "verifier/failures/rate_limit": 0,
                "verifier/failures/no_input": 0,
                "verifier/failures/no_score_tag": 0,
                "verifier/failures/all_attempts_failed": int(
                    answer_status in {"internal_error", "timeout"}
                ),
                "verifier/failures/num_retries": int(response.retry_count),
                "verifier/runtime/latency_per_request": float(response.elapsed_ms),
                "verifier/workers/restart_count": int(
                    verifier_health.get("restart_count", 0)
                ),
                "verifier/workers/worker_restarted": int(response.worker_restarted),
                "verifier/queue/depth": int(
                    verifier_health.get("inflight_requests", 0)
                ),
                "verifier/requests/count": int(
                    verifier_service_metrics.get("verifier/requests/count", 0)
                ),
                "verifier/requests/latency_ms": float(
                    verifier_service_metrics.get("verifier/requests/latency_ms", 0.0)
                ),
                "verifier/requests/timeout_count": int(
                    verifier_service_metrics.get("verifier/requests/timeout_count", 0)
                ),
                "verifier/requests/error_count": int(
                    verifier_service_metrics.get("verifier/requests/error_count", 0)
                ),
                "verifier/workers/heartbeat_lag_ms": float(
                    verifier_service_metrics.get(
                        "verifier/workers/heartbeat_lag_ms", 0.0
                    )
                ),
                "verifier/cache/hit_rate": self._gold_cache_hit_rate(),
                "verifier/runtime/input_tokens": 0,
                "verifier/runtime/output_tokens": 0,
            },
        )

    async def shutdown_verifier_service(self) -> None:
        """Explicit async shutdown hook for deterministic test and run cleanup."""
        await self._verifier_service.stop()

    def close(self) -> None:
        """Release the verifier worker pool before tearing down MCP resources."""
        self._verifier_service.close_sync()
        super().close()

    def _strip_reasoning(self, text: str) -> str:
        """Strip reasoning traces using configured delimiters."""
        return remove_reasoning(text, self._reasoning_delimiters)

    async def _grade_submission(self, submission: str) -> GradingResult:
        if self._current_problem is None:
            return GradingResult(
                score=0,
                feedback="No active problem. Call reset() first.",
                reward=0.0,
            )

        # Strip reasoning traces (e.g. <think>...</think>) before grading,
        # matching QED-Nano rollout behavior (remove_reasoning).
        grading_input = self._strip_reasoning(submission)
        if not grading_input.strip() and submission.strip():
            # Reasoning stripping produced empty output — grade original
            # to avoid silently awarding 0 for valid submissions that
            # don't follow the expected delimiter pattern.
            grading_input = submission

        # Use original_problem for grading when present (QED-Nano RC stream
        # semantics: the actor prompt may be a reformulated version, but the
        # grader must evaluate against the original problem statement).
        problem = self._current_problem.get(
            "original_problem"
        ) or self._current_problem.get("problem", "")
        reference_solution = self._current_problem.get("reference_solution", "")
        grading_guidelines = parse_schema(
            self._current_problem.get("grading_guidelines", "") or ""
        )
        evaluation_mode = self._current_problem.get("evaluation_mode", "proof")

        if evaluation_mode == "answer":
            problem_id = str(self._current_problem.get("problem_id", "")).strip()
            cached_reference_solution = self._get_cached_gold_answer(
                problem_id,
                reference_solution,
            )
            return await self._grade_answer_submission(
                grading_input,
                cached_reference_solution,
                problem_id=problem_id,
            )

        return await self._rubric.grade(
            grading_input,
            problem,
            reference_solution,
            grading_guidelines,
        )

    def _apply_reward_shaping(
        self,
        reward: float,
        output_length_tokens: int,
    ) -> float:
        """Apply discount factor and length penalty to a base reward.

        Ported from QED-Nano: training/pipelinerl/domains/math/rollouts.py.

        Discount: ``reward *= discount_factor ** output_length_tokens``
        Length penalty: additive penalty when output approaches max_tokens.

        Args:
            reward: Base normalized reward from grading.
            output_length_tokens: Server-side token count of the proof text.
                When 0 or negative (empty submission), shaping is skipped.

        Returns:
            Shaped reward.
        """
        if output_length_tokens <= 0:
            return reward

        reward = reward * (self._discount_factor**output_length_tokens)

        if self._buffer_tokens > 0 and self._max_tokens > 0:
            reward += length_penalty(
                self._max_tokens, output_length_tokens, self._buffer_tokens
            )

        return reward

    async def submit_proof_payload(self, proof: str) -> dict:
        """Payload for MCP tool submit_proof.

        Args:
            proof: The proof text submitted by the agent.
        """
        if self._current_problem is None:
            return ProofSubmissionObservation(
                proof=proof,
                score=0,
                feedback="Proof not graded because no problem is active.",
                done=True,
                reward=0.0,
                problem_type="proof",
                attempt_number=1,
                attempts_remaining=0,
                is_correct=False,
                metadata={"error": "No active problem. Call reset() first."},
            ).model_dump()

        self._attempt_count += 1
        problem_type = str(self._current_problem.get("problem_type", "proof"))
        is_multi_step = problem_type == "multi_step"

        if not proof.strip():
            result = GradingResult(
                score=0,
                feedback="Empty proof submission.",
                reward=0.0,
                metrics={
                    "verifier/rollouts/failure": 1,
                    "verifier/failures/no_input": 1,
                },
            )
        else:
            result = await self._grade_submission(proof)

        # Use harness-supplied token count only
        output_length_tokens = self._pending_output_length_tokens
        self._pending_output_length_tokens = 0  # consume
        shaped_reward = self._apply_reward_shaping(result.reward, output_length_tokens)

        success_threshold = _coerce_positive_int(
            self._current_problem.get("success_score_threshold"),
            default=7,
        )
        is_correct = result.score >= success_threshold
        attempts_remaining = max(0, self._current_max_attempts - self._attempt_count)
        done = (not is_multi_step) or is_correct or attempts_remaining == 0

        feedback = result.feedback
        if is_multi_step and not done:
            feedback = (
                f"{result.feedback} Continue: "
                f"attempt {self._attempt_count}/{self._current_max_attempts}."
            )

        grading_progress = self._build_grading_progress(
            proof=proof,
            feedback=feedback,
            is_correct=is_correct,
            done=done,
        )

        return ProofSubmissionObservation(
            proof=proof,
            score=result.score,
            feedback=feedback,
            done=done,
            reward=shaped_reward,
            problem_type=problem_type,
            attempt_number=self._attempt_count,
            attempts_remaining=attempts_remaining,
            is_correct=is_correct,
            metadata={
                "grading_progress": grading_progress,
                "status": grading_progress["status"],
                "base_reward": result.reward,
                "shaped_reward": shaped_reward,
                "output_length_tokens": output_length_tokens,
                "verifier_metrics": self._build_verifier_metrics(
                    result, shaped_reward, output_length_tokens, is_correct
                ),
            },
        ).model_dump()

    def _build_verifier_metrics(
        self,
        result: GradingResult,
        shaped_reward: float,
        output_length_tokens: int,
        is_correct: bool,
    ) -> dict[str, float | int | str]:
        """Build TrackIO-compatible verifier metrics dict.

        Aggregates rubric-level metrics from ``result.metrics`` with
        episode-level reward and problem metadata.  The returned dict
        uses QED-Nano naming conventions so it can be forwarded directly
        to TrackIO / WandB without transformation.
        """
        metrics: dict[str, float | int | str] = dict(result.metrics)

        # Reward breakdown
        metrics["reward/base"] = result.reward
        metrics["reward/shaped"] = shaped_reward
        metrics["reward/score_raw"] = result.score
        if (
            output_length_tokens > 0
            and self._buffer_tokens > 0
            and self._max_tokens > 0
        ):
            from .rubric import length_penalty as _lp

            metrics["reward/overlong_penalty"] = _lp(
                self._max_tokens, output_length_tokens, self._buffer_tokens
            )
        else:
            metrics["reward/overlong_penalty"] = 0.0

        # Episode context
        metrics["episode/attempt_number"] = self._attempt_count
        metrics["episode/is_correct"] = int(is_correct)
        if self._current_problem is not None:
            metrics["episode/problem_type"] = str(
                self._current_problem.get("problem_type", "proof")
            )
            metrics["episode/dataset_source"] = str(
                self._current_problem.get("dataset_source", "unknown")
            )

        return metrics

    def get_grading_guidelines_payload(self) -> dict:
        """Payload for MCP tool get_grading_guidelines."""
        if self._current_problem is None:
            return {
                "error": "No active problem. Call reset() first.",
                "grading_guidelines": "",
                "done": False,
                "reward": 0.0,
            }

        return {
            "grading_guidelines": self._current_grading_guidelines_text(),
            "problem_id": self._current_problem.get("problem_id", ""),
            "done": False,
            "reward": 0.0,
        }

    def _current_grading_guidelines_text(self) -> str:
        """Return current grading guidelines normalized to markdown text."""
        if self._current_problem is None:
            return ""
        return parse_schema(self._current_problem.get("grading_guidelines", "") or "")

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
