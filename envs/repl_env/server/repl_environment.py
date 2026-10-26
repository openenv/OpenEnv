# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment Implementation.

A Python REPL environment for training language models on code execution tasks,
based on the Recursive Language Models (RLM) paradigm.

References:
- RLM Paper: https://arxiv.org/abs/2512.24601
- Prime Intellect Blog: https://www.primeintellect.ai/blog/rlm
- Alex Zhang Blog: https://alexzhang13.github.io/blog/2025/rlm/
"""

import os
import re
from collections.abc import Callable
from typing import Any, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import CodeBlockResult, REPLAction, REPLObservation, REPLState
except ImportError:
    try:
        from repl_env.models import (
            CodeBlockResult,
            REPLAction,
            REPLObservation,
            REPLState,
        )
    except ImportError:
        from models import CodeBlockResult, REPLAction, REPLObservation, REPLState

try:
    from ..recursive_controller import create_server_recursive_controller
    from ..rubrics import REPLRubric
    from .python_executor import PythonExecutor
except ImportError:
    try:
        from repl_env.recursive_controller import create_server_recursive_controller
        from repl_env.rubrics import REPLRubric
        from .python_executor import PythonExecutor
    except ImportError:
        from recursive_controller import create_server_recursive_controller
        from rubrics import REPLRubric

        from .python_executor import PythonExecutor


class REPLEnvironment(Environment):
    """
    A REPL environment for training language models to use code execution.

    Based on the Recursive Language Models (RLM) paradigm, this environment allows
    language models to:
    - Execute Python code in a sandboxed REPL
    - Work with large contexts loaded as variables
    - Finalize answers via FINAL(), FINAL_VAR(), or answer dict pattern
    - Optionally make recursive LLM calls via llm_query() / llm_query_batched()

    Supports two finalization patterns:
    1. RLM-style: print('FINAL(answer)') or print('FINAL_VAR(var_name)')
    2. Prime Intellect style: answer = {"content": "...", "ready": True}

    Example:
        >>> env = REPLEnvironment(context="Hello World", task_prompt="Count chars")
        >>> obs = env.reset()
        >>> print(obs.context_preview)  # "Hello World"
        >>>
        >>> obs = env.step(REPLAction(code="result = len(context)"))
        >>> print(obs.result.success)  # True
        >>> print(obs.available_variables)  # ["context", "result", "answer"]
        >>>
        >>> obs = env.step(REPLAction(code="print(f'FINAL({result})')"))
        >>> print(obs.done)  # True
        >>> print(obs.metadata["final_answer"])  # "11"
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        context: Optional[str] = None,
        task_prompt: Optional[str] = None,
        max_iterations: int = 30,
        max_output_length: int = 8192,
        context_preview_length: int = 500,
        rubric: Optional[REPLRubric] = None,
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_batch_fn: Optional[Callable[[List[str]], List[str]]] = None,
        subcall_fn: Optional[Callable[[str, Optional[str]], str]] = None,
        subcall_batch_fn: Optional[
            Callable[[List[str], Optional[str]], List[str]]
        ] = None,
        rlm_max_depth: int = 1,
        rlm_max_iterations: int | None = None,
    ):
        """Initialize the REPL environment.

        Args:
            context: Initial context to load (can also be set via REPL_CONTEXT env var)
            task_prompt: Task description (can also be set via REPL_TASK_PROMPT env var)
            max_iterations: Maximum steps per episode (default 30, env var REPL_MAX_ITERATIONS)
            max_output_length: Max chars for stdout/stderr per turn (default 8192)
            context_preview_length: Chars to show in context preview (default 500)
            rubric: Optional REPLRubric for reward computation (default: REPLRubric())
            llm_query_fn: Optional function for llm_query() support
            llm_batch_fn: Optional function for llm_query_batched() support
            subcall_fn: Optional function for recursive rlm_query() support
            subcall_batch_fn: Optional function for recursive rlm_query_batched() support
            rlm_max_depth: Max recursion depth for server-backed rlm_query()
            rlm_max_iterations: Max iterations for recursive child runners
        """
        self.initial_context = context or os.environ.get("REPL_CONTEXT", "")
        self.initial_task_prompt = task_prompt or os.environ.get("REPL_TASK_PROMPT", "")
        self.max_iterations = int(os.environ.get("REPL_MAX_ITERATIONS", max_iterations))
        self.max_output_length = max_output_length
        self.context_preview_length = context_preview_length

        # Rubric for reward computation (OpenEnv RFC 004)
        self.rubric = rubric or REPLRubric()

        # Optional LLM functions for recursive calls
        self.llm_query_fn = llm_query_fn
        self.llm_batch_fn = llm_batch_fn
        self.subcall_fn = subcall_fn
        self.subcall_batch_fn = subcall_batch_fn
        self.rlm_max_depth = rlm_max_depth
        self.rlm_max_iterations = rlm_max_iterations or max_iterations

        # State (initialized on reset)
        self._state: Optional[REPLState] = None
        self._executor: Optional[PythonExecutor] = None
        self._runtime_controller = None
        self._runtime_controller_chat_fn: Optional[Callable[..., str]] = None
        self._current_llm_model: Optional[str] = None

    @staticmethod
    def _resolve_model(llm_model: Optional[str]) -> str:
        """Resolve the effective model name: explicit arg > env var > hard default."""
        return llm_model or os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-9B")

    @staticmethod
    def _build_hf_chat_fn(
        hf_token: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> Callable[..., str]:
        try:
            from huggingface_hub import InferenceClient, InferenceTimeoutError
        except ImportError:
            raise RuntimeError("huggingface_hub is required for HF-backed recursion")

        default_model = REPLEnvironment._resolve_model(llm_model)
        client = InferenceClient(model=default_model, token=hf_token, timeout=300)

        def chat_fn(messages: list[dict[str, str]], model: str | None = None) -> str:
            try:
                response = client.chat.completions.create(
                    model=model or default_model,
                    messages=messages,
                    max_tokens=2048,
                    # Qwen3.5 non-thinking mode for precise coding tasks (from model card)
                    temperature=0.6,
                    top_p=0.95,
                    presence_penalty=0.0,
                    extra_body={
                        "top_k": 20,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                return response.choices[0].message.content or ""
            except InferenceTimeoutError:
                return "Error: LLM inference timed out"
            except Exception as e:
                return f"Error: {e}"

        return chat_fn

    def _create_llm_functions(
        self,
        hf_token: Optional[str],
        llm_model: Optional[str] = None,
    ) -> None:
        """Create LLM/subcall functions dynamically using client-provided token."""
        try:
            chat_fn = self._build_hf_chat_fn(hf_token, llm_model)
        except RuntimeError:
            return

        self._runtime_controller_chat_fn = chat_fn
        self._runtime_controller = create_server_recursive_controller(
            chat_fn,
            max_depth=self.rlm_max_depth,
            max_iterations=self.rlm_max_iterations,
        )
        self.llm_query_fn = self._runtime_controller.llm_query_fn
        self.llm_batch_fn = self._runtime_controller.llm_batch_fn
        self.subcall_fn = self._runtime_controller.rlm_query_fn
        self.subcall_batch_fn = self._runtime_controller.rlm_batch_fn

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        context: Optional[str] = None,
        task_prompt: Optional[str] = None,
        hf_token: Optional[str] = None,
        llm_model: Optional[str] = None,
        **kwargs: Any,
    ) -> REPLObservation:
        """Reset the environment with optional new context.

        Args:
            seed: Optional random seed (for reproducibility)
            episode_id: Optional episode identifier (if not provided, one is generated)
            context: Context to load (overrides initial_context)
            task_prompt: Task description (overrides initial_task_prompt)
            hf_token: Optional HuggingFace token for llm_query/llm_query_batched.
                      If provided, creates LLM functions using this token.
                      Security: Token is NOT stored in state or logged.
            llm_model: Optional model name for LLM functions (default: from env or Qwen3.5-9B)
            **kwargs: Additional reset parameters including:
                expected_answer: Ground truth for rubric-based reward scoring
                rlm_max_depth: Override max recursion depth
                rlm_max_iterations: Override max iterations for recursive child runners

        Returns:
            Initial REPLObservation with environment ready message
        """
        effective_context = context or self.initial_context
        effective_task_prompt = task_prompt or self.initial_task_prompt

        # Set expected answer for rubric-based reward computation
        expected_answer = kwargs.get("expected_answer")
        self.rubric.reset()
        if expected_answer is not None:
            self.rubric.set_expected(expected_answer)

        runtime_rlm_max_depth = kwargs.get("rlm_max_depth")
        if runtime_rlm_max_depth is None:
            runtime_rlm_max_depth = self.rlm_max_depth
        runtime_rlm_max_depth = int(runtime_rlm_max_depth)

        runtime_rlm_max_iterations = kwargs.get("rlm_max_iterations")
        if runtime_rlm_max_iterations is None:
            runtime_rlm_max_iterations = self.rlm_max_iterations
        runtime_rlm_max_iterations = int(runtime_rlm_max_iterations)

        # Detect if recursion config changed — controller must be rebuilt
        depth_changed = (
            runtime_rlm_max_depth != self.rlm_max_depth
            or runtime_rlm_max_iterations != self.rlm_max_iterations
        )
        self.rlm_max_depth = runtime_rlm_max_depth
        self.rlm_max_iterations = runtime_rlm_max_iterations

        # Token resolution: explicit hf_token > HF_TOKEN env var > cached HF login.
        # Compare resolved model names so reset(llm_model=<default>) and
        # reset() are treated as equal and don't trigger a redundant rebuild.
        resolved_model = self._resolve_model(llm_model)
        has_runtime_llm = self._runtime_controller is not None
        model_changed = (
            has_runtime_llm and resolved_model != self._current_llm_model
        )
        token_provided = hf_token is not None
        if not self.llm_query_fn or model_changed or token_provided:
            effective_token = (
                hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
            )
            # Pass raw llm_model; _build_hf_chat_fn resolves it via
            # _resolve_model so the client matches the cache key below.
            self._create_llm_functions(effective_token, llm_model)
            self._current_llm_model = resolved_model
        elif depth_changed and self._runtime_controller is not None:
            # Rebuild controller with new depth/iteration config but reuse
            # the existing chat_fn — don't require re-providing credentials.
            self._runtime_controller.close()
            self._runtime_controller = create_server_recursive_controller(
                self._runtime_controller_chat_fn,
                max_depth=self.rlm_max_depth,
                max_iterations=self.rlm_max_iterations,
            )
            self.llm_query_fn = self._runtime_controller.llm_query_fn
            self.llm_batch_fn = self._runtime_controller.llm_batch_fn
            self.subcall_fn = self._runtime_controller.rlm_query_fn
            self.subcall_batch_fn = self._runtime_controller.rlm_batch_fn

        # Initialize state
        self._state = REPLState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            context=effective_context,
            task_prompt=effective_task_prompt,
            iteration=0,
            max_iterations=self.max_iterations,
            namespace_keys=[],
            final_answer=None,
            total_execution_time=0.0,
        )

        # Initialize executor
        self._executor = PythonExecutor(max_output_length=self.max_output_length)

        # Initialize answer dict (Prime Intellect style)
        self._executor.set_variable("answer", {"content": "", "ready": False})

        # Load context into namespace if provided
        if effective_context:
            self._executor.set_context(effective_context)

        def _call_single_query(prompt: str, model: str | None = None) -> str:
            if not self.llm_query_fn:
                raise RuntimeError("llm_query is not configured")
            try:
                return self.llm_query_fn(prompt, model)  # type: ignore[misc]
            except TypeError:
                return self.llm_query_fn(prompt)  # type: ignore[misc]

        def _call_batched_query(
            prompts: List[str], model: str | None = None
        ) -> List[str]:
            if not self.llm_batch_fn:
                raise RuntimeError("llm_query_batched is not configured")
            try:
                return self.llm_batch_fn(prompts, model)  # type: ignore[misc]
            except TypeError:
                return self.llm_batch_fn(prompts)  # type: ignore[misc]

        def _call_recursive_query(prompt: str, model: str | None = None) -> str:
            if self.subcall_fn is None:
                return _call_single_query(prompt, model)
            return self.subcall_fn(prompt, model)

        def _call_recursive_batched(
            prompts: List[str], model: str | None = None
        ) -> List[str]:
            if not prompts:
                return []
            if self.subcall_batch_fn is not None:
                return self.subcall_batch_fn(prompts, model)
            return _call_batched_query(prompts, model)

        # Inject LLM functions if provided
        # Names: llm_query (single), llm_query_batched (official RLM), llm_batch (alias)
        if self.llm_query_fn:
            self._executor.inject_function("llm_query", _call_single_query)
        if self.llm_batch_fn:
            self._executor.inject_function(
                "llm_query_batched", _call_batched_query
            )  # Official name
            self._executor.inject_function("llm_batch", _call_batched_query)  # Alias
        if self.llm_query_fn or self.subcall_fn:
            self._executor.inject_function("rlm_query", _call_recursive_query)
        if self.llm_batch_fn or self.subcall_batch_fn:
            self._executor.inject_function("rlm_query_batched", _call_recursive_batched)

        # Inject FINAL helper function so both FINAL(x) and print(f'FINAL({x})') work
        # Returns the FINAL pattern as a string so it appears in stdout for detection
        def final_helper(value):
            """Helper that returns FINAL(value) string for detection."""
            return f"FINAL({value})"

        self._executor.inject_function("FINAL", final_helper)

        # Inject FINAL_VAR helper that looks up variable and returns FINAL(value)
        # This matches official RLM behavior - strips quotes from var_name and looks up in namespace
        executor = self._executor  # Capture for closure

        def final_var_helper(var_name: str):
            """Look up variable by name and return FINAL(value) for detection."""
            # Strip quotes if present (handles both FINAL_VAR("x") and FINAL_VAR(x))
            var_name_clean = str(var_name).strip().strip("\"'")
            # Look up variable in executor namespace
            value = executor.get_variable(var_name_clean)
            if value is not None:
                return f"FINAL({value})"
            return f"FINAL_VAR({var_name_clean})"  # Fallback for regex detection

        self._executor.inject_function("FINAL_VAR", final_var_helper)

        def show_vars_helper():
            """Return the current non-private variables in the namespace."""
            return sorted(executor.list_variables())

        self._executor.inject_function("SHOW_VARS", show_vars_helper)

        # Update namespace keys
        self._state.namespace_keys = self._executor.list_variables()

        # Build initial message
        message_parts = ["REPL environment initialized."]
        if effective_context:
            message_parts.append(
                f"Context loaded ({len(effective_context)} chars). Use 'context' variable to access it."
            )
        if effective_task_prompt:
            message_parts.append(f"Task: {effective_task_prompt}")
        message_parts.append(
            "Use answer['content'] to store your answer, and set answer['ready'] = True when done."
        )

        return REPLObservation(
            result=CodeBlockResult(
                stdout="\n".join(message_parts),
                stderr="",
                locals_snapshot={},
                execution_time=0.0,
                success=True,
                exception=None,
            ),
            context_preview=(
                effective_context[: self.context_preview_length]
                if effective_context
                else None
            ),
            context_length=len(effective_context) if effective_context else 0,
            available_variables=self._state.namespace_keys,
            iteration=0,
            max_iterations=self.max_iterations,
            done=False,
            metadata={
                "task_prompt": effective_task_prompt,
                "message": "Environment ready.",
            },
        )

    def step(
        self,
        action: REPLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> REPLObservation:
        """Execute code and return observation.

        Args:
            action: REPLAction containing code to execute
            timeout_s: Optional timeout in seconds (not currently used)
            **kwargs: Additional step parameters

        Returns:
            REPLObservation with execution results
        """
        if self._state is None or self._executor is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.iteration += 1

        # Check if agent explicitly signals final answer
        if action.is_final:
            self._state.final_answer = action.final_answer or ""
            obs = self._create_final_observation(
                success=True,
                message="Final answer submitted.",
            )
            obs.reward = self._apply_rubric(action, obs)
            return obs

        # Check iteration limit
        if self._state.iteration >= self.max_iterations:
            # Check if there's a partial answer in the answer dict
            answer_var = self._executor.get_variable("answer")
            if isinstance(answer_var, dict) and answer_var.get("content"):
                self._state.final_answer = str(answer_var.get("content", ""))
            obs = self._create_final_observation(
                success=False,
                message=f"Maximum iterations ({self.max_iterations}) reached.",
            )
            obs.reward = self._apply_rubric(action, obs)
            return obs

        # Execute code
        result = self._executor.execute(action.code)
        self._state.total_execution_time += result["execution_time"]
        self._state.namespace_keys = self._executor.list_variables()

        # Check for final answer patterns
        final_answer = self._extract_final_answer(result["stdout"])
        done = final_answer is not None

        if done:
            self._state.final_answer = final_answer

        obs = REPLObservation(
            result=CodeBlockResult(
                stdout=result["stdout"],
                stderr=result["stderr"],
                locals_snapshot=result["locals_snapshot"],
                execution_time=result["execution_time"],
                success=result["success"],
                exception=result["exception"],
            ),
            context_preview=(
                self._state.context[: self.context_preview_length]
                if self._state.context
                else None
            ),
            context_length=len(self._state.context) if self._state.context else 0,
            available_variables=self._state.namespace_keys,
            iteration=self._state.iteration,
            max_iterations=self.max_iterations,
            done=done,
            metadata={
                "task_prompt": self._state.task_prompt,
                "final_answer": final_answer,
                "execution_time": result["execution_time"],
            },
        )
        obs.reward = self._apply_rubric(action, obs)
        return obs

    def _extract_final_answer(self, stdout: str) -> Optional[str]:
        """Extract final answer from output.

        Supports multiple patterns:
        1. RLM-style: FINAL(answer) in stdout
        2. RLM-style: FINAL_VAR(variable_name) in stdout
        3. Prime Intellect style: answer = {"content": "...", "ready": True} in namespace

        Args:
            stdout: Standard output from code execution

        Returns:
            Final answer string or None if not found
        """
        # Pattern 1: RLM-style FINAL(answer). Paren-counting handles nested
        # parens (e.g. FINAL(f(x))) and multi-line values without regex flag trade-offs.
        idx = stdout.find("FINAL(")
        if idx != -1:
            depth, start = 0, idx + len("FINAL")
            for i, ch in enumerate(stdout[idx + len("FINAL"):], start=idx + len("FINAL")):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return stdout[start + 1 : i].strip()

        # Pattern 2: RLM-style FINAL_VAR(variable_name)
        final_var_match = re.search(r"FINAL_VAR\((\w+)\)", stdout)
        if final_var_match and self._executor:
            var_name = final_var_match.group(1)
            value = self._executor.get_variable(var_name)
            if value is not None:
                return str(value)

        # Pattern 3: Prime Intellect style answer dict
        if self._executor:
            answer_var = self._executor.get_variable("answer")
            if isinstance(answer_var, dict):
                if answer_var.get("ready", False):
                    return str(answer_var.get("content", ""))

        return None

    def _create_final_observation(self, success: bool, message: str) -> REPLObservation:
        """Create observation for episode termination.

        Args:
            success: Whether the episode ended successfully
            message: Termination message

        Returns:
            Final REPLObservation with done=True (reward set by rubric)
        """
        return REPLObservation(
            result=CodeBlockResult(
                stdout=message,
                stderr="",
                locals_snapshot={},
                execution_time=0.0,
                success=success,
                exception=None,
            ),
            context_preview=None,
            context_length=0,
            available_variables=[],
            iteration=self._state.iteration if self._state else 0,
            max_iterations=self.max_iterations,
            done=True,
            metadata={
                "final_answer": self._state.final_answer if self._state else None,
                "total_execution_time": (
                    self._state.total_execution_time if self._state else 0
                ),
                "total_iterations": self._state.iteration if self._state else 0,
            },
        )

    @property
    def state(self) -> REPLState:
        """Get the current environment state.

        Returns:
            Current REPLState

        Raises:
            RuntimeError: If environment not initialized
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def close(self) -> None:
        """Cleanup resources."""
        if self._runtime_controller is not None:
            self._runtime_controller.close()
            self._runtime_controller = None
        self._executor = None
        self._state = None

    def get_metadata(self) -> EnvironmentMetadata:
        """Get environment metadata.

        Returns:
            EnvironmentMetadata with environment info
        """
        return EnvironmentMetadata(
            name="repl_env",
            description="Python REPL environment for RLM-style code execution",
            version="0.1.0",
        )
