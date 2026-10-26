# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local recursive RLM runner for repl_env.

This keeps the iterative prompting/orchestration layer outside the environment,
following the same separation used by the official RLM implementation and DSPy:
- `REPLEnvironment` executes code and exposes tools
- `LocalRLMRunner` owns prompting, message history, and recursive child runs
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from .local import LocalREPLEnv
from .prompts import (
    build_rlm_system_prompt,
    build_user_prompt,
    extract_code_blocks,
    format_observations,
    QueryMetadata,
    RLM_SYSTEM_PROMPT,
)
from .recursive_backends import BackendLimits, LocalChildRLMBackend, RecursiveBackend


ChatFn = Callable[..., str]


@dataclass
class RLMRunResult:
    final_answer: str | None
    messages: list[dict[str, str]]
    iterations: int
    depth: int
    child_traces: list[object]


class LocalRLMRunner:
    """Local recursive RLM orchestrator built on top of LocalREPLEnv."""

    def __init__(
        self,
        llm_chat_fn: ChatFn,
        *,
        system_prompt: str = RLM_SYSTEM_PROMPT,
        max_iterations: int = 30,
        max_depth: int = 2,
        depth: int = 0,
        env_max_iterations_multiplier: int = 5,
        max_batch_workers: int = 8,
        backend_factory: Callable[..., RecursiveBackend] | None = None,
        max_children_total: int | None = None,
        max_children_per_batch: int | None = None,
        result_truncation_limit: int | None = None,
        per_child_timeout_s: float | None = None,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None]
        | None = None,
        verbose: bool = False,
    ) -> None:
        self.llm_chat_fn = llm_chat_fn
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.depth = depth
        self.env_max_iterations_multiplier = env_max_iterations_multiplier
        self.max_batch_workers = max_batch_workers
        self.backend_factory = backend_factory or self._default_backend_factory
        self.max_children_total = max_children_total
        self.max_children_per_batch = max_children_per_batch
        self.result_truncation_limit = result_truncation_limit
        self.per_child_timeout_s = per_child_timeout_s
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete
        self.verbose = verbose

    def _default_backend_factory(
        self, llm_chat_fn: ChatFn, **kwargs
    ) -> RecursiveBackend:
        limits = BackendLimits(
            max_depth=self.max_depth,
            max_batch_workers=self.max_batch_workers,
            max_children_total=self.max_children_total,
            max_children_per_batch=self.max_children_per_batch,
            result_truncation_limit=self.result_truncation_limit,
            per_child_timeout_s=self.per_child_timeout_s,
        )
        return LocalChildRLMBackend(
            llm_chat_fn,
            runner_factory=LocalRLMRunner,
            system_prompt=kwargs["system_prompt"],
            max_iterations=kwargs["max_iterations"],
            env_max_iterations_multiplier=kwargs["env_max_iterations_multiplier"],
            depth=kwargs["depth"],
            limits=limits,
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
        )

    def run(
        self,
        context: str,
        task_prompt: str,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> RLMRunResult:
        backend = self.backend_factory(
            self.llm_chat_fn,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            depth=self.depth,
            env_max_iterations_multiplier=self.env_max_iterations_multiplier,
        )
        with LocalREPLEnv(
            llm_query_fn=backend.query,
            llm_batch_fn=backend.query_batched,
            subcall_fn=backend.recursive_query,
            subcall_batch_fn=backend.recursive_query_batched,
        ) as env:
            result = env.reset(
                context=context,
                task_prompt=task_prompt,
                max_iterations=self.max_iterations * self.env_max_iterations_multiplier,
                llm_model=model,
            )
            obs = result.observation

            query_metadata = QueryMetadata(
                context_lengths=[obs.context_length],
                context_total_length=obs.context_length,
                context_type="str",
            )
            messages = build_rlm_system_prompt(self.system_prompt, query_metadata)
            messages.append(build_user_prompt(root_prompt=task_prompt, iteration=0))

            run_start = time.perf_counter()

            for iteration in range(1, self.max_iterations + 1):
                # Cooperative timeout check (matches official RLM pattern)
                if timeout_s is not None:
                    elapsed = time.perf_counter() - run_start
                    if elapsed >= timeout_s:
                        return RLMRunResult(
                            final_answer=f"Error: child timeout after {elapsed:.3f}s",
                            messages=messages,
                            iterations=iteration - 1,
                            depth=self.depth,
                            child_traces=list(getattr(backend, "child_traces", [])),
                        )

                response = self._chat(messages, model)
                code_blocks = extract_code_blocks(response)
                code_block_observations = []

                if self.verbose:
                    print(
                        f"[depth={self.depth}] iteration={iteration} code_blocks={len(code_blocks)}"
                    )

                if not code_blocks:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please continue by writing Python code in ```repl``` blocks, "
                                "or submit the final answer with FINAL(...) / FINAL_VAR(...)."
                            ),
                        }
                    )
                    continue

                for code in code_blocks:
                    result = env.execute(code)
                    code_block_observations.append(result.observation)

                # Check for FINAL after all blocks executed (matches official RLM).
                # The model expects all blocks to run — it often writes exploration
                # code first and FINAL last in the same response.
                if any(obs.done for obs in code_block_observations):
                    return RLMRunResult(
                        final_answer=env.state().final_answer,
                        messages=messages
                        + [{"role": "assistant", "content": response}],
                        iterations=iteration,
                        depth=self.depth,
                        child_traces=list(getattr(backend, "child_traces", [])),
                    )

                observation_text = format_observations(
                    code_block_observations, code_blocks=code_blocks
                )
                next_prompt = build_user_prompt(
                    root_prompt=task_prompt,
                    iteration=iteration,
                )
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": observation_text + "\n\n" + next_prompt["content"],
                    }
                )

            # Max iterations exhausted — give the model one final chance to answer
            final_answer = env.state().final_answer
            if final_answer is None:
                final_answer = self._default_answer(messages, model)

            return RLMRunResult(
                final_answer=final_answer,
                messages=messages,
                iterations=self.max_iterations,
                depth=self.depth,
                child_traces=list(getattr(backend, "child_traces", [])),
            )

    def _default_answer(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str | None:
        """Make one final LLM call asking for an answer when iterations are exhausted."""
        final_prompt = messages + [
            {
                "role": "user",
                "content": (
                    "You have run out of REPL iterations. Based on all your work above, "
                    "provide your best final answer now. Use FINAL(your answer) to submit it. "
                    "If you stored the answer in a variable, use FINAL_VAR(variable_name) instead. "
                    "Do not write any more code — just provide the final answer."
                ),
            }
        ]
        try:
            response = self._chat(final_prompt, model)
            # Try to extract FINAL(...) from the response. Paren-counting handles
            # nested parens and mid-sentence FINAL without regex flag trade-offs.
            idx = response.find("FINAL(")
            if idx != -1:
                depth, start = 0, idx + len("FINAL")
                for i, ch in enumerate(response[idx + len("FINAL"):], start=idx + len("FINAL")):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            return response[start + 1 : i].strip()
            # If no FINAL pattern, return the raw response as best-effort
            return response.strip() if response.strip() else None
        except Exception:
            return None

    def _chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        try:
            return self.llm_chat_fn(messages, model)
        except TypeError:
            return self.llm_chat_fn(messages)
