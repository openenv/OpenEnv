# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Recursive backend abstractions for repl_env.

This module keeps direct LM calls and recursive child spawning out of the
runner and environment. The runner owns the iterative loop; the backend owns
query/query_batched/child-recursion behavior.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol


ChatFn = Callable[..., str]


class RecursiveBackend(Protocol):
    max_depth: int
    depth: int
    child_traces: list["ChildTrace"]

    def query(self, prompt: str, model: str | None = None) -> str: ...

    def query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]: ...

    def recursive_query(self, prompt: str, model: str | None = None) -> str: ...

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]: ...


@dataclass
class BackendLimits:
    max_depth: int = 1
    max_batch_workers: int = 8
    max_children_total: int | None = None
    max_children_per_batch: int | None = None
    result_truncation_limit: int | None = None
    # Cooperative timeout: checked between iterations, not during LLM calls.
    # A slow LLM call within an iteration will not be interrupted — the timeout
    # fires at the next iteration boundary. For mid-call cancellation, use
    # process-based isolation instead.
    per_child_timeout_s: float | None = None
    # Tree-global child counter shared across all recursion depths
    _children_spawned: int = field(default=0, init=False, repr=False)
    _children_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )


@dataclass
class ChildTrace:
    depth: int
    duration_s: float
    prompt_preview: str
    result_preview: str | None
    error: str | None


class DirectLMBackend:
    """Direct LM backend with no child recursion beyond fallback to itself."""

    def __init__(
        self,
        llm_chat_fn: ChatFn,
        *,
        depth: int = 0,
        limits: BackendLimits | None = None,
    ) -> None:
        self.llm_chat_fn = llm_chat_fn
        self.depth = depth
        self.limits = limits or BackendLimits()
        self.max_depth = self.limits.max_depth
        self.child_traces: list[ChildTrace] = []

    def query(self, prompt: str, model: str | None = None) -> str:
        try:
            result = self.llm_chat_fn([{"role": "user", "content": prompt}], model)
        except TypeError:
            result = self.llm_chat_fn([{"role": "user", "content": prompt}])
        return self._truncate(result)

    def query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        if not prompts:
            return []
        max_workers = min(len(prompts), self.limits.max_batch_workers)
        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.query, prompt, model): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = f"Error: {exc}"
        return results

    def recursive_query(self, prompt: str, model: str | None = None) -> str:
        return self.query(prompt, model)

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        return self.query_batched(prompts, model)

    def _truncate(self, result: str) -> str:
        limit = self.limits.result_truncation_limit
        if limit is not None and len(result) > limit:
            return result[:limit]
        return result


class LocalChildRLMBackend(DirectLMBackend):
    """Recursive backend that spawns child LocalRLMRunner instances."""

    def __init__(
        self,
        llm_chat_fn: ChatFn,
        *,
        runner_factory: Callable[..., object],
        system_prompt: str,
        max_iterations: int,
        env_max_iterations_multiplier: int,
        depth: int = 0,
        limits: BackendLimits | None = None,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None]
        | None = None,
    ) -> None:
        super().__init__(llm_chat_fn, depth=depth, limits=limits)
        self.runner_factory = runner_factory
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.env_max_iterations_multiplier = env_max_iterations_multiplier
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete

    def recursive_query(self, prompt: str, model: str | None = None) -> str:
        next_depth = self.depth + 1
        if next_depth >= self.max_depth:
            return self.query(prompt, model)
        with self.limits._children_lock:
            if self.limits.max_children_total is not None:
                if self.limits._children_spawned >= self.limits.max_children_total:
                    return "Error: max_children_total exceeded"
            self.limits._children_spawned += 1
        start = time.perf_counter()
        error: str | None = None
        result_text = ""
        resolved_model = model or "default"
        if self.on_subcall_start is not None:
            try:
                self.on_subcall_start(next_depth, str(resolved_model), prompt[:80])
            except Exception:
                pass
        try:
            child = self.runner_factory(
                self.llm_chat_fn,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations,
                max_depth=self.max_depth,
                depth=next_depth,
                env_max_iterations_multiplier=self.env_max_iterations_multiplier,
                max_batch_workers=self.limits.max_batch_workers,
                backend_factory=self._child_backend_factory,
                on_subcall_start=self.on_subcall_start,
                on_subcall_complete=self.on_subcall_complete,
            )
            result = child.run(
                prompt, prompt, model=model, timeout_s=self.limits.per_child_timeout_s
            )
            result_text = self._truncate(result.final_answer or "")
            return result_text
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration = time.perf_counter() - start
            self.child_traces.append(
                ChildTrace(
                    depth=next_depth,
                    duration_s=duration,
                    prompt_preview=prompt[:80],
                    result_preview=(result_text[:80] if result_text else None),
                    error=error,
                )
            )
            if self.on_subcall_complete is not None:
                try:
                    self.on_subcall_complete(
                        next_depth,
                        str(resolved_model),
                        duration,
                        error,
                    )
                except Exception:
                    pass

    def recursive_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        if not prompts:
            return []
        batch_limit = self.limits.max_children_per_batch
        if batch_limit is not None:
            prompts = prompts[:batch_limit]
        max_workers = min(len(prompts), self.limits.max_batch_workers)
        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.recursive_query, prompt, model): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = f"Error: {exc}"
        return results

    def _child_backend_factory(
        self, llm_chat_fn: ChatFn, **kwargs
    ) -> "LocalChildRLMBackend":
        return LocalChildRLMBackend(
            llm_chat_fn,
            runner_factory=self.runner_factory,
            system_prompt=kwargs["system_prompt"],
            max_iterations=kwargs["max_iterations"],
            env_max_iterations_multiplier=kwargs["env_max_iterations_multiplier"],
            depth=kwargs["depth"],
            limits=self.limits,
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
        )
