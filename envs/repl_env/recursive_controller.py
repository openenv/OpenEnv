# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Controller composition for recursive backends.

This keeps server-side recursion assembly outside `REPLEnvironment`:
- backend selection based on max_depth
- limits configuration
- uniform callable interface for the environment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .prompts import RLM_SYSTEM_PROMPT
from .recursive_backends import BackendLimits, DirectLMBackend, LocalChildRLMBackend


ChatFn = Callable[..., str]


@dataclass
class RecursiveController:
    llm_query_fn: Callable[[str, str | None], str]
    llm_batch_fn: Callable[[list[str], str | None], list[str]]
    rlm_query_fn: Callable[[str, str | None], str] | None
    rlm_batch_fn: Callable[[list[str], str | None], list[str]] | None
    backend: object

    def close(self) -> None:
        pass


def create_server_recursive_controller(
    chat_fn: ChatFn,
    *,
    max_depth: int,
    max_iterations: int,
    system_prompt: str = RLM_SYSTEM_PROMPT,
    max_batch_workers: int = 8,
    max_children_total: int | None = None,
    max_children_per_batch: int | None = None,
    result_truncation_limit: int | None = None,
    per_child_timeout_s: float | None = None,
    env_max_iterations_multiplier: int = 5,
) -> RecursiveController:
    limits = BackendLimits(
        max_depth=max_depth,
        max_batch_workers=max_batch_workers,
        max_children_total=max_children_total,
        max_children_per_batch=max_children_per_batch,
        result_truncation_limit=result_truncation_limit,
        per_child_timeout_s=per_child_timeout_s,
    )
    if max_depth > 1:
        from .runner import LocalRLMRunner

        backend = LocalChildRLMBackend(
            chat_fn,
            runner_factory=LocalRLMRunner,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            env_max_iterations_multiplier=env_max_iterations_multiplier,
            depth=0,
            limits=limits,
        )
        return RecursiveController(
            llm_query_fn=backend.query,
            llm_batch_fn=backend.query_batched,
            rlm_query_fn=backend.recursive_query,
            rlm_batch_fn=backend.recursive_query_batched,
            backend=backend,
        )

    backend = DirectLMBackend(chat_fn, depth=0, limits=limits)
    return RecursiveController(
        llm_query_fn=backend.query,
        llm_batch_fn=backend.query_batched,
        rlm_query_fn=None,
        rlm_batch_fn=None,
        backend=backend,
    )
