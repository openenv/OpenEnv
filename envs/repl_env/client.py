# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment clients.

`REPLEnv` is the standard async OpenEnv client for remote/server-backed usage.
Use `async with` / `await` directly, or call `.sync()` for synchronous code.

This module intentionally contains only the remote OpenEnv client.
"""

from __future__ import annotations

from typing import Any

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import CodeBlockResult, REPLAction, REPLObservation, REPLState
except ImportError:
    from models import CodeBlockResult, REPLAction, REPLObservation, REPLState
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class REPLEnv(EnvClient[REPLAction, REPLObservation, REPLState]):
    """
    Async client for the remote REPL environment.

    Use this client when connecting to a running OpenEnv server over WebSocket.
    For synchronous code, call `.sync()` on an instance.

    Example:
        >>> async with REPLEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(context="Hello World", task_prompt="Count chars")
        ...     result = await env.execute("count = len(context)")
        ...     result = await env.execute("print(f'FINAL({count})')")
        ...     print(result.done)

        >>> with REPLEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(context="Hello World", task_prompt="Count chars")
        ...     result = env.execute("count = len(context)")
        ...     result = env.execute("print(f'FINAL({count})')")
        ...     print(result.done)
    """

    def _step_payload(self, action: REPLAction) -> dict[str, Any]:
        return {
            "code": action.code,
            "is_final": action.is_final,
            "final_answer": action.final_answer,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[REPLObservation]:
        obs_data = payload.get("observation", {})
        result_data = obs_data.get("result", {})

        observation = REPLObservation(
            result=CodeBlockResult(
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                locals_snapshot=result_data.get("locals_snapshot", {}),
                execution_time=result_data.get("execution_time", 0.0),
                success=result_data.get("success", True),
                exception=result_data.get("exception"),
            ),
            context_preview=obs_data.get("context_preview"),
            context_length=obs_data.get("context_length", 0),
            available_variables=obs_data.get("available_variables", []),
            iteration=obs_data.get("iteration", 0),
            max_iterations=obs_data.get("max_iterations", 30),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> REPLState:
        return REPLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            context=payload.get("context"),
            task_prompt=payload.get("task_prompt"),
            iteration=payload.get("iteration", 0),
            max_iterations=payload.get("max_iterations", 30),
            namespace_keys=payload.get("namespace_keys", []),
            final_answer=payload.get("final_answer"),
            total_execution_time=payload.get("total_execution_time", 0.0),
        )

    async def execute(self, code: str) -> StepResult[REPLObservation]:
        """Execute Python code in the REPL."""
        return await self.step(REPLAction(code=code))

    async def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]:
        """Submit a final answer and terminate the episode."""
        return await self.step(REPLAction(code="", is_final=True, final_answer=answer))

    async def get_variable(self, name: str) -> StepResult[REPLObservation]:
        """Retrieve and print a variable from the REPL namespace."""
        return await self.execute(f"print(repr({name}))")

    async def list_variables(self) -> list[str]:
        """Return the current REPL namespace keys."""
        return (await self.state()).namespace_keys
