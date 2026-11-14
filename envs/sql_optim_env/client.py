# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client for the SQL Query Optimization environment.

Maintains a persistent WebSocket connection to the environment server for
efficient multi-step episodes. Async by default; use `.sync()` for a
synchronous wrapper.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SQLOptimAction, SQLOptimObservation, SQLOptimState


class SQLOptimEnv(EnvClient[SQLOptimAction, SQLOptimObservation, SQLOptimState]):
    """
    Client for the SQL Query Optimization environment.

    The agent receives a slow SQL query plus its schema, returns a
    [`SQLOptimAction`][envs.sql_optim_env.models.SQLOptimAction] with a
    rewritten `optimized_query`, and is rewarded by real DuckDB execution
    speedup and result-correctness.

    Examples:

    ```python
    with SQLOptimEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset(task_id="task_1_basic_antipatterns").observation
        result = env.step(
            SQLOptimAction(
                optimized_query="SELECT id FROM orders WHERE customer_id = 5000",
                summary="Dropped SELECT * and the non-sargable CAST.",
            )
        )
        print(result.reward, result.done)
    ```
    """

    def _step_payload(self, action: SQLOptimAction) -> Dict[str, Any]:
        """Serialize a [`SQLOptimAction`] into the step request payload."""
        return {
            "suggestions": action.suggestions,
            "optimized_query": action.optimized_query,
            "summary": action.summary,
            "estimated_improvement": action.estimated_improvement,
            "approved": action.approved,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLOptimObservation]:
        """Parse a server step/reset response into a `StepResult`."""
        obs_data = payload.get("observation", {})
        # `serialize_observation` keeps reward and done on the envelope rather
        # than in the observation dict, so mirror them back onto the
        # observation. Without this `result.observation.reward` stayed None and
        # `result.observation.done` False while `StepResult` carried the right
        # values, trapping callers that read the observation instead.
        reward = payload.get("reward", obs_data.get("reward"))
        done = bool(payload.get("done", obs_data.get("done", False)))
        observation = SQLOptimObservation(
            **{**obs_data, "reward": reward, "done": done}
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            metadata=payload.get("metadata", observation.metadata),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SQLOptimState:
        """Parse a server state response into a [`SQLOptimState`]."""
        return SQLOptimState(**payload)
