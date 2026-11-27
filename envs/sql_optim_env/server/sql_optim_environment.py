# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Server-side environment for SQL query optimization.

The agent receives a slow SQL query plus schema context and returns an
optimized rewrite. The environment executes both the original and optimized
queries against a real in-memory DuckDB database, measures the actual speedup
and result-correctness, and grades the result — reward lives entirely inside
the environment, per OpenEnv's design.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

# Support both in-repo and standalone imports
try:
    # In-repo imports (running from the OpenEnv repository)
    from ..models import SQLOptimAction, SQLOptimObservation, SQLOptimState
    from .graders import grade
    from .tasks import TASKS
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (running via uvicorn server.app:app)
    from models import SQLOptimAction, SQLOptimObservation, SQLOptimState
    from server.graders import grade
    from server.tasks import TASKS

DEFAULT_TASK_ID = "task_1_basic_antipatterns"


class SQLOptimEnvironment(Environment):
    """
    Execution-grounded SQL optimization environment.

    Multi-step episodes: `issues_found_so_far` accumulates the issue types the
    agent has flagged, and `last_execution` carries the real DuckDB timing
    comparison back into the observation so the agent can refine its rewrite on
    later steps. An episode ends after the task's `max_steps` or once a step
    scores `>= 0.95`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._task_data: Optional[dict] = None
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._issues_found: list[str] = []
        self._last_execution: Optional[dict] = None
        self._episode_id: str = ""
        self._state: SQLOptimState = SQLOptimState()
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SQLOptimObservation:
        """
        Start a new episode on the requested task.

        Args:
            seed (`int`, *optional*):
                Accepted for Gym API compatibility; the task data is
                deterministic, so it does not affect the episode.
            episode_id (`str`, *optional*):
                Custom episode identifier. A UUID is generated when omitted.
            task_id (`str`, *optional*, defaults to `"task_1_basic_antipatterns"`):
                Which task to load. Must be a key of `TASKS`.

        Returns:
            `SQLOptimObservation`: The initial observation for the task.
        """
        task_id = task_id or DEFAULT_TASK_ID
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}"
            )
        self._task_data = TASKS[task_id]
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._issues_found = []
        self._last_execution = None
        self._episode_id = episode_id or str(uuid.uuid4())
        self._sync_state()
        return self._make_obs()

    def step(self, action: SQLOptimAction) -> SQLOptimObservation:
        """
        Grade one optimization attempt and advance the episode.

        Args:
            action (`SQLOptimAction`):
                The agent's suggestions and rewritten `optimized_query`.

        Returns:
            `SQLOptimObservation`: The next observation, with the step reward on
            `reward`, the composite breakdown and execution comparison under
            `metadata`, and `done` set when the episode terminates.
        """
        if self._task_data is None:
            raise RuntimeError("No active episode - call reset() first.")
        if self._done:
            raise RuntimeError("Episode finished - call reset() to start a new one.")

        self._step_count += 1

        # Grade (runs DuckDB internally). Reuse the comparison it already ran
        # rather than executing both queries a second time. `execution` is
        # `None` when the rewrite was empty or failed, so a skipped step never
        # reports the previous step's stale comparison.
        reward = grade(self._task_data, action)
        self._cumulative_reward += reward.score
        self._last_execution = reward.execution

        # Track flagged issue types for progressive context.
        for suggestion in action.suggestions:
            itype = suggestion.get("issue_type", "")
            if itype and itype not in self._issues_found:
                self._issues_found.append(itype)

        max_steps = self._task_data["max_steps"]
        self._done = self._step_count >= max_steps or reward.score >= 0.95
        self._sync_state()

        return self._make_obs(
            reward=reward.score,
            done=self._done,
            metadata={
                "reward_breakdown": reward.breakdown,
                "feedback": reward.feedback,
                "execution": self._last_execution,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "issues_found": len(self._issues_found),
            },
        )

    @property
    def state(self) -> SQLOptimState:
        """Current [`SQLOptimState`] for this episode."""
        return self._state

    # ── Internal ──────────────────────────────────────────────────────────

    def _make_obs(
        self,
        reward: Optional[float] = None,
        done: bool = False,
        metadata: Optional[dict] = None,
    ) -> SQLOptimObservation:
        d = self._task_data
        return SQLOptimObservation(
            task_id=d["task_id"],
            task_name=d["task_name"],
            task_description=d["task_description"],
            sql_query=d["sql_query"],
            schema_info=d["schema_info"],
            dialect=d.get("dialect", "duckdb/postgresql"),
            difficulty=d["difficulty"],
            step_count=self._step_count,
            max_steps=d["max_steps"],
            issues_found_so_far=list(self._issues_found),
            last_execution=self._last_execution,
            reward=reward,
            done=done,
            metadata=metadata or {},
        )

    def _sync_state(self) -> None:
        if self._task_data is None:
            self._state = SQLOptimState(episode_id=self._episode_id)
            return
        self._state = SQLOptimState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_data["task_id"],
            max_steps=self._task_data["max_steps"],
            episode_done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            current_task=self._task_data["task_name"],
        )
