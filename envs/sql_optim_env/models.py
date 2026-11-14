# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SQL Query Optimization environment.

Defines the `Action`, `Observation`, and `State` wire types exchanged between
the [`SQLOptimEnv`][envs.sql_optim_env.client.SQLOptimEnv] client and the
environment server. All three inherit from the OpenEnv base types so they
serialize over the WebSocket/HTTP transport.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class SQLOptimAction(Action):
    """
    Action emitted by the agent for one optimization step.

    Attributes:
        suggestions (`list[dict]`):
            Detected issues. Each entry is a dict with `issue_type`, `line`,
            `description`, `severity`, and `fix` keys.
        optimized_query (`str`):
            The complete rewritten SQL. It is executed against the real DuckDB
            data so the measured speedup can be rewarded.
        summary (`str`):
            Overall analysis and performance profile of the query.
        estimated_improvement (`str`):
            The agent's own expected speedup (e.g. `"10x faster"`).
        approved (`bool`):
            `True` if the agent judges the query already optimal, `False` if it
            still needs changes.
    """

    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    optimized_query: str = ""
    summary: str = ""
    estimated_improvement: str = ""
    approved: bool = False


class SQLOptimObservation(Observation):
    """
    Observation returned to the agent after `reset` and each `step`.

    Inherits `done`, `reward`, and `metadata` from
    [`~openenv.core.env_server.Observation`]. The serializer carries `reward`
    and `done` on the response envelope rather than in the observation payload;
    the typed client mirrors them back, so `observation.reward` and
    `StepResult.reward` agree after a step. `reset` leaves `reward` as `None`,
    since nothing has been scored yet. The composite reward breakdown,
    per-criterion feedback, and full execution comparison are attached under
    `metadata` (keys `reward_breakdown`, `feedback`, `execution`).

    Attributes:
        task_id (`str`):
            Identifier of the active task.
        task_name (`str`):
            Human-readable task name.
        task_description (`str`):
            What the agent must do for this task.
        sql_query (`str`):
            The slow SQL query to analyze and rewrite.
        schema_info (`str`):
            Table schema, row counts, and index notes.
        dialect (`str`):
            SQL dialect the query targets.
        difficulty (`str`):
            One of `easy`, `medium`, `hard`, `expert`.
        step_count (`int`):
            Steps taken so far in this episode.
        max_steps (`int`):
            Maximum steps allowed for this task.
        issues_found_so_far (`list[str]`):
            Issue types the agent flagged in previous steps.
        last_execution (`dict`, *optional*):
            Execution comparison from the previous step, so the agent can refine
            its `optimized_query`.
    """

    task_id: str = ""
    task_name: str = ""
    task_description: str = ""
    sql_query: str = ""
    schema_info: str = ""
    dialect: str = "duckdb/postgresql"
    difficulty: str = ""
    step_count: int = 0
    max_steps: int = 5
    issues_found_so_far: List[str] = Field(default_factory=list)
    last_execution: Optional[Dict[str, Any]] = None


class SQLOptimState(State):
    """
    Internal environment state for the current episode.

    Inherits `episode_id` and `step_count` from
    [`~openenv.core.env_server.State`].

    Attributes:
        task_id (`str`):
            Identifier of the active task (`"none"` when no episode is active).
        max_steps (`int`):
            Maximum steps allowed for the active task.
        episode_done (`bool`):
            Whether the current episode has terminated.
        cumulative_reward (`float`):
            Sum of per-step rewards in this episode.
        current_task (`str`):
            Human-readable name of the active task.
    """

    task_id: str = "none"
    max_steps: int = 0
    episode_done: bool = True
    cumulative_reward: float = 0.0
    current_task: str = "No active episode"
