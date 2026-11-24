# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Server-internal scoring types for the SQL Query Optimization environment.

`Reward` is produced by [`grade`][envs.sql_optim_env.server.graders.grade] and
consumed by the environment, which folds it into the observation's `reward`
field and `metadata`. It is not a wire type and never leaves the server.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """
    Composite, execution-grounded reward for one optimization step.

    Attributes:
        score (`float`):
            Composite reward in `[0.0, 1.0]`.
        breakdown (`dict[str, float]`):
            Per-criterion contribution to `score` (execution speedup, result
            correctness, issue detection, approval, summary, severity).
        feedback (`str`):
            Human-readable explanation, including real DuckDB timings.
        execution (`dict`, *optional*):
            The DuckDB `compare()` result this reward was scored from (timings,
            speedup, correctness), or `None` when no query was executed. Carried
            here so the environment can surface it without re-running the queries.
    """

    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    execution: Optional[Dict[str, Any]] = None
