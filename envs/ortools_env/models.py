# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State of the OR-Tools environment, shared by the client and the server."""

from typing import Optional

from openenv.core.env_server.types import State


class ORToolsState(State):
    """Episode state: the task, the sandbox budget, and the outcome once submitted."""

    problem: Optional[str] = None
    size: Optional[str] = None
    level: int = 1
    instance_seed: Optional[int] = None
    reference: Optional[float] = None
    baseline: Optional[float] = None
    lower_bound: Optional[float] = None
    eval_seeds: Optional[list[int]] = None
    eval_references: Optional[list[Optional[float]]] = None
    eval_baselines: Optional[list[Optional[float]]] = None
    allow_ortools: bool = False
    compute_budget_s: float = 0.0
    compute_used_s: float = 0.0
    max_python_calls: int = 0
    python_calls: int = 0
    python_timeout_s: float = 0.0
    solve_time_limit_s: float = 0.0
    reference_time_limit_s: float = 0.0
    submitted: bool = False
    objective: Optional[float] = None
    reward: Optional[float] = None
