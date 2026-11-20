# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State of the data-analysis environment, shared by the client and the server."""

from typing import List, Optional

from openenv.core.env_server.types import State

# The agent's tools. Defined here so the client and docs don't import the server.
AVAILABLE_TOOLS = ["run_python", "submit_answer"]


class DataAnalysisState(State):
    """
    Episode state. The gold answer is never part of it: it stays in the server process.

    Attributes:
        split: SmolDataEnvs split the task comes from (`train`, `eval` or `test`).
        task_id: SmolDataEnvs task id.
        question: The question to answer.
        files: Absolute paths of the task's data files, as model code sees them.
        difficulty_tier: The dataset's difficulty tier (`easy`, `medium`, `hard`).
        max_python_calls: `run_python` calls allowed in the episode.
        python_calls: `run_python` calls made so far.
        python_errors: Calls that raised, timed out or crashed (`exit_code != 0`).
        last_exit_code: Exit code of the last `run_python` call.
        submitted: Whether `submit_answer` was called (it ends the episode).
        submitted_answer: The submitted answer.
        reward: 1.0 if the answer matches the gold answer under the dataset's grader, else 0.0.
        grade_method: How the grader decided (`exact`, `numeric`, `list`, `math_verify`, `miss`, ...).
    """

    split: Optional[str] = None
    task_id: Optional[str] = None
    question: str = ""
    files: List[str] = []
    difficulty_tier: Optional[str] = None
    max_python_calls: int = 0
    python_calls: int = 0
    python_errors: int = 0
    last_exit_code: int = 0
    submitted: bool = False
    submitted_answer: Optional[str] = None
    reward: Optional[float] = None
    grade_method: Optional[str] = None
