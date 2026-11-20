# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data-analysis Environment Implementation.

Each episode is one SmolDataEnvs task: a question about one or more data files. The agent has two
MCP tools: `run_python(code)`, which runs code in the episode's sandboxed Python session (see
sandbox.py), and `submit_answer(answer)`, which grades the answer with the dataset's grader and
ends the episode. The reward (1.0 or 0.0) arrives on the observation of the submitting step.
"""

import asyncio
import random
import threading
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

from ..models import AVAILABLE_TOOLS, DataAnalysisState
from . import tasks
from .grading import grade_answer
from .sandbox import MAX_START_S, SandboxSession, settings as sandbox_settings

DEFAULT_MAX_PYTHON_CALLS = 30
MAX_PYTHON_CALLS_LIMIT = 200
MAX_TOOL_OUTPUT_CHARS = 20_000
RANDOM_PICK_ATTEMPTS = 8
# Added to the sandbox's own limits (start, then one call) for the MCP tool timeout.
TOOL_TIMEOUT_MARGIN_S = 10.0


class DataAnalysisEnvironment(MCPEnvironment):
    """MCP environment for data analysis, with tools `run_python` and `submit_answer`."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("data_analysis_env")

        # The tools wait on a worker process and on downloads, so they run in threads to keep
        # the server loop (shared by all sessions) responsive.
        @mcp.tool
        async def run_python(code: str) -> str:
            """
            Run Python code in this task's live session and return its output.

            Variables persist between calls, and the value of a last-line expression is
            printed, as in a notebook. pandas, numpy and scikit-learn are available, and only the task's files
            can be read.

            Args:
                code: The Python source to run.

            Returns:
                stdout, then stderr (errors and warnings).
            """
            return await asyncio.to_thread(self._run_python, code)

        @mcp.tool
        async def submit_answer(answer: str) -> str:
            """
            Submit the final answer. Only one answer is accepted, and it ends the task.

            Args:
                answer: The answer, short and clean: a number, a label, yes/no or a
                    comma-separated list.

            Returns:
                A confirmation.
            """
            return await asyncio.to_thread(self._submit_answer, answer)

        super().__init__(mcp)
        self._state = DataAnalysisState(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[dict] = (
            None  # the row, with its gold answer: never sent out
        )
        self._session: Optional[SandboxSession] = None
        self._rng = random.Random()
        # A tool call that the MCP layer gave up on keeps running in its thread. The lock makes
        # the next call (or reset, submit, close) wait for it instead of sharing the worker.
        self._lock = threading.Lock()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        split: Optional[str] = None,
        task_id: Optional[str] = None,
        max_python_calls: Optional[int] = None,
    ) -> Observation:
        """
        Start an episode on one task and return its prompt.

        Args:
            seed: Reseed the task choice. The same seed picks the same task, and the same
                sequence of tasks on the unseeded resets that follow.
            episode_id: Episode identifier (generated if omitted).
            split: SmolDataEnvs split: `train`, `eval`, `test`, `eval_verified` or `test_verified`
                (default `DATA_ANALYSIS_SPLIT` or `eval`).
            task_id: A specific task of that split. Random (among tasks with CSV/TSV/JSON files
                under the size cap) if omitted.
            max_python_calls: `run_python` calls allowed (default 30, at most 200).

        Returns:
            An observation whose metadata holds `prompt`, `task_id`, `split`, `question`,
            `files` (absolute paths) and `difficulty_tier`. Never the answer.
        """
        conf = tasks.settings()
        split = str(split or conf["split"])
        calls = DEFAULT_MAX_PYTHON_CALLS if max_python_calls is None else int(max_python_calls)
        if not 1 <= calls <= MAX_PYTHON_CALLS_LIMIT:
            raise ValueError(
                f"max_python_calls must be in [1, {MAX_PYTHON_CALLS_LIMIT}]"
            )
        if seed is not None:
            self._rng = random.Random(seed)
        # The new task is fetched before the current episode is touched, so a failed reset
        # leaves that episode as it was.
        rows = tasks.load_split(split)
        attempts = 1 if task_id is not None else RANDOM_PICK_ATTEMPTS
        for attempt in range(attempts):
            row = tasks.pick_task(rows, self._rng, task_id)
            try:
                task_dir, paths = tasks.fetch_task_files(
                    row, split, str(conf["tasks_root"]), int(conf["max_file_bytes"])
                )
                break
            except tasks.TaskError:
                if attempt == attempts - 1:
                    raise
                # Don't draw the same task again on the next attempt.
                rows = [r for r in rows if r["task_id"] != row["task_id"]]
        with self._lock:
            self._close_session()
            self._task = row
            self._session = SandboxSession(read_dirs=[task_dir])
            self._state = DataAnalysisState(
                episode_id=episode_id or str(uuid4()),
                step_count=0,
                split=split,
                task_id=row["task_id"],
                question=str(row["question"]),
                files=paths,
                difficulty_tier=(
                    str(row["difficulty_tier"]) if row.get("difficulty_tier") else None
                ),
                max_python_calls=calls,
            )
        return Observation(
            done=False,
            reward=None,
            metadata={
                "prompt": tasks.build_prompt(row, paths),
                "task_id": self._state.task_id,
                "split": split,
                "question": self._state.question,
                "files": paths,
                "difficulty_tier": self._state.difficulty_tier,
                "tools": list(AVAILABLE_TOOLS),
            },
        )

    def _run_python(self, code: str) -> str:
        with self._lock:
            return self._run_python_locked(code)

    def _run_python_locked(self, code: str) -> str:
        if self._state.submitted:
            return "Error: an answer was already submitted. The episode is over."
        if self._task is None or self._session is None:
            return "Error: no task. The episode has not started."
        if self._state.python_calls >= self._state.max_python_calls:
            return "Error: no run_python calls left. Submit your answer now."
        stdout, stderr, exit_code = self._session.run(code)
        self._state.python_calls += 1
        self._state.last_exit_code = exit_code
        self._state.python_errors += int(exit_code != 0)
        return tasks.tool_output(stdout, stderr, exit_code, MAX_TOOL_OUTPUT_CHARS)

    def _submit_answer(self, answer: str) -> str:
        with self._lock:
            return self._submit_answer_locked(answer)

    def _submit_answer_locked(self, answer: str) -> str:
        if self._task is None:
            return "Error: no task. The episode has not started."
        if self._state.submitted:
            return "Error: an answer was already submitted. The episode is over."
        reward, method = grade_answer(self._task, answer)
        self._state.submitted = True
        self._state.submitted_answer = str(answer)[:1000]
        self._state.reward = reward
        self._state.grade_method = method
        self._close_session()  # no more code runs in this episode
        return "Answer submitted. The task is over."

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _finish(self, obs: Observation) -> Observation:
        if self._state.submitted:
            obs.done = True
            obs.reward = self._state.reward
        return obs

    def _step_impl(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    @staticmethod
    def _tool_timeout_s(timeout_s: Optional[float]) -> float:
        """The MCP tool timeout: by default longer than a cold sandbox start plus one call."""
        if timeout_s is not None:
            return timeout_s
        exec_timeout_s = float(sandbox_settings()["exec_timeout_s"])
        return MAX_START_S + exec_timeout_s + TOOL_TIMEOUT_MARGIN_S

    def step(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        self._state.step_count += 1
        obs = super().step(action, timeout_s=self._tool_timeout_s(timeout_s), **kwargs)
        return self._finish(obs)

    async def step_async(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(
            action, timeout_s=self._tool_timeout_s(timeout_s), **kwargs
        )
        return self._finish(obs)

    @property
    def state(self) -> DataAnalysisState:
        return self._state

    def close(self) -> None:
        """Called by the server when the session ends: stop the worker process."""
        with self._lock:
            self._close_session()
        super().close()
