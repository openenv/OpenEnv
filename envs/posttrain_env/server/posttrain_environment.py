# SPDX-License-Identifier: BSD-3-Clause

"""Post-training environment: the agent trains models through brokered HF Jobs.

The agent never holds a credential. It submits training scripts through MCP
tools; the server's broker starts them on HF Jobs with its own token, reserves
each job's worst-case cost against the episode budget before it starts, and
settles the real cost when it ends. `dev_evaluate` scores a checkpoint on a
public dev split. `submit` runs the grader-only evaluation on a hidden split,
in its own job, and ends the episode with that score as the reward.

Server configuration (environment variables):

- `HF_TOKEN`: the broker's token. A fine-grained token with only "Start and
  manage Jobs" plus write access to the namespace's `jobs-artifacts` bucket.
- `POSTTRAIN_NAMESPACE`: namespace jobs run in, defaults to the token's owner.
- `POSTTRAIN_RESOURCE_GROUP_ID`: optional Resource Group to bill, so its Jobs
  spend limit caps what the server can spend in total.
- `POSTTRAIN_TEST_SEED`: seed of the hidden test split. Random per process if
  unset, which keeps it unguessable but makes scores comparable only within
  one server process.
"""

import logging
import os
import secrets
import time
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation, State

from .broker import Broker, BrokerError, TERMINAL_STATES
from .tasks import Task, TASKS

logger = logging.getLogger(__name__)

TEST_SEED = int(os.environ.get("POSTTRAIN_TEST_SEED") or secrets.randbelow(2**31))
TOOL_TIMEOUT_S = 120.0
MAX_LOG_CHARS = 20_000


class PostTrainEnvironment(MCPEnvironment):
    """
    MCP environment where the agent post-trains a model on HF Jobs.

    Args:
        api (`huggingface_hub.HfApi`, *optional*):
            HF client holding the broker's token. Defaults to `HfApi()`, which
            reads `HF_TOKEN`.
        task_id (`str`, *optional*, defaults to `"t0_arithmetic_format"`):
            Task used when `reset` is called without `task_id`.
        max_steps (`int`, *optional*, defaults to `100`):
            Tool calls before the episode ends with reward 0.
        max_wait_s (`int`, *optional*, defaults to `300`):
            Longest single `jobs_wait` call.
        poll_s (`float`, *optional*, defaults to `10.0`):
            Interval between job status polls.
        cancel_timeout_s (`float`, *optional*, defaults to `120.0`):
            How long to wait for HF Jobs to confirm cancellations at episode end.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        api: Any | None = None,
        task_id: str = "t0_arithmetic_format",
        max_steps: int = 100,
        max_wait_s: int = 300,
        poll_s: float = 10.0,
        cancel_timeout_s: float = 120.0,
    ):
        mcp = FastMCP("posttrain_env")
        self._api = api if api is not None else HfApi()
        self._default_task_id = task_id
        self._max_steps = max_steps
        self._max_wait_s = max_wait_s
        self._poll_s = poll_s
        self._cancel_timeout_s = cancel_timeout_s
        self._namespace: str | None = None
        self._task: Task | None = None
        self._broker: Broker | None = None
        self._nonce = ""
        self._started_at = 0.0
        self._done = False
        self._reward: float | None = None
        self._score_job_id: str | None = None
        self._end: dict = {}

        @mcp.tool
        def budget() -> dict:
            """
            Show the remaining budget, the hardware you may use with its price,
            and how many jobs and dev evaluations are left.
            """
            return self._call(self._budget)

        @mcp.tool
        def jobs_submit(
            script: str,
            flavor: str,
            timeout_minutes: int,
            dependencies: list[str] | None = None,
            idempotency_key: str | None = None,
        ) -> dict:
            """
            Run a Python script as an HF Job, like `hf jobs uv run`.

            The job's worst-case cost (price per minute times `timeout_minutes`)
            is reserved before it starts and must fit the remaining budget. The
            job has no credentials. Write outputs to `/data/outputs/`.

            Args:
                script: Python source of the job. A PEP 723 header may list dependencies.
                flavor: Hardware flavor, one of those listed by `budget`.
                timeout_minutes: Job timeout; the job is stopped when it is reached.
                dependencies: Extra packages, passed to `uv run --with`.
                idempotency_key: Retrying with the same key returns the same job.
            """
            return self._call(
                lambda: self._job_view(
                    self._broker.submit(
                        script,
                        flavor=flavor,
                        timeout_minutes=timeout_minutes,
                        allowed_flavors=self._task.flavors,
                        max_timeout_minutes=self._task.max_timeout_minutes,
                        dependencies=dependencies,
                        idempotency_key=idempotency_key,
                    )
                )
            )

        @mcp.tool
        def jobs_status(job_id: str) -> dict:
            """
            Get a job's state. Finished agent jobs list their artifact files;
            finished evaluations include their result.

            Args:
                job_id: Id returned by `jobs_submit` or `dev_evaluate`.
            """
            return self._call(
                lambda: self._job_view(self._broker.refresh(self._broker.get(job_id)))
            )

        @mcp.tool
        def jobs_wait(job_id: str, timeout_s: int = 60) -> dict:
            """
            Wait until a job finishes, or up to `timeout_s` seconds (at most 300).

            Args:
                job_id: Id returned by `jobs_submit` or `dev_evaluate`.
                timeout_s: Seconds to wait before returning the current state.
            """
            wait_s = min(max(timeout_s, 0), self._max_wait_s)
            return self._call(
                lambda: self._job_view(
                    self._broker.wait(self._broker.get(job_id), wait_s, self._poll_s)
                )
            )

        @mcp.tool
        def jobs_logs(job_id: str, tail: int = 100) -> dict:
            """
            Read the last lines of one of your jobs' logs.

            Args:
                job_id: Id returned by `jobs_submit`.
                tail: Number of lines, at most 500.
            """
            return self._call(lambda: self._logs(job_id, tail))

        @mcp.tool
        def jobs_cancel(job_id: str) -> dict:
            """
            Cancel a job. Its cost settles for the minutes it ran.

            Args:
                job_id: Id returned by `jobs_submit` or `dev_evaluate`.
            """
            return self._call(
                lambda: self._job_view(
                    self._broker.cancel(self._broker.get(job_id), "cancelled by agent")
                )
            )

        @mcp.tool
        def dev_evaluate(job_id: str) -> dict:
            """
            Score a finished job's artifact on the public dev split, in a new
            evaluation job charged to your budget. Poll it with `jobs_status`
            or `jobs_wait`; the result appears once it finishes.

            Args:
                job_id: Id of a succeeded job from `jobs_submit`.
            """
            return self._call(
                lambda: self._job_view(self._evaluate(job_id, "dev_eval"))
            )

        @mcp.tool
        def submit(job_id: str) -> dict:
            """
            Submit a finished job's artifact as your final answer. This cancels
            your other jobs, scores the artifact on the hidden test split and
            ends the episode. It can take several minutes.

            Args:
                job_id: Id of a succeeded job from `jobs_submit`.
            """
            return self._call(lambda: self._submit(job_id))

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # -- episode lifecycle --------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Start an episode, after cancelling every job of the previous one.

        Args:
            seed (`int`, *optional*):
                Unused; tasks are deterministic apart from the artifact nonce.
            episode_id (`str`, *optional*):
                Episode id, also used as the jobs' `openenv-episode` label.
            task_id (`str`, *optional*):
                One of the task ids in `TASKS`. Defaults to the constructor's.

        Raises:
            `RuntimeError`: If a job of the previous episode could not be
                confirmed cancelled. It may still be running and billing.
        """
        self._end_jobs("episode reset")
        task_id = task_id or self._default_task_id
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id {task_id!r}; choose from {sorted(TASKS)}"
            )
        if self._namespace is None:
            self._namespace = (
                os.environ.get("POSTTRAIN_NAMESPACE") or self._api.whoami()["name"]
            )
        self._task = TASKS[task_id]
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0, task_id=task_id
        )
        self._broker = Broker(
            self._api,
            namespace=self._namespace,
            episode_id=self._state.episode_id,
            budget_usd=self._task.budget_usd,
            max_jobs=self._task.max_jobs,
            max_dev_evals=self._task.max_dev_evals,
            resource_group_id=os.environ.get("POSTTRAIN_RESOURCE_GROUP_ID"),
        )
        self._nonce = secrets.token_hex(8)
        self._started_at = time.monotonic()
        self._done = False
        self._reward = None
        self._end = {}
        self._score_job_id = None
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task_id": task_id,
                "prompt": self._task.prompt.format(nonce=self._nonce),
                "budget_usd": self._task.budget_usd,
                "max_steps": self._max_steps,
                "max_episode_minutes": self._task.max_episode_minutes,
                "score_timeout_s": self._score_timeout_s(),
                "service_spend": self._broker.spend(),
            },
        )

    def _end_jobs(self, reason: str) -> None:
        if self._broker is None:
            return
        orphaned = self._broker.cancel_all(reason, self._cancel_timeout_s, self._poll_s)
        if orphaned:
            raise RuntimeError(
                f"HF Jobs did not confirm cancelling {orphaned} from episode "
                f"{self._broker.episode_id}; they may still be running and billing"
            )

    def _score_timeout_s(self) -> float:
        # Queueing plus install time come on top of the evaluator's own timeout.
        return self._task.eval_timeout_minutes * 60 + 900.0

    def _tool_timeout(
        self, action: Action, timeout_s: Optional[float]
    ) -> Optional[float]:
        if not isinstance(action, CallToolAction) or self._task is None:
            return timeout_s
        if action.tool_name == "submit":
            return self._score_timeout_s() + self._cancel_timeout_s + TOOL_TIMEOUT_S
        if action.tool_name == "jobs_wait":
            return self._max_wait_s + TOOL_TIMEOUT_S
        return TOOL_TIMEOUT_S

    def _finish(self, obs: Observation) -> Observation:
        if self._broker is None:
            return obs
        if not self._done:
            elapsed_min = (time.monotonic() - self._started_at) / 60
            if self._state.step_count >= self._max_steps:
                self._stop("max_steps", f"Reached {self._max_steps} tool calls")
            elif elapsed_min >= self._task.max_episode_minutes:
                self._stop(
                    "time_limit", f"Reached {self._task.max_episode_minutes} minutes"
                )
        metadata = {**obs.metadata, "service_spend": self._broker.spend()}
        if self._done:
            obs.done = True
            obs.reward = self._reward
            metadata["episode_end"] = self._end
        obs.metadata = metadata
        return obs

    def _stop(self, kind: str, message: str) -> None:
        orphaned = self._broker.cancel_all(kind, self._cancel_timeout_s, self._poll_s)
        self._done = True
        self._reward = 0.0
        self._end = {"reason": kind, "message": message, "orphaned_jobs": orphaned}

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = super().step(
            action, timeout_s=self._tool_timeout(action, timeout_s), **kwargs
        )
        return self._finish(obs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(
            action, timeout_s=self._tool_timeout(action, timeout_s), **kwargs
        )
        return self._finish(obs)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        try:
            self._end_jobs("environment closed")
        finally:
            super().close()

    # -- tools --------------------------------------------------------------

    def _call(self, fn) -> dict:
        if self._broker is None:
            return BrokerError("not_started", "Call reset before using tools").as_dict()
        if self._done:
            return BrokerError("episode_over", "The episode has ended").as_dict()
        try:
            return fn()
        except BrokerError as e:
            return e.as_dict()
        except HfHubHTTPError as e:
            # Hub errors carry request ids and URLs; the agent gets the status only.
            logger.warning("HF Jobs request failed: %s", e)
            return BrokerError(
                "hf_error", f"HF Jobs request failed (HTTP {e.response.status_code})"
            ).as_dict()
        except Exception:
            logger.exception("posttrain_env tool failed")
            return BrokerError(
                "internal_error", "The environment hit an internal error"
            ).as_dict()

    def _budget(self) -> dict:
        broker = self._broker
        return {
            "remaining_usd": broker.remaining_micro_usd() / 1e6,
            "budget_usd": self._task.budget_usd,
            "flavors": {
                f: {"usd_per_hour": broker.rate_micro_usd_per_minute(f) * 60 / 1e6}
                for f in self._task.flavors
            },
            "max_timeout_minutes": self._task.max_timeout_minutes,
            "jobs_left": self._task.max_jobs - broker.count("agent"),
            "dev_evals_left": self._task.max_dev_evals - broker.count("dev_eval"),
            "dev_evaluate_usd": broker.rate_micro_usd_per_minute(self._task.eval_flavor)
            * self._task.eval_timeout_minutes
            / 1e6,
        }

    def _job_view(self, job) -> dict:
        view = job.view()
        if job.role == "agent" and job.state == "succeeded":
            view["artifact"] = {
                "uri": f"openenv-artifact://{self._state.episode_id}/{job.job_id}/outputs",
                "files": self._broker.artifacts(job),
            }
        view["remaining_usd"] = self._broker.remaining_micro_usd() / 1e6
        return view

    def _logs(self, job_id: str, tail: int) -> dict:
        job = self._broker.get(job_id)
        if job.role != "agent":
            raise BrokerError("not_allowed", "Only your own jobs' logs are readable")
        lines = self._api.fetch_job_logs(
            job_id=job.job_id, namespace=self._namespace, tail=min(max(tail, 1), 500)
        )
        return {"job_id": job_id, "logs": "\n".join(lines)[-MAX_LOG_CHARS:]}

    def _evaluate(self, job_id: str, role: str):
        task = self._task
        source = self._broker.refresh(self._broker.get(job_id))
        split, seed = (
            ("dev", task.dev_seed) if role == "dev_eval" else ("test", TEST_SEED)
        )
        return self._broker.evaluate(
            source,
            role=role,
            script_path=task.eval_script_path,
            flavor=task.eval_flavor,
            timeout_minutes=task.eval_timeout_minutes,
            dependencies=list(task.eval_dependencies),
            env={
                "POSTTRAIN_SPLIT": split,
                "POSTTRAIN_SEED": str(seed),
                "POSTTRAIN_EXPECTED": self._nonce,
            },
        )

    def _submit(self, job_id: str) -> dict:
        # A retry after a transient HF error reuses the score job already started.
        if self._score_job_id is None:
            self._broker.require_artifact(
                self._broker.refresh(self._broker.get(job_id))
            )
            for job in self._broker.jobs.values():
                if job.state not in TERMINAL_STATES:
                    self._broker.cancel(job, "submitted")
            self._score_job_id = self._evaluate(job_id, "score").job_id
        score_job = self._broker.get(self._score_job_id)
        self._broker.wait(score_job, self._score_timeout_s(), self._poll_s)
        if score_job.state not in TERMINAL_STATES:
            self._broker.cancel(score_job, "score timed out")
        result = score_job.result or {"status": "timeout", "score": 0.0}
        orphaned = self._broker.cancel_all(
            "submitted", self._cancel_timeout_s, self._poll_s
        )
        self._done = True
        self._reward = result["score"]
        self._end = {
            "reason": "submitted",
            "job_id": score_job.source_job_id,
            "score_job_id": score_job.job_id,
            "score": result,
            "orphaned_jobs": orphaned,
        }
        return {"submitted": score_job.source_job_id, "result": result}
