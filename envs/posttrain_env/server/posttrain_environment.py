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

import asyncio
import logging
import os
import secrets
import threading
import time
from contextvars import ContextVar
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from fastmcp.tools.tool import ToolResult
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation, State

from .broker import Broker, BrokerError, TERMINAL_STATES
from .tasks import Task, TASKS

logger = logging.getLogger(__name__)

if os.environ.get("POSTTRAIN_TEST_SEED"):
    TEST_SEED = int(os.environ["POSTTRAIN_TEST_SEED"])
else:
    TEST_SEED = secrets.randbelow(2**31)
    logger.warning(
        "POSTTRAIN_TEST_SEED is not set: the hidden test split is random for this "
        "process, so scores are not comparable with other server processes"
    )
TOOL_TIMEOUT_S = 120.0
MAX_LOG_CHARS = 20_000
# Keep asynchronous close fallbacks alive after core releases the session.
_CLOSE_TASKS: set[asyncio.Task] = set()


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
        self._score_source_job_id: str | None = None
        self._score_lock = threading.Lock()
        self._tool_episode = ContextVar("posttrain_episode", default=None)
        self._end: dict = {}
        self._lifecycle_lock = threading.RLock()
        self._close_tasks: dict[Broker | None, asyncio.Task] = {}

        owner = self

        class EpisodeDispatch(Middleware):
            async def on_call_tool(self, context, call_next):
                # The MCP session task may have inherited an earlier episode.
                with owner._lifecycle_lock:
                    ownership = (
                        owner._broker,
                        owner._task,
                        owner._nonce,
                        owner._score_lock,
                    )
                meta = context.fastmcp_context.request_context.meta
                expected = (
                    None if meta is None else meta.model_dump().get("posttrain_episode")
                )
                if expected is not None and expected != ownership[2]:
                    return ToolResult(
                        structured_content=BrokerError(
                            "episode_over", "The episode has ended"
                        ).as_dict()
                    )
                token = owner._tool_episode.set(ownership)
                try:
                    return await call_next(context)
                finally:
                    owner._tool_episode.reset(token)

        mcp.add_middleware(EpisodeDispatch())

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
            datasets: list[str] | None = None,
            idempotency_key: str | None = None,
        ) -> dict:
            """
            Run a Python script as an HF Job, like `hf jobs uv run`.

            The job's worst-case cost (price per minute times `timeout_minutes`)
            is reserved before it starts and must fit the remaining budget. The
            job has no credentials and no network access, and runs as an
            unprivileged user. Write outputs to `/data/outputs/`.

            Args:
                script: Python source of the job. A PEP 723 header may list dependencies.
                flavor: Hardware flavor, one of those listed by `budget`.
                timeout_minutes: Job timeout; the job is stopped when it is reached.
                dependencies: Extra PyPI packages, installed from wheels before the job starts.
                datasets: Public Hub dataset repo ids, mounted read-only at `/datasets/<repo_id>`.
                idempotency_key: Retrying with the same key returns the same job.
            """
            ownership = self._episode()
            broker, task, _, _ = ownership
            return self._call(
                lambda: self._job_view(
                    broker.submit(
                        script,
                        flavor=flavor,
                        timeout_minutes=timeout_minutes,
                        allowed_flavors=task.flavors,
                        max_timeout_minutes=task.max_timeout_minutes,
                        dependencies=dependencies,
                        datasets=datasets,
                        denied_datasets=task.denied_datasets,
                        volumes=task.volumes(),
                        idempotency_key=idempotency_key,
                    ),
                    broker=broker,
                ),
                ownership,
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
            ownership = self._episode()
            broker, task, nonce, _ = ownership
            return self._call(
                lambda: self._job_view(
                    self._evaluate(
                        job_id, "dev_eval", broker=broker, task=task, nonce=nonce
                    ),
                    broker=broker,
                ),
                ownership,
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
            ownership = self._episode()
            return self._call(lambda: self._submit(job_id, ownership), ownership)

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
        with self._lifecycle_lock:
            self._end_jobs("episode reset", self._broker)
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
            self._score_source_job_id = None
            self._score_lock = threading.Lock()
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

    def _end_jobs(self, reason: str, broker: Broker | None) -> None:
        if broker is None:
            return
        orphaned = broker.cancel_all(reason, self._cancel_timeout_s, self._poll_s)
        if orphaned:
            raise RuntimeError(
                f"HF Jobs did not confirm cancelling {orphaned} from episode "
                f"{broker.episode_id}; they may still be running and billing"
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

    def _finish(self, obs: Observation, broker: Broker | None) -> Observation:
        with self._lifecycle_lock:
            if broker is None or broker is not self._broker:
                return obs
            if not self._done:
                elapsed_min = (time.monotonic() - self._started_at) / 60
                if self._state.step_count >= self._max_steps:
                    self._stop("max_steps", f"Reached {self._max_steps} tool calls")
                elif elapsed_min >= self._task.max_episode_minutes:
                    self._stop(
                        "time_limit",
                        f"Reached {self._task.max_episode_minutes} minutes",
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
        with self._lifecycle_lock:
            ownership = self._episode()
            broker = ownership[0]
            self._state.step_count += 1
        token = self._tool_episode.set(ownership)
        try:
            obs = super().step(
                action, timeout_s=self._tool_timeout(action, timeout_s), **kwargs
            )
            return self._finish(obs, broker)
        finally:
            self._tool_episode.reset(token)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        with self._lifecycle_lock:
            ownership = self._episode()
            broker = ownership[0]
            self._state.step_count += 1
        token = self._tool_episode.set(ownership)
        try:
            obs = await super().step_async(
                action, timeout_s=self._tool_timeout(action, timeout_s), **kwargs
            )
            # Cancellation confirmation must stay off the session event loop.
            return await asyncio.to_thread(self._finish, obs, broker)
        finally:
            self._tool_episode.reset(token)

    async def _async_call_tool(self, tool_name: str, arguments: dict) -> Any:
        # MCP runs the server in a separate task: Python context alone does not
        # cross that transport. Reject a queued call after reset at dispatch.
        ownership = self._episode()
        async with self.mcp_session() as client:
            return await client.call_tool(
                tool_name, arguments, meta={"posttrain_episode": ownership[2]}
            )

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
        broker = self._broker
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._close(broker)
        else:
            # Core retries failed executor cleanup directly on its event loop.
            # Retain that retry without waiting for HF cancellation on the loop.
            task = self._close_tasks.get(broker)
            if task is None or task.done():
                task = loop.create_task(self._close_in_thread(broker))
                self._close_tasks[broker] = task
                _CLOSE_TASKS.add(task)

                def finished(task):
                    _CLOSE_TASKS.discard(task)
                    if self._close_tasks.get(broker) is task:
                        del self._close_tasks[broker]

                task.add_done_callback(finished)

    def _close(self, broker: Broker | None) -> None:
        with self._lifecycle_lock:
            try:
                self._end_jobs("environment closed", broker)
            finally:
                if broker is self._broker:
                    super().close()

    async def _close_in_thread(self, broker: Broker | None) -> None:
        try:
            await asyncio.to_thread(self._close, broker)
        except Exception:
            logger.exception("posttrain_env close could not confirm job cancellation")

    # -- tools --------------------------------------------------------------

    def _episode(self):
        ownership = self._tool_episode.get()
        if ownership is not None:
            return ownership
        with self._lifecycle_lock:
            return self._broker, self._task, self._nonce, self._score_lock

    def _call(self, fn, ownership=None) -> dict:
        broker = (ownership if ownership is not None else self._episode())[0]
        with self._lifecycle_lock:
            if broker is None:
                return BrokerError(
                    "not_started", "Call reset before using tools"
                ).as_dict()
            if broker is not self._broker or self._done:
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

    def _job_view(self, job, *, broker=None) -> dict:
        broker = broker if broker is not None else self._broker
        view = job.view()
        if job.role == "agent" and job.state == "succeeded":
            view["artifact"] = {
                "uri": f"openenv-artifact://{broker.episode_id}/{job.job_id}/outputs",
                "files": broker.artifacts(job),
            }
        view["remaining_usd"] = broker.remaining_micro_usd() / 1e6
        return view

    def _logs(self, job_id: str, tail: int) -> dict:
        job = self._broker.get(job_id)
        if job.role != "agent":
            raise BrokerError("not_allowed", "Only your own jobs' logs are readable")
        lines = self._api.fetch_job_logs(
            job_id=job.job_id, namespace=self._namespace, tail=min(max(tail, 1), 500)
        )
        return {"job_id": job_id, "logs": "\n".join(lines)[-MAX_LOG_CHARS:]}

    def _evaluate(self, job_id: str, role: str, *, broker=None, task=None, nonce=None):
        with self._lifecycle_lock:
            broker = broker if broker is not None else self._broker
            task = task if task is not None else self._task
            nonce = nonce if nonce is not None else self._nonce
        source = broker.refresh(broker.get(job_id))
        env = {"POSTTRAIN_EXPECTED": nonce, "HF_HUB_OFFLINE": "1"}
        # The hidden seed goes as a secret, so it doesn't show in the job's spec.
        if role == "dev_eval":
            env |= {"POSTTRAIN_SPLIT": "dev", "POSTTRAIN_SEED": str(task.dev_seed)}
            secrets = {}
        else:
            env |= {"POSTTRAIN_SPLIT": "test"}
            secrets = {"POSTTRAIN_SEED": str(TEST_SEED)}
        return broker.evaluate(
            source,
            role=role,
            script_path=task.eval_script_path,
            flavor=task.eval_flavor,
            timeout_minutes=task.eval_timeout_minutes,
            env=env,
            secrets=secrets,
            volumes=task.volumes(),
        )

    def _submit(self, job_id: str, ownership=None) -> dict:
        broker, task, nonce, score_lock = ownership or self._episode()
        # Serialize scoring, without blocking reset/close on a provider call.
        if not score_lock.acquire(timeout=TOOL_TIMEOUT_S):
            raise BrokerError("creation_pending", "Scoring is in progress; retry later")
        try:
            return self._submit_score(job_id, broker, task, nonce)
        finally:
            score_lock.release()

    def _submit_score(self, job_id, broker, task, nonce) -> dict:
        with self._lifecycle_lock:
            if broker is not self._broker:
                return BrokerError("episode_over", "The episode has ended").as_dict()
            score_job_id = self._score_job_id
            job_id = self._score_source_job_id or job_id
            score_timeout_s = task.eval_timeout_minutes * 60 + 900.0
        if score_job_id is None:
            broker.require_artifact(broker.refresh(broker.get(job_id)))
            with self._lifecycle_lock:
                if broker is not self._broker:
                    return BrokerError(
                        "episode_over", "The episode has ended"
                    ).as_dict()
                # Preserve the selected artifact even when create/recovery fails.
                self._score_source_job_id = job_id
            for job in list(broker.jobs.values()):
                if job.role != "score" and job.state not in TERMINAL_STATES:
                    broker.cancel(job, "submitted")
            score_job = self._evaluate(
                job_id, "score", broker=broker, task=task, nonce=nonce
            )
            with self._lifecycle_lock:
                if broker is not self._broker:
                    return BrokerError(
                        "episode_over", "The episode has ended"
                    ).as_dict()
                self._score_job_id = score_job.job_id
        else:
            score_job = broker.get(score_job_id)
        broker.wait(score_job, score_timeout_s, self._poll_s)
        if score_job.state not in TERMINAL_STATES:
            broker.cancel(score_job, "score timed out")
        result = score_job.result or {"status": "timeout", "score": 0.0}
        orphaned = broker.cancel_all("submitted", self._cancel_timeout_s, self._poll_s)
        response = {"submitted": score_job.source_job_id, "result": result}
        with self._lifecycle_lock:
            if broker is not self._broker:
                return response
            self._done = True
            # A score job that did not run to the end, or printed no valid score, is
            # the grader's fault, not the agent's: the episode ends unscored, and a
            # missing reward is not zero.
            scored = score_job.state == "succeeded" and result["status"] not in (
                "error",
                "timeout",
            )
            self._reward = result["score"] if scored else None
            self._end = {
                "reason": "submitted" if scored else "score_failed",
                "job_id": score_job.source_job_id,
                "score_job_id": score_job.job_id,
                "score": result,
                "orphaned_jobs": orphaned,
            }
        return response
