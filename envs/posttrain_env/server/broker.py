# SPDX-License-Identifier: BSD-3-Clause

"""HF Jobs broker with a reserve-then-settle budget ledger.

The broker is the only holder of the HF token. Agents never see it: jobs are
started with `run_uv_job` and explicit flavor, namespace, timeout and labels,
and with no secrets. A `job.write` token passed into a job could start further
jobs that skip this ledger, so the broker never passes one in.

Money is tracked in integer micro-USD. A job reserves its worst case (the
flavor's per-minute rate times its timeout) before it starts, and settles at
the per-minute rate for the minutes it actually ran.
"""

import json
import math
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import Volume
from huggingface_hub.errors import HfHubHTTPError

# HF Jobs stage -> job state exposed to the agent.
STATE_FROM_STAGE = {
    "SCHEDULING": "queued",
    "RUNNING": "running",
    "COMPLETED": "succeeded",
    "ERROR": "failed",
    "CANCELED": "cancelled",
    "DELETED": "failed",
}
# States HF Jobs has confirmed; "orphaned" is terminal for the ledger but not confirmed.
CONFIRMED_STATES = {"succeeded", "failed", "cancelled"}
TERMINAL_STATES = CONFIRMED_STATES | {"orphaned"}
TERMINAL_STAGES = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
SCORE_MARKER = "POSTTRAIN_SCORE "
ARTIFACTS_MOUNT = "/data"
MAX_SCRIPT_BYTES = 200_000
MAX_DEPENDENCIES = 20
MAX_ARTIFACT_FILES = 50
# Logs can lag a finished job; re-read before declaring the score missing.
MAX_RESULT_READS = 3


class BrokerError(Exception):
    """A request the broker refuses, with a typed reason for the agent."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    def as_dict(self) -> dict:
        return {"error": {"type": self.kind, "message": self.message}}


@dataclass
class Job:
    """One HF Job started by the broker, as tracked in the ledger."""

    job_id: str
    role: str  # "agent", "dev_eval" or "score"
    flavor: str
    timeout_minutes: int
    reserved_micro_usd: int
    bucket: str | None = None
    path: str | None = None
    source_job_id: str | None = None
    state: str = "queued"
    reason: str | None = None
    running_secs: int | None = None
    settled_micro_usd: int = 0
    settled: bool = False
    result: dict | None = None
    result_reads: int = 0

    def view(self) -> dict:
        view = {
            "job_id": self.job_id,
            "role": self.role,
            "state": self.state,
            "flavor": self.flavor,
            "timeout_minutes": self.timeout_minutes,
            "running_secs": self.running_secs or 0,
        }
        if self.reason:
            view["reason"] = self.reason
        if self.source_job_id:
            view["evaluates"] = self.source_job_id
        if self.settled:
            view["cost_usd"] = self.settled_micro_usd / 1e6
        if self.result is not None:
            view["result"] = self.result
        return view


class Broker:
    """
    Start, track and cancel HF Jobs for one episode, against a budget.

    Args:
        api (`huggingface_hub.HfApi`):
            Client holding the broker's token.
        namespace (`str`):
            Namespace jobs run and are billed in.
        episode_id (`str`):
            Episode the jobs belong to, added as a label.
        budget_usd (`float`):
            Budget for agent jobs and dev evaluations.
        max_jobs (`int`):
            Maximum number of agent jobs.
        max_dev_evals (`int`):
            Maximum number of dev evaluations.
        resource_group_id (`str`, *optional*):
            Resource Group to bill, so its Jobs spend limit is a hard backstop.
    """

    def __init__(
        self,
        api: Any,
        *,
        namespace: str,
        episode_id: str,
        budget_usd: float,
        max_jobs: int,
        max_dev_evals: int,
        resource_group_id: str | None = None,
    ):
        self._api = api
        self.namespace = namespace
        self.episode_id = episode_id
        self.budget_micro_usd = round(budget_usd * 1e6)
        self.max_jobs = max_jobs
        self.max_dev_evals = max_dev_evals
        self.resource_group_id = resource_group_id
        self.jobs: dict[str, Job] = {}
        self._idempotency: dict[str, str] = {}
        self._hardware: dict[str, Any] | None = None

    # -- ledger -------------------------------------------------------------

    def count(self, role: str) -> int:
        """Return how many jobs of `role` this episode has started."""
        return sum(1 for j in self.jobs.values() if j.role == role)

    def _charged(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.role in ("agent", "dev_eval")]

    def remaining_micro_usd(self) -> int:
        used = sum(
            j.settled_micro_usd if j.settled else j.reserved_micro_usd
            for j in self._charged()
        )
        return self.budget_micro_usd - used

    def rate_micro_usd_per_minute(self, flavor: str) -> int:
        return self._hardware_info(flavor).unit_cost_micro_usd

    def _hardware_info(self, flavor: str) -> Any:
        if self._hardware is None:
            self._hardware = {h.name: h for h in self._api.list_jobs_hardware()}
        if flavor not in self._hardware:
            raise BrokerError("flavor_unknown", f"Unknown hardware flavor {flavor!r}")
        return self._hardware[flavor]

    def _reserve(self, flavor: str, timeout_minutes: int, charged: bool) -> int:
        reservation = self.rate_micro_usd_per_minute(flavor) * timeout_minutes
        if charged and reservation > self.remaining_micro_usd():
            raise BrokerError(
                "budget_exhausted",
                f"Reserving ${reservation / 1e6:.4f} ({flavor} for {timeout_minutes} min) "
                f"exceeds the remaining ${self.remaining_micro_usd() / 1e6:.4f}",
            )
        return reservation

    def _settle(self, job: Job) -> None:
        if job.running_secs is None:
            # A job that ended without a reported duration may still have run;
            # charge its full reservation unless it never started.
            cost = 0 if job.state == "cancelled" else job.reserved_micro_usd
        else:
            minutes = math.ceil(job.running_secs / 60)
            cost = minutes * self.rate_micro_usd_per_minute(job.flavor)
        job.settled_micro_usd = min(cost, job.reserved_micro_usd)
        job.settled = True

    def spend(self) -> dict:
        """Return the `service_spend` observation entry for this episode."""
        charged = self._charged()
        gpu_s = 0.0
        for job in charged:
            accelerator = self._hardware_info(job.flavor).accelerator
            if accelerator is not None:
                gpu_s += (job.running_secs or 0) * int(accelerator.quantity)
        return {
            "version": 1,
            "settled": {
                "usd": sum(j.settled_micro_usd for j in charged if j.settled) / 1e6,
                "gpu_s": gpu_s,
                "jobs": self.count("agent"),
                "calls": self.count("dev_eval"),
            },
            "reserved": {
                "usd": sum(j.reserved_micro_usd for j in charged if not j.settled) / 1e6
            },
            "final": all(j.state in CONFIRMED_STATES for j in self.jobs.values()),
        }

    # -- jobs ---------------------------------------------------------------

    def submit(
        self,
        script: str,
        *,
        flavor: str,
        timeout_minutes: int,
        allowed_flavors: tuple[str, ...],
        max_timeout_minutes: int,
        dependencies: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Job:
        """Start an agent job after checking limits and reserving its worst case."""
        if idempotency_key and idempotency_key in self._idempotency:
            return self.jobs[self._idempotency[idempotency_key]]
        if flavor not in allowed_flavors:
            raise BrokerError(
                "flavor_not_allowed",
                f"Flavor {flavor!r} is not one of {list(allowed_flavors)}",
            )
        if not 1 <= timeout_minutes <= max_timeout_minutes:
            raise BrokerError(
                "timeout_not_allowed",
                f"timeout_minutes must be in [1, {max_timeout_minutes}]",
            )
        if not script.strip() or len(script.encode()) > MAX_SCRIPT_BYTES:
            raise BrokerError(
                "invalid_script", f"script must be 1 to {MAX_SCRIPT_BYTES} bytes"
            )
        dependencies = dependencies or []
        if len(dependencies) > MAX_DEPENDENCIES:
            raise BrokerError(
                "invalid_script", f"At most {MAX_DEPENDENCIES} dependencies"
            )
        if self.count("agent") >= self.max_jobs:
            raise BrokerError(
                "limit_reached", f"This episode allows {self.max_jobs} jobs"
            )
        reservation = self._reserve(flavor, timeout_minutes, charged=True)
        job = self._run(
            script,
            role="agent",
            flavor=flavor,
            timeout_minutes=timeout_minutes,
            reservation=reservation,
            dependencies=dependencies,
        )
        if idempotency_key:
            self._idempotency[idempotency_key] = job.job_id
        return job

    def evaluate(
        self,
        source: Job,
        *,
        role: str,
        script_path: Path,
        flavor: str,
        timeout_minutes: int,
        dependencies: list[str],
        env: dict[str, str],
    ) -> Job:
        """Run an evaluator job on `source`'s artifact, mounted read-only at `/model`."""
        self.require_artifact(source)
        if role == "dev_eval" and self.count("dev_eval") >= self.max_dev_evals:
            raise BrokerError(
                "limit_reached",
                f"This episode allows {self.max_dev_evals} dev evaluations",
            )
        reservation = self._reserve(flavor, timeout_minutes, charged=role == "dev_eval")
        model = Volume(
            type="bucket",
            source=source.bucket,
            path=f"{source.path}/outputs",
            mount_path="/model",
            read_only=True,
        )
        job = self._run(
            script_path.read_text(),
            role=role,
            flavor=flavor,
            timeout_minutes=timeout_minutes,
            reservation=reservation,
            dependencies=dependencies,
            env=env,
            volumes=[model],
        )
        job.source_job_id = source.job_id
        return job

    def _run(
        self,
        script: str,
        *,
        role: str,
        flavor: str,
        timeout_minutes: int,
        reservation: int,
        dependencies: list[str],
        env: dict[str, str] | None = None,
        volumes: list[Volume] | None = None,
    ) -> Job:
        # run_uv_job uploads a local script to the namespace's jobs-artifacts
        # bucket and mounts that job's folder read-write at /data, so outputs
        # come back without a token inside the job.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{role}.py"
            path.write_text(script)
            info = self._api.run_uv_job(
                script=str(path),
                dependencies=dependencies,
                env=env or {},
                secrets={},
                flavor=flavor,
                timeout=f"{timeout_minutes}m",
                labels={
                    "openenv-env": "posttrain_env",
                    "openenv-episode": self.episode_id,
                    "openenv-role": role,
                },
                volumes=volumes,
                namespace=self.namespace,
                resource_group_id=self.resource_group_id,
            )
        artifacts = [
            v
            for v in info.volumes or []
            if v.type == "bucket" and v.mount_path == ARTIFACTS_MOUNT
        ]
        job = Job(
            job_id=info.id,
            role=role,
            flavor=flavor,
            timeout_minutes=timeout_minutes,
            reserved_micro_usd=reservation,
            bucket=artifacts[0].source if artifacts else None,
            path=artifacts[0].path if artifacts else None,
        )
        self.jobs[job.job_id] = job
        return job

    def require_artifact(self, job: Job) -> None:
        """Raise `not_ready` unless `job` is a succeeded agent job with an artifact."""
        if job.role != "agent" or job.state != "succeeded" or job.bucket is None:
            raise BrokerError(
                "not_ready", f"Job {job.job_id} has no successful artifact"
            )

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise BrokerError("unknown_job", f"No job {job_id!r} in this episode")
        return self.jobs[job_id]

    def finished(self, job: Job) -> bool:
        """True once `job` is settled and, for evaluations, its result is read."""
        return job.settled and (job.role == "agent" or job.result is not None)

    def refresh(self, job: Job) -> Job:
        """Update `job` from HF Jobs, settling it and reading its score once it ends."""
        if self.finished(job):
            return job
        if not job.settled:
            info = self._api.inspect_job(job_id=job.job_id, namespace=self.namespace)
            stage = info.status.stage
            if job.state != "cancelling" or stage in TERMINAL_STAGES:
                job.state = STATE_FROM_STAGE.get(stage, job.state)
            if info.status.message and job.state in ("failed", "cancelled"):
                job.reason = job.reason or info.status.message
            if info.durations is not None and info.durations.running_secs is not None:
                job.running_secs = info.durations.running_secs
            if job.state in TERMINAL_STATES:
                self._settle(job)
        if job.settled and job.role != "agent":
            job.result = self._read_result(job)
        return job

    def _read_result(self, job: Job) -> dict | None:
        if job.state != "succeeded":
            return {"status": job.state, "score": 0.0}
        job.result_reads += 1
        lines = self._api.fetch_job_logs(
            job_id=job.job_id, namespace=self.namespace, tail=200
        )
        for line in reversed(list(lines)):
            if not line.startswith(SCORE_MARKER):
                continue
            try:
                result = json.loads(line[len(SCORE_MARKER) :])
                score = float(result["score"])
            except (ValueError, KeyError, TypeError):
                score = -1.0
            if not 0.0 <= score <= 1.0:
                return {
                    "status": "invalid",
                    "score": 0.0,
                    "feedback": "malformed score",
                }
            metrics = result.get("metrics")
            return {
                "status": str(result.get("status", "ok")),
                "score": score,
                "metrics": dict(list(metrics.items())[:5])
                if isinstance(metrics, dict)
                else {},
                "feedback": str(result.get("feedback", ""))[:500],
            }
        if job.result_reads < MAX_RESULT_READS:
            return None
        return {
            "status": "invalid",
            "score": 0.0,
            "feedback": "evaluator printed no score",
        }

    def cancel(self, job: Job, reason: str) -> Job:
        if self.refresh(job).state in TERMINAL_STATES:
            return job
        # The job may have just finished; the next refresh decides.
        with suppress(HfHubHTTPError):
            self._api.cancel_job(job_id=job.job_id, namespace=self.namespace)
        job.state = "cancelling"
        job.reason = reason
        return self.refresh(job)

    def wait(self, job: Job, timeout_s: float, poll_s: float) -> Job:
        """Poll `job` until it is finished or `timeout_s` passes."""
        deadline = time.monotonic() + timeout_s
        while not self.finished(self.refresh(job)) and time.monotonic() < deadline:
            time.sleep(poll_s)
        return job

    def cancel_all(self, reason: str, timeout_s: float, poll_s: float) -> list[str]:
        """
        Cancel every live job, re-check earlier orphans, and wait for HF Jobs to confirm.

        Jobs still live after `timeout_s` are marked `orphaned` and settled at
        their full reservation, since their real cost is unknown. Orphans are
        retried on every later call until HF Jobs confirms they ended.

        Returns:
            `list[str]`: ids of the orphaned jobs.
        """
        live = [j for j in self.jobs.values() if j.state not in CONFIRMED_STATES]
        for job in live:
            if job.state == "orphaned":
                job.state, job.settled = "cancelling", False
            with suppress(HfHubHTTPError):
                self.cancel(job, reason)
        deadline = time.monotonic() + timeout_s
        while (
            any(j.state not in TERMINAL_STATES for j in live)
            and time.monotonic() < deadline
        ):
            time.sleep(poll_s)
            for job in live:
                with suppress(HfHubHTTPError):
                    self.refresh(job)
        orphaned = []
        for job in live:
            if job.state not in TERMINAL_STATES:
                job.state = "orphaned"
                job.reason = f"{reason}: cancel not confirmed"
                job.settled_micro_usd = job.reserved_micro_usd
                job.settled = True
                orphaned.append(job.job_id)
        return orphaned

    def artifacts(self, job: Job) -> list[dict]:
        """List the files under the job's `/data/outputs/`, with their content hashes."""
        if job.state != "succeeded" or job.bucket is None:
            return []
        prefix = f"{job.path}/outputs/"
        files = []
        for entry in self._api.list_bucket_tree(
            job.bucket, prefix=prefix, recursive=True
        ):
            if entry.type == "file":
                files.append(
                    {
                        "path": entry.path[len(prefix) :],
                        "size": entry.size,
                        "xet_hash": entry.xet_hash,
                    }
                )
            if len(files) >= MAX_ARTIFACT_FILES:
                break
        return files
