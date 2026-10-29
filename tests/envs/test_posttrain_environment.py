# SPDX-License-Identifier: BSD-3-Clause

"""Tests for posttrain_env: the HF Jobs broker, its budget ledger and the episode flow.

HF Jobs is replaced by an in-memory fake, so these tests make no network calls
and spend nothing.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import ctypes.util
import dataclasses
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from envs.posttrain_env.server import posttrain_environment
from envs.posttrain_env.server.broker import LAUNCHER, MAX_RESULT_READS, SCORE_MARKER
from envs.posttrain_env.server.posttrain_environment import PostTrainEnvironment
from envs.posttrain_env.server.tasks import TASKS
from huggingface_hub import Volume
from huggingface_hub.errors import HfHubHTTPError
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

ENV_DIR = Path(__file__).parents[2] / "envs" / "posttrain_env"
CPU_BASIC = 167  # micro-USD per minute
A10G_SMALL = 16_667


def hardware(name, micro_usd_per_minute, gpus=None):
    accelerator = None if gpus is None else SimpleNamespace(quantity=str(gpus))
    return SimpleNamespace(
        name=name, unit_cost_micro_usd=micro_usd_per_minute, accelerator=accelerator
    )


class NotFound(HfHubHTTPError):
    """A Hub 404, without the HTTP response the real error carries."""

    def __init__(self, repo_id):
        Exception.__init__(self, f"{repo_id} not found")


class FakeHfApi:
    """The subset of `HfApi` the broker uses, backed by a dict of jobs."""

    def __init__(self):
        self.jobs = {}
        self.runs = []
        self.cancel_confirms = True
        # Applied the first time a job with this role label is inspected.
        self.on_inspect = {}
        self.datasets = {
            "open/data": SimpleNamespace(id="open/data", sha="abc123", gated=False),
            "gated/data": SimpleNamespace(id="gated/data", sha="def456", gated="auto"),
        }
        self.dataset_lookups = []
        # Raise from the next run_uv_job call after the job was created.
        self.fail_after_create = False

    def whoami(self):
        return {"name": "operator"}

    def list_jobs_hardware(self):
        return [
            hardware("cpu-basic", CPU_BASIC),
            hardware("cpu-upgrade", 500),
            hardware("t4-small", 6_667, gpus=1),
            hardware("a10g-small", A10G_SMALL, gpus=1),
        ]

    def run_uv_job(self, *, script, **kwargs):
        n = len(self.runs)
        job_id = f"job{n}"
        self.runs.append(
            {"job_id": job_id, "script": Path(script).read_text(), **kwargs}
        )
        self.jobs[job_id] = {
            "stage": "SCHEDULING",
            "running_secs": None,
            "logs": [],
            "role": kwargs["labels"]["openenv-role"],
        }
        artifacts = Volume(
            type="bucket",
            source="operator/jobs-artifacts",
            path=f"run{n}",
            mount_path="/data",
            read_only=False,
        )
        info = SimpleNamespace(
            id=job_id, volumes=(kwargs["volumes"] or []) + [artifacts]
        )
        self.jobs[job_id].update(labels=kwargs["labels"], info=info)
        if self.fail_after_create:
            self.fail_after_create = False
            raise ConnectionError("read timed out")
        return info

    def list_jobs(self, *, labels, namespace):
        return [
            job["info"]
            for job in self.jobs.values()
            if labels.items() <= job["labels"].items()
        ]

    def inspect_job(self, *, job_id, namespace):
        job = self.jobs[job_id]
        if job["stage"] == "SCHEDULING" and job["role"] in self.on_inspect:
            job.update(self.on_inspect[job["role"]])
        return SimpleNamespace(
            status=SimpleNamespace(stage=job["stage"], message=job.get("message")),
            durations=SimpleNamespace(running_secs=job["running_secs"]),
        )

    def cancel_job(self, *, job_id, namespace):
        job = self.jobs[job_id]
        job["cancel_requested"] = True
        if self.cancel_confirms:
            job["stage"] = "CANCELED"

    def fetch_job_logs(self, *, job_id, namespace, tail):
        return iter(self.jobs[job_id]["logs"][-tail:])

    def dataset_info(self, repo_id, *, token):
        self.dataset_lookups.append((repo_id, token))
        if repo_id not in self.datasets:
            raise NotFound(repo_id)
        return self.datasets[repo_id]

    def list_bucket_tree(self, bucket_id, prefix, recursive):
        return [
            SimpleNamespace(
                type="file", path=f"{prefix}model.safetensors", size=10, xet_hash="h1"
            ),
            SimpleNamespace(type="directory", path=f"{prefix}sub"),
        ]

    def finish(self, job_id, stage="COMPLETED", running_secs=90, logs=()):
        self.jobs[job_id].update(
            stage=stage, running_secs=running_secs, logs=list(logs)
        )


def score_line(score, **extra):
    return SCORE_MARKER + json.dumps({"status": "ok", "score": score, **extra})


def launched(run):
    """Return the agent script and requirements embedded in a launcher job."""
    script = run["script"]
    encoded = re.search(r'^AGENT_SCRIPT = "([^"]*)"', script, re.M).group(1)
    requirements = re.search(r"^REQUIREMENTS = (.*)$", script, re.M).group(1)
    return base64.b64decode(encoded).decode(), ast.literal_eval(requirements)


@pytest.fixture
def api():
    return FakeHfApi()


@pytest.fixture
def env(api):
    env = PostTrainEnvironment(api=api, poll_s=0.0, cancel_timeout_s=0.01)
    yield env
    api.cancel_confirms = True
    env.close()


def call(env, tool, **arguments):
    obs = env.step(CallToolAction(tool_name=tool, arguments=arguments))
    assert obs.error is None, obs.error
    return obs.result.data, obs


def submit_job(
    env, flavor="cpu-basic", timeout_minutes=10, script="print('train')", **extra
):
    data, _ = call(
        env,
        "jobs_submit",
        script=script,
        flavor=flavor,
        timeout_minutes=timeout_minutes,
        **extra,
    )
    return data


class TestReset:
    def test_reset_reports_task_budget_and_zero_spend(self, env):
        obs = env.reset(task_id="t0_artifact")
        meta = obs.metadata
        assert meta["task_id"] == "t0_artifact"
        assert "/data/outputs/answer.txt" in meta["prompt"]
        assert meta["budget_usd"] == 0.05
        assert meta["service_spend"] == {
            "version": 1,
            "settled": {"usd": 0.0, "gpu_s": 0.0, "jobs": 0, "calls": 0},
            "reserved": {"usd": 0.0},
            "final": True,
        }

    def test_unknown_task_is_rejected(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nope")

    def test_tools_match_the_manifest(self, env):
        env.reset(task_id="t0_artifact")
        names = {t.name for t in env.step(ListToolsAction()).tools}
        manifest = yaml.safe_load((ENV_DIR / "openenv.yaml").read_text())
        assert names == set(manifest["validation"]["capabilities"]["declared_tools"])

    def test_tools_before_reset_are_refused(self, env):
        data, _ = call(env, "budget")
        assert data["error"]["type"] == "not_started"


class TestSubmitJobs:
    def test_job_starts_with_explicit_settings_and_no_secrets(self, env, api):
        env.reset(task_id="t0_artifact", episode_id="ep1")
        job = submit_job(env, timeout_minutes=10, dependencies=["numpy"])
        run = api.runs[0]
        assert run["secrets"] == {}
        assert run["namespace"] == "operator"
        assert run["flavor"] == "cpu-basic"
        assert run["timeout"] == "10m"
        request = run["labels"].pop("openenv-request")
        assert len(request) == 32
        assert run["labels"] == {
            "openenv-env": "posttrain_env",
            "openenv-episode": "ep1",
            "openenv-role": "agent",
        }
        assert run["volumes"] == []
        assert job["state"] == "queued"
        assert job["remaining_usd"] == pytest.approx(0.05 - 10 * CPU_BASIC / 1e6)

    def test_script_runs_in_the_launcher_with_wheel_only_dependencies(self, env, api):
        env.reset(task_id="t0_artifact")
        script = '# /// script\r\n# dependencies = ["torch==2.14.1"]\r\n# ///\r\n'
        submit_job(env, script=script, dependencies=["numpy>=2"])
        run = api.runs[0]
        agent_script, requirements = launched(run)
        assert agent_script == script
        assert requirements == ["torch==2.14.1", "numpy>=2"]
        # The launcher's own header installs nothing: it installs the
        # requirements itself, from wheels, before the filter.
        assert "# dependencies = []" in run["script"]
        assert '"--no-build"' in run["script"]
        assert run["dependencies"] == []
        assert "install_filter()" in run["script"]
        compile(run["script"], "launcher.py", "exec")

    @pytest.mark.parametrize(
        "dependencies, script",
        [
            (["pkg @ https://example.com/pkg.whl"], "print(1)"),
            (["git+https://github.com/owner/repo"], "print(1)"),
            (["numpy; nonsense"], "print(1)"),
            (['foo; python_version == "3.1\n0"'], "print(1)"),
            (["numpy"] * 21, "print(1)"),
            ([], "# /// script\n# dependencies = [\n# ///\nprint(1)\n"),
            ([], '# /// script\n# dependencies = ["x @ file:///etc/passwd"]\n# ///\n'),
        ],
    )
    def test_dependencies_must_be_pypi_packages(self, env, api, dependencies, script):
        env.reset(task_id="t0_artifact")
        data, _ = call(
            env,
            "jobs_submit",
            script=script,
            flavor="cpu-basic",
            timeout_minutes=5,
            dependencies=dependencies,
        )
        assert data["error"]["type"] in ("invalid_dependency", "invalid_script")
        assert api.runs == []

    def test_datasets_are_mounted_read_only_at_their_revision(self, env, api):
        env.reset(task_id="t0_artifact")
        submit_job(env, datasets=["open/data", "open/data"])
        assert api.runs[0]["volumes"] == [
            Volume(
                type="dataset",
                source="open/data",
                revision="abc123",
                mount_path="/datasets/open/data",
                read_only=True,
            )
        ]
        # Looked up anonymously, so the broker's token cannot open private repos.
        assert api.dataset_lookups == [("open/data", False)]

    @pytest.mark.parametrize(
        "dataset",
        ["gated/data", "private/data", "denied/data", "old/name", "../data", "data"],
    )
    def test_gated_missing_denied_and_malformed_datasets_are_refused(
        self, env, api, monkeypatch, dataset
    ):
        task = dataclasses.replace(
            TASKS["t0_artifact"], denied_datasets=("Denied/Data",)
        )
        monkeypatch.setitem(TASKS, "t0_artifact", task)
        denied = SimpleNamespace(id="denied/data", sha="x", gated=False)
        # "old/name" is the denied repo's former id, which redirects to it.
        api.datasets.update({"denied/data": denied, "old/name": denied})
        env.reset(task_id="t0_artifact")
        data, _ = call(
            env,
            "jobs_submit",
            script="print(1)",
            flavor="cpu-basic",
            timeout_minutes=5,
            datasets=[dataset],
        )
        assert data["error"]["type"] == "dataset_not_allowed"
        assert api.runs == []

    def test_base_model_is_mounted_for_agent_jobs_and_evaluations(self, env, api):
        env.reset(task_id="t0_arithmetic_format")
        job = submit_job(env, flavor="cpu-upgrade", timeout_minutes=5)
        api.finish(job["job_id"])
        call(env, "dev_evaluate", job_id=job["job_id"])
        base = Volume(
            type="model",
            source="HuggingFaceTB/SmolLM2-135M-Instruct",
            revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
            mount_path="/base_model",
            read_only=True,
        )
        agent_run, eval_run = api.runs
        assert agent_run["volumes"] == [base]
        assert eval_run["volumes"][1:] == [base]
        assert eval_run["env"]["HF_HUB_OFFLINE"] == "1"

    def test_resource_group_bills_the_backstop(self, env, api, monkeypatch):
        monkeypatch.setenv("POSTTRAIN_RESOURCE_GROUP_ID", "rg-123")
        env.reset(task_id="t0_artifact")
        submit_job(env)
        assert api.runs[0]["resource_group_id"] == "rg-123"

    def test_flavor_timeout_and_budget_are_enforced_before_starting(self, env, api):
        env.reset(task_id="t0_arithmetic_format")
        denied = submit_job(env, flavor="a100-large")
        assert denied["error"]["type"] == "flavor_not_allowed"
        denied = submit_job(env, flavor="a10g-small", timeout_minutes=61)
        assert denied["error"]["type"] == "timeout_not_allowed"
        # Each a10g-small hour reserves about $1.00 of the $3.00 budget.
        for _ in range(2):
            assert "error" not in submit_job(
                env, flavor="a10g-small", timeout_minutes=60
            )
        denied = submit_job(env, flavor="a10g-small", timeout_minutes=60)
        assert denied["error"]["type"] == "budget_exhausted"
        assert len(api.runs) == 2

    def test_job_count_limit(self, env):
        env.reset(task_id="t0_artifact")
        for _ in range(3):
            assert "error" not in submit_job(env, timeout_minutes=1)
        assert submit_job(env, timeout_minutes=1)["error"]["type"] == "limit_reached"

    def test_idempotency_key_reuses_the_job_and_its_reservation(self, env, api):
        env.reset(task_id="t0_artifact")
        first = submit_job(env, idempotency_key="k1")
        again = submit_job(env, idempotency_key="k1")
        assert again["job_id"] == first["job_id"]
        assert again["remaining_usd"] == first["remaining_usd"]
        assert len(api.runs) == 1


class TestJobLifecycle:
    def test_finished_job_settles_per_started_minute_and_lists_its_artifact(
        self, env, api
    ):
        env.reset(task_id="t0_artifact", episode_id="ep1")
        job = submit_job(env, timeout_minutes=10)
        api.finish(job["job_id"], running_secs=90)
        data, obs = call(env, "jobs_status", job_id=job["job_id"])
        assert data["state"] == "succeeded"
        assert data["cost_usd"] == pytest.approx(2 * CPU_BASIC / 1e6)
        assert data["remaining_usd"] == pytest.approx(0.05 - 2 * CPU_BASIC / 1e6)
        assert data["artifact"] == {
            "uri": f"openenv-artifact://ep1/{job['job_id']}/outputs",
            "files": [{"path": "model.safetensors", "size": 10, "xet_hash": "h1"}],
        }
        spend = obs.metadata["service_spend"]
        assert spend["settled"]["usd"] == pytest.approx(2 * CPU_BASIC / 1e6)
        assert spend["reserved"]["usd"] == 0.0
        assert spend["final"] is True

    def test_gpu_seconds_count_accelerators(self, env, api):
        env.reset(task_id="t0_arithmetic_format")
        job = submit_job(env, flavor="t4-small", timeout_minutes=5)
        api.finish(job["job_id"], running_secs=120)
        _, obs = call(env, "jobs_status", job_id=job["job_id"])
        assert obs.metadata["service_spend"]["settled"]["gpu_s"] == 120

    def test_failed_job_reports_its_reason(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        api.jobs[job["job_id"]]["message"] = "OOM"
        api.finish(job["job_id"], stage="ERROR", running_secs=30)
        data, _ = call(env, "jobs_status", job_id=job["job_id"])
        assert data["state"] == "failed"
        assert data["reason"] == "OOM"
        assert "artifact" not in data

    def test_cancel_settles_the_minutes_that_ran(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env, timeout_minutes=10)
        api.jobs[job["job_id"]].update(stage="RUNNING", running_secs=61)
        data, _ = call(env, "jobs_cancel", job_id=job["job_id"])
        assert data["state"] == "cancelled"
        assert data["cost_usd"] == pytest.approx(2 * CPU_BASIC / 1e6)

    def test_unknown_job(self, env):
        env.reset(task_id="t0_artifact")
        data, _ = call(env, "jobs_status", job_id="other")
        assert data["error"]["type"] == "unknown_job"

    def test_logs_are_readable_for_agent_jobs(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        api.finish(job["job_id"], logs=["step 1", "step 2"])
        data, _ = call(env, "jobs_logs", job_id=job["job_id"], tail=1)
        assert data["logs"] == "step 2"


class TestEvaluation:
    def _finished_job(self, env, api):
        job = submit_job(env)
        api.finish(job["job_id"])
        return job

    def test_dev_evaluate_mounts_the_artifact_read_only_and_reads_the_score(
        self, env, api
    ):
        env.reset(task_id="t0_arithmetic_format")
        job = submit_job(env, flavor="cpu-upgrade", timeout_minutes=5)
        api.finish(job["job_id"])
        evaluation, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        run = api.runs[-1]
        model = run["volumes"][0]
        assert (model.source, model.path, model.mount_path, model.read_only) == (
            "operator/jobs-artifacts",
            "run0/outputs",
            "/model",
            True,
        )
        assert run["env"]["POSTTRAIN_SPLIT"] == "dev"
        assert run["env"]["POSTTRAIN_SEED"] == "1234"
        assert run["dependencies"] == []
        assert run["labels"]["openenv-role"] == "dev_eval"
        assert "POSTTRAIN_SCORE" in run["script"]

        api.finish(
            evaluation["job_id"],
            logs=["loading", score_line(0.42, metrics={"accuracy": 0.42})],
        )
        data, obs = call(env, "jobs_status", job_id=evaluation["job_id"])
        assert data["result"]["score"] == 0.42
        assert data["result"]["metrics"] == {"accuracy": 0.42}
        assert obs.metadata["service_spend"]["settled"]["calls"] == 1

    def test_dev_evaluate_needs_a_succeeded_agent_job(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        data, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        assert data["error"]["type"] == "not_ready"

    def test_dev_evaluate_limit(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished_job(env, api)
        for _ in range(2):
            assert "error" not in call(env, "dev_evaluate", job_id=job["job_id"])[0]
        data, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        assert data["error"]["type"] == "limit_reached"

    def test_evaluation_logs_are_not_readable(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished_job(env, api)
        evaluation, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        data, _ = call(env, "jobs_logs", job_id=evaluation["job_id"])
        assert data["error"]["type"] == "not_allowed"

    @pytest.mark.parametrize(
        "logs",
        [
            [],
            ["POSTTRAIN_SCORE not json"],
            [SCORE_MARKER + json.dumps({"score": 1.5})],
            [SCORE_MARKER + json.dumps([0.5])],
        ],
    )
    def test_missing_or_malformed_scores_are_evaluator_errors(self, env, api, logs):
        env.reset(task_id="t0_artifact")
        job = self._finished_job(env, api)
        evaluation, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        api.finish(evaluation["job_id"], logs=logs)
        # A missing score line is re-read a few times in case the logs lag.
        for _ in range(MAX_RESULT_READS):
            data, _ = call(env, "jobs_status", job_id=evaluation["job_id"])
        assert data["result"]["status"] == "error"
        assert data["result"]["score"] == 0.0


class TestSubmitEpisode:
    def test_submit_scores_the_hidden_split_and_ends_the_episode(self, env, api):
        env.reset(task_id="t0_arithmetic_format")
        best = submit_job(env, flavor="cpu-upgrade", timeout_minutes=5)
        api.finish(best["job_id"])
        straggler = submit_job(env, flavor="cpu-upgrade", timeout_minutes=5)
        api.jobs[straggler["job_id"]].update(stage="RUNNING", running_secs=30)
        api.on_inspect["score"] = {
            "stage": "COMPLETED",
            "running_secs": 200,
            "logs": [score_line(0.8)],
        }

        data, obs = call(env, "submit", job_id=best["job_id"])

        assert data["result"]["score"] == 0.8
        assert obs.done is True
        assert obs.reward == 0.8
        score_run = api.runs[-1]
        assert score_run["labels"]["openenv-role"] == "score"
        assert score_run["env"]["POSTTRAIN_SPLIT"] == "test"
        # The hidden seed is a secret, not part of the job's visible spec.
        assert "POSTTRAIN_SEED" not in score_run["env"]
        assert score_run["secrets"] == {
            "POSTTRAIN_SEED": str(posttrain_environment.TEST_SEED)
        }
        assert api.jobs[straggler["job_id"]]["cancel_requested"]
        end = obs.metadata["episode_end"]
        assert end["reason"] == "submitted"
        assert end["orphaned_jobs"] == []
        # The grader's evaluation is not charged to the agent's budget.
        spend = obs.metadata["service_spend"]
        assert spend["settled"]["usd"] == pytest.approx((2 + 1) * 500 / 1e6)
        assert spend["final"] is True

    def test_artifact_task_expects_the_episode_nonce(self, env, api):
        obs = env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        call(env, "dev_evaluate", job_id=job["job_id"])
        assert api.runs[-1]["env"]["POSTTRAIN_EXPECTED"] in obs.metadata["prompt"]

    def _finished(self, env, api):
        job = submit_job(env)
        api.finish(job["job_id"])
        return job

    def test_tools_after_the_episode_ends_are_refused(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        api.on_inspect["score"] = {
            "stage": "COMPLETED",
            "running_secs": 10,
            "logs": [score_line(1.0)],
        }
        call(env, "submit", job_id=job["job_id"])
        data, obs = call(env, "budget")
        assert data["error"]["type"] == "episode_over"
        assert obs.done is True

    def test_failed_score_job_ends_the_episode_unscored(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        # The evaluator printed a score, then the job hit its timeout.
        api.on_inspect["score"] = {
            "stage": "ERROR",
            "running_secs": 10,
            "logs": ['POSTTRAIN_SCORE {"score": 1.0}'],
        }
        data, obs = call(env, "submit", job_id=job["job_id"])
        assert data["result"] == {"status": "failed", "score": 0.0}
        assert obs.done is True
        assert obs.reward is None
        assert obs.metadata["episode_end"]["reason"] == "score_failed"

    def test_score_job_without_a_score_line_ends_the_episode_unscored(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        api.on_inspect["score"] = {"stage": "COMPLETED", "running_secs": 10}
        data, obs = call(env, "submit", job_id=job["job_id"])
        assert data["result"]["status"] == "error"
        assert obs.reward is None
        assert obs.metadata["episode_end"]["reason"] == "score_failed"

    def test_max_steps_ends_the_episode_and_cancels_jobs(self, api):
        env = PostTrainEnvironment(
            api=api, max_steps=2, poll_s=0.0, cancel_timeout_s=0.01
        )
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        _, obs = call(env, "budget")
        assert obs.done is True
        assert obs.reward == 0.0
        assert obs.metadata["episode_end"]["reason"] == "max_steps"
        assert api.jobs[job["job_id"]]["stage"] == "CANCELED"


class TestEpisodeBoundaries:
    def test_reset_cancels_the_previous_episodes_jobs(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        env.reset(task_id="t0_artifact")
        assert api.jobs[job["job_id"]]["stage"] == "CANCELED"

    def test_reset_refuses_while_a_cancel_is_unconfirmed(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        api.jobs[job["job_id"]]["stage"] = "RUNNING"
        api.cancel_confirms = False
        with pytest.raises(RuntimeError, match="did not confirm cancelling"):
            env.reset(task_id="t0_artifact")


def test_websocket_path_ends_the_episode_with_the_score(api):
    """`step_async` is what the WebSocket server calls; it must end episodes too."""
    import asyncio

    env = PostTrainEnvironment(api=api, poll_s=0.0, cancel_timeout_s=0.01)
    env.reset(task_id="t0_artifact")

    async def run():
        job = await env.step_async(
            CallToolAction(
                tool_name="jobs_submit",
                arguments={
                    "script": "print(1)",
                    "flavor": "cpu-basic",
                    "timeout_minutes": 5,
                },
            )
        )
        job_id = job.result.data["job_id"]
        api.finish(job_id)
        api.on_inspect["score"] = {
            "stage": "COMPLETED",
            "running_secs": 10,
            "logs": [score_line(1.0)],
        }
        return await env.step_async(
            CallToolAction(tool_name="submit", arguments={"job_id": job_id})
        )

    obs = asyncio.run(run())
    assert obs.done is True
    assert obs.reward == 1.0
    env.close()


def test_websocket_path_waits_for_cancels_off_the_event_loop(api):
    """Other sessions share the event loop, so ending an episode must not block it."""
    import asyncio

    env = PostTrainEnvironment(api=api, max_steps=2, poll_s=0.05, cancel_timeout_s=0.5)
    env.reset(task_id="t0_artifact")
    job = submit_job(env)
    api.jobs[job["job_id"]]["stage"] = "RUNNING"
    api.cancel_confirms = False
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def run():
        ticker = asyncio.create_task(tick())
        obs = await env.step_async(CallToolAction(tool_name="budget", arguments={}))
        ticker.cancel()
        return obs

    obs = asyncio.run(run())
    assert obs.metadata["episode_end"]["orphaned_jobs"] == [job["job_id"]]
    assert ticks >= 10
    api.cancel_confirms = True
    env.close()


def http_error(status=500):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    request = httpx.Request("GET", "https://huggingface.co/api/jobs/x")
    return HfHubHTTPError("boom", response=httpx.Response(status, request=request))


class TestFailureHandling:
    """Regressions for HF Jobs behaving badly: lagging logs, failed calls, lost cancels."""

    def _finished(self, env, api):
        job = submit_job(env)
        api.finish(job["job_id"])
        return job

    def test_score_line_that_arrives_late_is_still_read(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        evaluation, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        api.finish(evaluation["job_id"], logs=[])
        data, _ = call(env, "jobs_status", job_id=evaluation["job_id"])
        assert data["state"] == "succeeded"
        assert "result" not in data
        api.jobs[evaluation["job_id"]]["logs"] = [score_line(0.9)]
        data, _ = call(env, "jobs_status", job_id=evaluation["job_id"])
        assert data["result"]["score"] == 0.9

    def test_job_without_a_duration_is_charged_its_reservation(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env, timeout_minutes=10)
        api.finish(job["job_id"], running_secs=None)
        data, _ = call(env, "jobs_status", job_id=job["job_id"])
        assert data["cost_usd"] == pytest.approx(10 * CPU_BASIC / 1e6)

    def test_job_cancelled_before_it_started_costs_nothing(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env, timeout_minutes=10)
        data, _ = call(env, "jobs_cancel", job_id=job["job_id"])
        assert data["state"] == "cancelled"
        assert data["cost_usd"] == 0.0

    @pytest.mark.parametrize("error", [http_error(409), ConnectionError("reset")])
    def test_a_failing_cancel_call_does_not_stop_the_others(
        self, env, api, monkeypatch, error
    ):
        env.reset(task_id="t0_artifact")
        first, second = submit_job(env), submit_job(env)
        cancel = api.cancel_job

        def flaky_cancel(*, job_id, namespace):
            if job_id == first["job_id"]:
                raise error
            cancel(job_id=job_id, namespace=namespace)

        monkeypatch.setattr(api, "cancel_job", flaky_cancel)
        with pytest.raises(RuntimeError, match=first["job_id"]):
            env.reset(task_id="t0_artifact")
        assert api.jobs[second["job_id"]]["stage"] == "CANCELED"

    def test_job_created_by_a_failed_call_is_still_tracked(self, env, api):
        env.reset(task_id="t0_artifact")
        api.fail_after_create = True
        job = submit_job(env, timeout_minutes=10)
        assert job["job_id"] == "job0"
        assert len(api.runs) == 1
        assert job["remaining_usd"] == pytest.approx(0.05 - 10 * CPU_BASIC / 1e6)
        env.reset(task_id="t0_artifact")
        assert api.jobs["job0"]["stage"] == "CANCELED"

    def test_orphans_are_retried_until_hf_confirms_them(self, env, api):
        env.reset(task_id="t0_artifact")
        job = submit_job(env)
        api.jobs[job["job_id"]]["stage"] = "RUNNING"
        api.cancel_confirms = False
        for _ in range(2):
            with pytest.raises(RuntimeError, match="did not confirm cancelling"):
                env.reset(task_id="t0_artifact")
        api.cancel_confirms = True
        env.reset(task_id="t0_artifact")
        assert api.jobs[job["job_id"]]["stage"] == "CANCELED"

    def test_submit_retry_after_an_hf_error_reuses_the_score_job(
        self, env, api, monkeypatch
    ):
        env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        inspect = api.inspect_job
        failures = {"left": 1}

        def flaky_inspect(*, job_id, namespace):
            if api.jobs[job_id]["role"] == "score" and failures["left"]:
                failures["left"] -= 1
                raise http_error(503)
            return inspect(job_id=job_id, namespace=namespace)

        monkeypatch.setattr(api, "inspect_job", flaky_inspect)
        api.on_inspect["score"] = {
            "stage": "COMPLETED",
            "running_secs": 10,
            "logs": [score_line(1.0)],
        }
        data, obs = call(env, "submit", job_id=job["job_id"])
        assert data == {
            "error": {
                "type": "hf_error",
                "message": "HF Jobs request failed (HTTP 503)",
            }
        }
        assert obs.done is False
        data, obs = call(env, "submit", job_id=job["job_id"])
        assert data["result"]["score"] == 1.0
        assert obs.done is True
        assert (
            sum(1 for run in api.runs if run["labels"]["openenv-role"] == "score") == 1
        )

    def test_unexpected_errors_are_not_shown_to_the_agent(self, env, api, monkeypatch):
        env.reset(task_id="t0_artifact")

        def broken(**kwargs):
            raise ConnectionError("https://internal.example/secret-path")

        monkeypatch.setattr(api, "run_uv_job", broken)
        data = submit_job(env)
        assert data == {
            "error": {
                "type": "internal_error",
                "message": "The environment hit an internal error",
            }
        }


def test_non_object_metrics_are_dropped(env, api):
    env.reset(task_id="t0_artifact")
    job = submit_job(env)
    api.finish(job["job_id"])
    evaluation, _ = call(env, "dev_evaluate", job_id=job["job_id"])
    api.finish(evaluation["job_id"], logs=[score_line(0.5, metrics=[1, 2])])
    data, _ = call(env, "jobs_status", job_id=evaluation["job_id"])
    assert data["result"]["score"] == 0.5
    assert data["result"]["metrics"] == {}


@pytest.mark.skipif(
    sys.platform != "linux" or ctypes.util.find_library("seccomp") is None,
    reason="the launcher's seccomp filter needs Linux and libseccomp",
)
def test_launcher_filter_blocks_network_sockets_but_not_unix_sockets():
    code = f"""
import importlib.util, socket
spec = importlib.util.spec_from_file_location("launcher", {str(LAUNCHER)!r})
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.install_filter()
for family, kind in [
    (socket.AF_INET, socket.SOCK_STREAM),
    (socket.AF_INET, socket.SOCK_DGRAM),
    (socket.AF_INET6, socket.SOCK_STREAM),
]:
    try:
        socket.socket(family, kind)
        print("open")
    except PermissionError:
        print("blocked")
socket.socket(socket.AF_UNIX).close()
print("unix")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["blocked", "blocked", "blocked", "unix"]


@pytest.mark.parametrize("same_key", [False, True])
@pytest.mark.parametrize("lost_response", [False, True])
def test_concurrent_fastmcp_admission(env, api, monkeypatch, same_key, lost_response):
    env.reset(task_id="t0_arithmetic_format")
    api.fail_after_create = lost_response
    entered = threading.Event()
    release = threading.Event()
    api_lock = threading.Lock()
    run = api.run_uv_job

    def blocked_run(**kwargs):
        entered.set()
        assert release.wait(5)
        with api_lock:
            return run(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", blocked_run)

    async def exercise():
        async with env.mcp_session() as client:

            def request(i):
                return client.call_tool(
                    "jobs_submit",
                    {
                        "script": "print(1)",
                        "flavor": "a10g-small",
                        "timeout_minutes": 60,
                        "idempotency_key": "same" if same_key else str(i),
                    },
                )

            calls = [asyncio.create_task(request(i)) for i in range(4)]
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                # A budget call traverses the same FastMCP thread pool while
                # provider submission is pending.
                budget = await client.call_tool("budget", {})
                assert budget.data["remaining_usd"] <= 3 - 60 * A10G_SMALL / 1e6
            finally:
                release.set()
            return [result.data for result in await asyncio.gather(*calls)]

    results = asyncio.run(exercise())
    successes = [r for r in results if "job_id" in r]
    if same_key:
        assert len(successes) == 4
        assert len({r["job_id"] for r in successes}) == 1
        assert len(api.runs) == 1
    else:
        assert len(successes) == 2
        assert len(api.runs) == 2
        assert all(
            r["error"]["type"] == "budget_exhausted" for r in results if "error" in r
        )
    assert env._broker.remaining_micro_usd() >= 0
    assert (
        env._broker.spend()["reserved"]["usd"] == len(api.runs) * 60 * A10G_SMALL / 1e6
    )


def test_cancelled_finish_cannot_end_reset_episode(api, monkeypatch):
    env = PostTrainEnvironment(api=api, max_steps=2, poll_s=0, cancel_timeout_s=0.01)
    env.reset(task_id="t0_artifact")
    submit_job(env)
    old_broker = env._broker
    entered = threading.Event()
    release = threading.Event()
    cancel_all = old_broker.cancel_all
    first = True
    worker_finished = threading.Event()
    finish = env._finish

    def blocked_cancel(*args):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(5)
        return cancel_all(*args)

    def finish_and_signal(*args):
        try:
            return finish(*args)
        finally:
            worker_finished.set()

    monkeypatch.setattr(old_broker, "cancel_all", blocked_cancel)
    monkeypatch.setattr(env, "_finish", finish_and_signal)

    async def exercise():
        step = asyncio.create_task(
            env.step_async(CallToolAction(tool_name="budget", arguments={}))
        )
        assert await asyncio.to_thread(entered.wait, 5)
        step.cancel()
        with pytest.raises(asyncio.CancelledError):
            await step
        reset = asyncio.create_task(
            asyncio.to_thread(env.reset, task_id="t0_artifact", episode_id="new")
        )
        # Let reset finish if it can overtake the old completion. With the
        # lifecycle lock it instead waits until we release that completion.
        await asyncio.wait({reset}, timeout=0.05)
        release.set()
        obs = await reset
        assert await asyncio.to_thread(worker_finished.wait, 5)
        assert not obs.done
        assert env.state.episode_id == "new"
        assert not env._done
        assert env._end == {}

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        env.close()


def test_session_cleanup_retry_keeps_event_loop_responsive(api, monkeypatch):
    from openenv.core.env_server.http_server import HTTPEnvServer

    env = PostTrainEnvironment(api=api, poll_s=0.01, cancel_timeout_s=0.1)
    env.reset(task_id="t0_artifact")
    job = submit_job(env)
    api.cancel_confirms = False
    server = object.__new__(HTTPEnvServer)
    server._shared_session_executor = None
    loop_calls = []
    cancel = api.cancel_job

    def record_cancel(**kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            loop_calls.append(kwargs["job_id"])
        return cancel(**kwargs)

    monkeypatch.setattr(api, "cancel_job", record_cancel)

    async def exercise():
        with ThreadPoolExecutor(max_workers=1) as executor:
            await server._cleanup_session_resources(env, executor)
        assert not loop_calls
        # If fallback scheduled another attempt, it remains owned until done.
        if env._close_tasks:
            await asyncio.gather(*env._close_tasks.values())
        assert env._broker.jobs[job["job_id"]].state == "orphaned"
        api.cancel_confirms = True
        await asyncio.to_thread(env.close)
        assert env._broker.jobs[job["job_id"]].state == "cancelled"

    try:
        asyncio.run(exercise())
    finally:
        api.cancel_confirms = True
        env.close()


@pytest.mark.parametrize("role", ["agent", "dev_eval"])
def test_pending_admission_enforces_job_limits(env, api, monkeypatch, role):
    from envs.posttrain_env.server.broker import BrokerError

    env.reset(task_id="t0_artifact")
    source = env._broker.get(submit_job(env)["job_id"])
    api.finish(source.job_id)
    env._broker.refresh(source)
    broker = env._broker
    broker.max_jobs = 2  # The source consumes the first slot.
    broker.max_dev_evals = 1
    barrier = threading.Barrier(4)
    entered = threading.Event()
    release = threading.Event()
    api_lock = threading.Lock()
    run = api.run_uv_job

    def blocked_run(**kwargs):
        entered.set()
        assert release.wait(5)
        with api_lock:
            return run(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", blocked_run)

    def start():
        barrier.wait(5)
        try:
            if role == "agent":
                return broker.submit(
                    "print(1)",
                    flavor="cpu-basic",
                    timeout_minutes=1,
                    allowed_flavors=("cpu-basic",),
                    max_timeout_minutes=1,
                )
            return broker.evaluate(
                source,
                role=role,
                script_path=env._task.eval_script_path,
                flavor="cpu-basic",
                timeout_minutes=1,
                env={},
            )
        except BrokerError as error:
            return error.kind

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(start) for _ in range(4)]
        try:
            assert entered.wait(5)
            done, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
            assert done, (
                "Other admissions must reject while the provider call is pending"
            )
        finally:
            release.set()
        results = [f.result(5) for f in futures]
    assert results.count("limit_reached") == 3
    assert len(api.runs) == 2


def test_failed_admission_releases_only_its_own_reservation(env, api, monkeypatch):
    env.reset(task_id="t0_artifact")
    broker = env._broker
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    api_lock = threading.Lock()
    run = api.run_uv_job
    attempts = 0

    def failing_run(**kwargs):
        nonlocal attempts
        with api_lock:
            index = attempts
            attempts += 1
        if index < 2:
            entered[index].set()
            assert release[index].wait(5)
        if index == 0:
            raise ConnectionError("failed before creating job")
        with api_lock:
            return run(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", failing_run)

    async def exercise():
        async with env.mcp_session() as client:

            def start(key):
                return client.call_tool(
                    "jobs_submit",
                    {
                        "script": "print(1)",
                        "flavor": "cpu-basic",
                        "timeout_minutes": 10,
                        "idempotency_key": key,
                    },
                )

            first = asyncio.create_task(start("retry"))
            assert await asyncio.to_thread(entered[0].wait, 5)
            second = asyncio.create_task(start("other"))
            try:
                assert await asyncio.to_thread(entered[1].wait, 5)
                release[0].set()
                failed = await first
                assert failed.data["error"]["type"] == "internal_error"
                assert broker.remaining_micro_usd() == 50_000 - 10 * CPU_BASIC
                retry = await start("retry")
                assert "job_id" in retry.data
                assert broker.remaining_micro_usd() == 50_000 - 20 * CPU_BASIC
            finally:
                for event in release:
                    event.set()
                await asyncio.gather(first, second)
        assert len(api.runs) == 2
        assert broker.spend()["reserved"]["usd"] == 20 * CPU_BASIC / 1e6

    asyncio.run(exercise())


def test_finish_queued_before_reset_keeps_episode_ownership(api, monkeypatch):
    env = PostTrainEnvironment(api=api, max_steps=1)
    env.reset(task_id="t0_artifact")
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    finish = env._finish

    def queued_finish(*args):
        entered.set()
        assert release.wait(5)
        try:
            return finish(*args)
        finally:
            completed.set()

    monkeypatch.setattr(env, "_finish", queued_finish)

    async def exercise():
        step = asyncio.create_task(
            env.step_async(CallToolAction(tool_name="budget", arguments={}))
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            step.cancel()
            with pytest.raises(asyncio.CancelledError):
                await step
            await asyncio.to_thread(env.reset, task_id="t0_artifact", episode_id="new")
            # New episode activity must never be completed by the old worker.
            env.state.step_count = 1
        finally:
            release.set()
        assert await asyncio.to_thread(completed.wait, 5)
        assert not env._done
        assert env._end == {}
        assert env.state.episode_id == "new"

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        env.close()


def test_close_after_reset_runs_while_old_close_is_pending(api, monkeypatch, caplog):
    env = PostTrainEnvironment(api=api, poll_s=0, cancel_timeout_s=0)
    env.reset(task_id="t0_artifact")
    old_job = submit_job(env)
    old_broker = env._broker
    entered = threading.Event()
    release = threading.Event()
    close = env._close
    closed_brokers = []

    def queued_close(broker):
        closed_brokers.append(broker)
        if broker is old_broker:
            entered.set()
            assert release.wait(5)
        close(broker)

    monkeypatch.setattr(env, "_close", queued_close)

    async def exercise():
        env.close()
        old_owner = env._close_tasks[old_broker]
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await asyncio.to_thread(env.reset, task_id="t0_artifact", episode_id="new")
            new_broker = env._broker
            new_job = await asyncio.to_thread(submit_job, env)
            api.cancel_confirms = False
            env.close()
            new_owner = env._close_tasks[new_broker]
            assert new_owner is not old_owner
            env.close()
            assert env._close_tasks[new_broker] is new_owner
            await asyncio.wait_for(asyncio.shield(new_owner), 3)
            assert not old_owner.done()
            assert new_broker not in env._close_tasks
            assert old_broker in env._close_tasks
            assert new_broker.jobs[new_job["job_id"]].state == "orphaned"
            assert not new_broker.spend()["final"]
            assert "close could not confirm job cancellation" in caplog.text

            api.cancel_confirms = True
            env.close()
            retry = env._close_tasks[new_broker]
            assert retry is not new_owner
            await asyncio.wait_for(asyncio.shield(retry), 3)
            assert not old_owner.done()
            assert api.jobs[new_job["job_id"]]["stage"] == "CANCELED"
            assert new_broker.spend()["final"]
            assert closed_brokers == [old_broker, new_broker, new_broker]
        finally:
            api.cancel_confirms = True
            release.set()
            await old_owner
        assert api.jobs[old_job["job_id"]]["stage"] == "CANCELED"
        assert not env._close_tasks
        assert old_owner not in posttrain_environment._CLOSE_TASKS
        assert new_owner not in posttrain_environment._CLOSE_TASKS
        assert retry not in posttrain_environment._CLOSE_TASKS

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        api.cancel_confirms = True
        close(env._broker)


def test_deferred_close_retains_the_original_broker(api, monkeypatch):
    env = PostTrainEnvironment(api=api)
    env.reset(task_id="t0_artifact")
    old_job = submit_job(env)
    entered = threading.Event()
    release = threading.Event()
    close = env._close

    def queued_close(broker):
        entered.set()
        assert release.wait(5)
        close(broker)

    monkeypatch.setattr(env, "_close", queued_close)

    async def exercise():
        env.close()
        owner = env._close_tasks[env._broker]
        env.close()
        assert env._close_tasks[env._broker] is owner
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await asyncio.to_thread(env.reset, task_id="t0_artifact", episode_id="new")
            new_job = await asyncio.to_thread(submit_job, env)
        finally:
            release.set()
        await owner
        assert api.jobs[old_job["job_id"]]["stage"] == "CANCELED"
        assert api.jobs[new_job["job_id"]]["stage"] == "SCHEDULING"
        assert env.mcp_server is not None

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        env.close()


@pytest.mark.parametrize("pause_at", ["refresh", "evaluate", "wait", "cancel_all"])
def test_cancelled_submit_cannot_touch_reset_episode(env, api, monkeypatch, pause_at):
    env.reset(task_id="t0_artifact")
    source = submit_job(env)
    api.finish(source["job_id"])
    old_broker = env._broker
    if pause_at == "cancel_all":
        api.on_inspect["score"] = {
            "stage": "COMPLETED",
            "running_secs": 1,
            "logs": [score_line(1.0)],
        }
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    original = getattr(old_broker, pause_at)
    submit = env._submit
    first = True

    def blocked(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    def signalled_submit(*args):
        try:
            return submit(*args)
        finally:
            completed.set()

    monkeypatch.setattr(old_broker, pause_at, blocked)
    monkeypatch.setattr(env, "_submit", signalled_submit)

    async def exercise():
        async with env.mcp_session():
            request = asyncio.create_task(
                env.step_async(
                    CallToolAction(
                        tool_name="submit", arguments={"job_id": source["job_id"]}
                    )
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                await asyncio.to_thread(
                    env.reset, task_id="t0_artifact", episode_id="new"
                )
                new_broker = env._broker
                new_job = await env.step_async(
                    CallToolAction(
                        tool_name="jobs_submit",
                        arguments={
                            "script": "print(1)",
                            "flavor": "cpu-basic",
                            "timeout_minutes": 1,
                        },
                    )
                )
                job_id = new_job.result.data["job_id"]
            finally:
                release.set()
            assert await asyncio.to_thread(completed.wait, 5)
            assert env._broker is new_broker
            assert env.state.episode_id == "new"
            assert not env._done
            assert env._reward is None
            assert env._end == {}
            assert env._score_job_id is None
            assert not new_broker._closing
            assert api.jobs[job_id]["stage"] == "SCHEDULING"
            assert not api.jobs[job_id].get("cancel_requested")

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_uncertain_creation_keeps_ownership_until_reconciliation(env, api, monkeypatch):
    from envs.posttrain_env.server.broker import BrokerError

    env.reset(task_id="t0_arithmetic_format")
    broker = env._broker
    list_jobs = api.list_jobs
    api.fail_after_create = True

    def unavailable(**kwargs):
        raise ConnectionError("reconciliation unavailable")

    monkeypatch.setattr(api, "list_jobs", unavailable)
    arguments = dict(
        flavor="a10g-small",
        timeout_minutes=60,
        allowed_flavors=("a10g-small",),
        max_timeout_minutes=60,
        idempotency_key="same",
    )
    with pytest.raises(ConnectionError):
        broker.submit("print(1)", **arguments)
    reservation = 60 * A10G_SMALL
    assert broker.remaining_micro_usd() == 3_000_000 - reservation
    assert broker.spend()["reserved"]["usd"] == reservation / 1e6
    assert not broker.spend()["final"]
    assert broker.count("agent") == 1
    for _ in range(2):
        with pytest.raises(BrokerError, match="unconfirmed"):
            broker.submit("print(1)", **arguments)
    assert len(api.runs) == 1
    # The unknown job's reservation still constrains other keys.
    broker.submit("print(2)", **{**arguments, "idempotency_key": "other"})
    with pytest.raises(BrokerError) as error:
        broker.submit("print(3)", **{**arguments, "idempotency_key": "third"})
    assert error.value.kind == "budget_exhausted"
    assert broker.spend()["reserved"]["usd"] == 2 * reservation / 1e6
    for cleanup in (env.close, lambda: env.reset(task_id="t0_artifact")):
        with pytest.raises(RuntimeError, match="did not confirm cancelling"):
            cleanup()
        assert env._broker is broker
        assert broker._pending
        assert not broker.spend()["final"]
    assert api.jobs["job0"]["stage"] == "SCHEDULING"
    assert api.jobs["job1"]["stage"] == "CANCELED"
    monkeypatch.setattr(api, "list_jobs", list_jobs)
    # Once closing starts, cleanup owns recovery and cancels the recovered id.
    with pytest.raises(BrokerError, match="unconfirmed"):
        broker.submit("print(1)", **arguments)
    env.close()
    assert "job0" in broker.jobs
    assert len(api.runs) == 2
    assert all(job["stage"] == "CANCELED" for job in api.jobs.values())
    assert not broker._pending
    assert broker.spend()["final"]


@pytest.mark.parametrize("found", [False, True])
def test_uncertain_creation_retry_recovers_without_duplicate(
    env, api, monkeypatch, found
):
    from envs.posttrain_env.server.broker import BrokerError

    env.reset(task_id="t0_arithmetic_format")
    broker = env._broker
    run = api.run_uv_job
    list_jobs = api.list_jobs
    lookups = []

    def lost_response(**kwargs):
        if found:
            run(**kwargs)
        raise ConnectionError("response lost")

    def unavailable(**kwargs):
        lookups.append(kwargs["labels"]["openenv-request"])
        raise ConnectionError("list unavailable")

    monkeypatch.setattr(api, "run_uv_job", lost_response)
    monkeypatch.setattr(api, "list_jobs", unavailable)
    arguments = dict(
        flavor="a10g-small",
        timeout_minutes=60,
        allowed_flavors=("a10g-small",),
        max_timeout_minutes=60,
        idempotency_key="retry",
    )
    with pytest.raises(ConnectionError):
        broker.submit("print(1)", **arguments)
    with pytest.raises(BrokerError, match="unconfirmed"):
        broker.submit("print(1)", **arguments)
    assert lookups == [lookups[0], lookups[0]]
    assert list(broker._pending) == [lookups[0]]
    monkeypatch.setattr(api, "run_uv_job", run)
    monkeypatch.setattr(api, "list_jobs", list_jobs)

    async def exercise():
        async with env.mcp_session() as client:
            replies = await asyncio.gather(
                *(
                    client.call_tool(
                        "jobs_submit",
                        {
                            "script": "print(1)",
                            "flavor": "a10g-small",
                            "timeout_minutes": 60,
                            "idempotency_key": "retry",
                        },
                    )
                    for _ in range(4)
                )
            )
            assert all(reply.data["job_id"] == "job0" for reply in replies)

    asyncio.run(exercise())
    assert len(api.runs) == 1
    assert not broker._pending
    assert broker.count("agent") == 1
    assert broker.spend()["reserved"]["usd"] == 60 * A10G_SMALL / 1e6
    assert broker.remaining_micro_usd() == 3_000_000 - 60 * A10G_SMALL


@pytest.mark.parametrize(
    "cleanup", ["close", "reset", "max_steps", "time_limit", "duplicate_key"]
)
def test_hung_creation_waits_have_deadlines(env, api, monkeypatch, cleanup):
    from envs.posttrain_env.server import broker as broker_module
    from envs.posttrain_env.server.broker import BrokerError

    env.reset(task_id="t0_artifact")
    broker = env._broker
    monkeypatch.setattr(broker_module, "ADMISSION_TIMEOUT_S", 0.02, raising=False)
    entered = threading.Event()
    release = threading.Event()
    run = api.run_uv_job

    def blocked_run(**kwargs):
        entered.set()
        assert release.wait(5)
        return run(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", blocked_run)
    arguments = dict(
        flavor="cpu-basic",
        timeout_minutes=10,
        allowed_flavors=("cpu-basic",),
        max_timeout_minutes=10,
        idempotency_key="hung",
    )

    def operation():
        if cleanup == "duplicate_key":
            with pytest.raises(BrokerError) as error:
                broker.submit("print(1)", **arguments)
            assert error.value.kind == "creation_pending"
        elif cleanup in ("max_steps", "time_limit"):
            if cleanup == "max_steps":
                env._max_steps = 1
            else:
                env._started_at -= 60 * env._task.max_episode_minutes
            obs = env.step(CallToolAction(tool_name="budget", arguments={}))
            assert obs.done
            assert obs.metadata["episode_end"]["orphaned_jobs"]
            assert not obs.metadata["service_spend"]["final"]
        else:
            with pytest.raises(RuntimeError, match="did not confirm cancelling"):
                if cleanup == "close":
                    env.close()
                else:
                    env.reset(task_id="t0_artifact", episode_id="new")

    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(broker.submit, "print(1)", **arguments)
        try:
            assert entered.wait(5)
            # The worker stays blocked until all deadline assertions are done.
            attempt = pool.submit(operation)
            attempt.result(timeout=0.5)
            assert not release.is_set()
            assert not submission.done()
            assert env._broker is broker
            assert broker.spend()["reserved"]["usd"] == 10 * CPU_BASIC / 1e6
            assert not broker.spend()["final"]
            assert broker.count("agent") == 1
        finally:
            release.set()
        job = submission.result(timeout=5)
    assert len(api.runs) == 1
    if cleanup != "duplicate_key":
        assert api.jobs[job.job_id]["stage"] == "CANCELED"
    env.close()
    assert broker.spend()["final"]


def test_hung_score_creation_does_not_hold_lifecycle_lock(env, api, monkeypatch):
    env.reset(task_id="t0_artifact")
    source = submit_job(env)
    api.finish(source["job_id"])
    broker = env._broker
    entered = threading.Event()
    release = threading.Event()
    run = api.run_uv_job

    def blocked_score(**kwargs):
        if kwargs["labels"]["openenv-role"] == "score":
            entered.set()
            assert release.wait(5)
        return run(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", blocked_score)
    with ThreadPoolExecutor(max_workers=2) as pool:
        scoring = pool.submit(env._submit, source["job_id"])
        try:
            assert entered.wait(5)

            def cleanup():
                with pytest.raises(RuntimeError, match="did not confirm cancelling"):
                    env.close()

            pool.submit(cleanup).result(timeout=0.5)
            assert not scoring.done()
            assert not broker.spend()["final"]
            assert broker.count("score") == 1
        finally:
            release.set()
        scoring.result(timeout=5)
    assert len(api.runs) == 2
    assert api.jobs["job1"]["stage"] == "CANCELED"
    assert broker.spend()["final"]


def test_cleanup_reports_job_published_after_its_live_snapshot(env, api, monkeypatch):
    env.reset(task_id="t0_artifact")
    submit_job(env)
    broker = env._broker
    creating = threading.Event()
    release_creation = threading.Event()
    cancelling = [threading.Event(), threading.Event()]
    release_cancel = [threading.Event(), threading.Event()]
    run = api.run_uv_job
    cancel = api.cancel_job

    def blocked_run(**kwargs):
        creating.set()
        assert release_creation.wait(5)
        return run(**kwargs)

    def blocked_cancel(**kwargs):
        index = int(kwargs["job_id"][-1])
        cancelling[index].set()
        assert release_cancel[index].wait(5)
        return cancel(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", blocked_run)
    monkeypatch.setattr(api, "cancel_job", blocked_cancel)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(
            broker.submit,
            "print(1)",
            flavor="cpu-basic",
            timeout_minutes=1,
            allowed_flavors=("cpu-basic",),
            max_timeout_minutes=1,
        )
        try:
            assert creating.wait(5)
            closing = pool.submit(env.close)
            # Cleanup has snapshotted job0, then the late worker publishes job1.
            assert cancelling[0].wait(5)
            release_creation.set()
            assert cancelling[1].wait(5)
            release_cancel[0].set()
            with pytest.raises(RuntimeError, match="job1"):
                closing.result(timeout=0.5)
            assert not broker.spend()["final"]
        finally:
            release_creation.set()
            for event in release_cancel:
                event.set()
        submission.result(timeout=5)
    env.close()
    assert broker.spend()["final"]


@pytest.mark.parametrize("source", ["readme", "client_docstring"])
def test_documented_client_episode_returns_reward_and_spend(
    env, api, monkeypatch, tmp_path, source
):
    """Execute the examples across server serialization and client parsing."""
    from contextlib import nullcontext

    import posttrain_env
    from openenv.core.mcp_client import MCPToolClient

    if source == "readme":
        text = (ENV_DIR / "README.md").read_text().split("## Client", 1)[1]
    else:
        text = posttrain_env.PostTrainEnv.__doc__
    code = re.search(r"```python\n(.*?)\n\s*```", text, re.S).group(1)
    # The docstring is indented, while the README fence is not.
    import textwrap

    code = textwrap.dedent(code)
    api.on_inspect.update(
        agent={"stage": "COMPLETED", "running_secs": 90},
        score={
            "stage": "COMPLETED",
            "running_secs": 10,
            "logs": [score_line(1.0)],
        },
    )
    parser = MCPToolClient(base_url="http://localhost:8000")
    results = []

    def wire_result(obs):
        payload = {
            "observation": obs.model_dump(mode="json"),
            "reward": obs.reward,
            "done": obs.done,
        }
        return parser._parse_result(json.loads(json.dumps(payload)))

    def step(action):
        result = wire_result(env.step(action))
        results.append(result)
        return result

    client = SimpleNamespace(
        reset=lambda **kwargs: wire_result(env.reset(**kwargs)), step=step
    )
    monkeypatch.setattr(
        posttrain_env,
        "PostTrainEnv",
        lambda **kwargs: SimpleNamespace(sync=lambda: nullcontext(client)),
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('train')")
    exec(compile(code, source, "exec"), {"script": "print('train')"})
    assert env.state.step_count == 3
    final = results[-1]
    assert final.done is True
    assert final.reward == 1.0
    spend = final.observation.metadata["service_spend"]
    assert spend["final"] is True
    assert spend["settled"]["usd"] == pytest.approx(2 * CPU_BASIC / 1e6)
    assert spend["settled"]["jobs"] == 1
    assert spend["reserved"]["usd"] == 0


def test_local_server_binds_only_loopback(monkeypatch):
    import uvicorn
    from envs.posttrain_env.server import app

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    app.main()
    assert calls == [{"host": "127.0.0.1", "port": 8000}]


@pytest.mark.parametrize("lost_response", [False, True])
def test_score_admission_fastmcp(env, api, monkeypatch, lost_response):
    env.reset(task_id="t0_artifact")
    source = submit_job(env)["job_id"]
    api.finish(source)
    api.on_inspect["score"] = {
        "stage": "COMPLETED",
        "running_secs": 1,
        "logs": [score_line(0.75)],
    }
    entered, release = threading.Event(), threading.Event()
    run, lookup = api.run_uv_job, api.list_jobs
    creates = []
    second_entered = threading.Event()
    submit = env._submit
    submit_count = 0

    def counted_submit(*args):
        nonlocal submit_count
        submit_count += 1
        if submit_count == 2:
            second_entered.set()
        return submit(*args)

    monkeypatch.setattr(env, "_submit", counted_submit)

    def blocked_run(**kwargs):
        if kwargs["labels"]["openenv-role"] == "score":
            creates.append(kwargs)
            entered.set()
            assert release.wait(5)
        return run(**kwargs)

    monkeypatch.setattr(api, "run_uv_job", blocked_run)

    async def exercise():
        async with env.mcp_session() as client:
            if lost_response:
                api.fail_after_create = True

                def unavailable(**kwargs):
                    raise ConnectionError("lookup unavailable")

                monkeypatch.setattr(api, "list_jobs", unavailable)
                release.set()
                failed = await client.call_tool("submit", {"job_id": source})
                assert "error" in failed.data
                for _ in range(2):
                    failed = await client.call_tool("submit", {"job_id": source})
                    assert "error" in failed.data
                assert len(creates) == 1
                assert len(env._broker._pending) == 1
                monkeypatch.setattr(api, "list_jobs", lookup)
                results = [await client.call_tool("submit", {"job_id": source})]
            else:
                first = asyncio.create_task(
                    client.call_tool("submit", {"job_id": source})
                )
                assert await asyncio.to_thread(entered.wait, 5)
                second = asyncio.create_task(
                    client.call_tool("submit", {"job_id": source})
                )
                assert await asyncio.to_thread(second_entered.wait, 5)
                await client.call_tool("budget", {})
                release.set()
                results = await asyncio.gather(first, second)
            assert all(r.data.get("result", {}).get("score") == 0.75 for r in results)

    try:
        asyncio.run(exercise())
        assert len(creates) == 1
        assert len(api.runs) == 2
        assert not api.jobs["job1"].get("cancel_requested")
        assert env._reward == 0.75
        assert env._end["reason"] == "submitted"
        assert env._broker.spend()["settled"]["jobs"] == 1
    finally:
        release.set()
        monkeypatch.setattr(api, "list_jobs", lookup)


@pytest.mark.parametrize("preopen", [False, True])
@pytest.mark.parametrize("old_tool", ["budget", "jobs_submit"])
def test_fresh_episode_dispatch_during_old_request(
    env, api, monkeypatch, preopen, old_tool
):
    env.reset(task_id="t0_artifact", episode_id="old")
    old = env._broker
    entered, release = threading.Event(), threading.Event()
    original = env._call

    def blocked(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(env, "_call", blocked)

    async def exercise():
        args = {"script": "print(1)", "flavor": "cpu-basic", "timeout_minutes": 1}
        async with AsyncExitStack() as stack:
            if preopen:
                await stack.enter_async_context(env.mcp_session())
            request = asyncio.create_task(
                env.step_async(
                    CallToolAction(
                        tool_name=old_tool,
                        arguments={} if old_tool == "budget" else args,
                    )
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                await asyncio.to_thread(
                    env.reset, task_id="t0_artifact", episode_id="new"
                )
                new = env._broker
                fresh = await env.step_async(
                    CallToolAction(tool_name="jobs_submit", arguments=args)
                )
            finally:
                release.set()
                stale = await request
            assert fresh.error is None
            assert fresh.result.data["job_id"] in new.jobs
            assert stale.error is None
            assert stale.result.data["error"]["type"] == "episode_over"
            assert env._broker is new and new is not old
            assert not old.jobs and len(new.jobs) == 1 and len(api.runs) == 1
            assert env.state.step_count == 1
            assert not new._closing and not env._done and env._end == {}

    try:
        asyncio.run(exercise())
    finally:
        release.set()


@pytest.mark.parametrize("tool", ["jobs_submit", "dev_evaluate"])
@pytest.mark.parametrize("pause_at", ["call", "dispatch"])
def test_tool_dispatch_keeps_episode_ownership(env, api, monkeypatch, tool, pause_at):
    env.reset(task_id="t0_artifact", episode_id="old")
    source = submit_job(env)["job_id"]
    api.finish(source)
    old = env._broker
    entered, release = threading.Event(), threading.Event()
    original = env._call if pause_at == "call" else env._async_call_tool

    if pause_at == "call":

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(env, "_call", blocked)
    else:

        async def blocked(*args, **kwargs):
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)
            return await original(*args, **kwargs)

        monkeypatch.setattr(env, "_async_call_tool", blocked)

    async def exercise():
        async with env.mcp_session():
            args = (
                {"job_id": source}
                if tool == "dev_evaluate"
                else {"script": "print(1)", "flavor": "cpu-basic", "timeout_minutes": 1}
            )
            request = asyncio.create_task(
                env.step_async(CallToolAction(tool_name=tool, arguments=args))
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                await asyncio.to_thread(
                    env.reset, task_id="t0_arithmetic_format", episode_id="new"
                )
                new = env._broker
            finally:
                release.set()
            obs = await request
            assert obs.result.data["error"]["type"] == "episode_over"
            assert env._broker is new and new is not old
            assert not new.jobs and not new._pending and not new._closing
            assert new.remaining_micro_usd() == 3_000_000
            assert env.state.step_count == 0
            assert env._score_job_id is None and not env._done and env._end == {}
            assert len(api.runs) == 1

    try:
        asyncio.run(exercise())
    finally:
        release.set()


@pytest.mark.parametrize("created", [False, True])
def test_score_recovery_pins_source_and_key(env, api, monkeypatch, created):
    env.reset(task_id="t0_artifact")
    first = submit_job(env, idempotency_key="score")["job_id"]
    other = submit_job(env)["job_id"]
    api.finish(first)
    api.finish(other)
    run, lookup = api.run_uv_job, api.list_jobs

    def uncertain(**kwargs):
        if created:
            run(**kwargs)
        raise ConnectionError("create response lost")

    def unavailable(**kwargs):
        raise ConnectionError("lookup unavailable")

    monkeypatch.setattr(api, "run_uv_job", uncertain)
    monkeypatch.setattr(api, "list_jobs", unavailable)
    failed, _ = call(env, "submit", job_id=first)
    assert "error" in failed
    pending = list(env._broker._pending)
    for _ in range(2):
        failed, _ = call(env, "submit", job_id=other)
        assert failed["error"]["type"] == "creation_pending"
        assert list(env._broker._pending) == pending
        assert len(api.runs) == 2 + int(created)
    monkeypatch.setattr(api, "run_uv_job", run)
    monkeypatch.setattr(api, "list_jobs", lookup)
    api.on_inspect["score"] = {
        "stage": "COMPLETED",
        "running_secs": 1,
        "logs": [score_line(0.5)],
    }
    result, obs = call(env, "submit", job_id=other)
    assert result["submitted"] == first
    assert obs.reward == 0.5
    assert len(api.runs) == 3 and not env._broker._pending
    assert not api.jobs["job2"].get("cancel_requested")
    assert api.runs[-1]["volumes"][0].path == "run0/outputs"


def test_queued_old_submit_cannot_cancel_new_score(env, api, monkeypatch):
    env.reset(task_id="t0_artifact", episode_id="old")
    source = submit_job(env)["job_id"]
    api.finish(source)
    entered, release = threading.Event(), threading.Event()
    original = env._async_call_tool

    async def blocked(*args, **kwargs):
        entered.set()
        assert await asyncio.to_thread(release.wait, 5)
        return await original(*args, **kwargs)

    monkeypatch.setattr(env, "_async_call_tool", blocked)

    async def exercise():
        async with env.mcp_session():
            request = asyncio.create_task(
                env.step_async(
                    CallToolAction(tool_name="submit", arguments={"job_id": source})
                )
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                await asyncio.to_thread(
                    env.reset, task_id="t0_artifact", episode_id="new"
                )
                broker = env._broker
                job = await asyncio.to_thread(
                    broker.submit,
                    "print(1)",
                    flavor="cpu-basic",
                    timeout_minutes=1,
                    allowed_flavors=("cpu-basic",),
                    max_timeout_minutes=1,
                )
                api.finish(job.job_id)
                score = await asyncio.to_thread(env._evaluate, job.job_id, "score")
                spend = broker.spend()
            finally:
                release.set()
            obs = await request
            assert obs.result.data["error"]["type"] == "episode_over"
            assert broker.spend() == spend
            assert not broker._closing and not env._done
            assert not api.jobs[score.job_id].get("cancel_requested")
            assert len(api.runs) == 3

    try:
        asyncio.run(exercise())
    finally:
        release.set()


@pytest.mark.parametrize("tool", ["jobs_submit", "dev_evaluate"])
def test_tool_result_keeps_old_budget_after_reset(env, api, monkeypatch, tool):
    env.reset(task_id="t0_artifact", episode_id="old")
    source = submit_job(env)["job_id"]
    api.finish(source)
    old = env._broker
    entered, release = threading.Event(), threading.Event()
    view = env._job_view

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return view(*args, **kwargs)

    monkeypatch.setattr(env, "_job_view", blocked)

    async def exercise():
        async with env.mcp_session():
            args = (
                {"job_id": source}
                if tool == "dev_evaluate"
                else {"script": "print(1)", "flavor": "cpu-basic", "timeout_minutes": 1}
            )
            request = asyncio.create_task(
                env.step_async(CallToolAction(tool_name=tool, arguments=args))
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                await asyncio.to_thread(
                    env.reset, task_id="t0_arithmetic_format", episode_id="new"
                )
            finally:
                release.set()
            obs = await request
            assert obs.result.data["remaining_usd"] == old.remaining_micro_usd() / 1e6
            assert obs.result.data["remaining_usd"] != 3.0
            assert not env._broker.jobs and not env._broker._closing
            assert env.state.step_count == 0 and not env._done

    try:
        asyncio.run(exercise())
    finally:
        release.set()
