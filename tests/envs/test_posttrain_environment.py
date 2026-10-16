# SPDX-License-Identifier: BSD-3-Clause

"""Tests for posttrain_env: the HF Jobs broker, its budget ledger and the episode flow.

HF Jobs is replaced by an in-memory fake, so these tests make no network calls
and spend nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from envs.posttrain_env.server import posttrain_environment
from envs.posttrain_env.server.broker import MAX_RESULT_READS, SCORE_MARKER
from envs.posttrain_env.server.posttrain_environment import PostTrainEnvironment
from huggingface_hub import Volume
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

ENV_DIR = Path(__file__).parents[2] / "envs" / "posttrain_env"
CPU_BASIC = 167  # micro-USD per minute
A10G_SMALL = 16_667


def hardware(name, micro_usd_per_minute, gpus=None):
    accelerator = None if gpus is None else SimpleNamespace(quantity=str(gpus))
    return SimpleNamespace(
        name=name, unit_cost_micro_usd=micro_usd_per_minute, accelerator=accelerator
    )


class FakeHfApi:
    """The subset of `HfApi` the broker uses, backed by a dict of jobs."""

    def __init__(self):
        self.jobs = {}
        self.runs = []
        self.cancel_confirms = True
        # Applied the first time a job with this role label is inspected.
        self.on_inspect = {}

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
        return SimpleNamespace(
            id=job_id, volumes=(kwargs["volumes"] or []) + [artifacts]
        )

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


def submit_job(env, flavor="cpu-basic", timeout_minutes=10, **extra):
    data, _ = call(
        env,
        "jobs_submit",
        script="print('train')",
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
        assert run["dependencies"] == ["numpy"]
        assert run["labels"] == {
            "openenv-env": "posttrain_env",
            "openenv-episode": "ep1",
            "openenv-role": "agent",
        }
        assert run["script"] == "print('train')"
        assert job["state"] == "queued"
        assert job["remaining_usd"] == pytest.approx(0.05 - 10 * CPU_BASIC / 1e6)

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
        assert run["dependencies"] == ["torch", "transformers"]
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
    def test_missing_or_out_of_range_scores_count_as_invalid(self, env, api, logs):
        env.reset(task_id="t0_artifact")
        job = self._finished_job(env, api)
        evaluation, _ = call(env, "dev_evaluate", job_id=job["job_id"])
        api.finish(evaluation["job_id"], logs=logs)
        # A missing score line is re-read a few times in case the logs lag.
        for _ in range(MAX_RESULT_READS):
            data, _ = call(env, "jobs_status", job_id=evaluation["job_id"])
        assert data["result"]["status"] == "invalid"
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
        assert score_run["env"]["POSTTRAIN_SEED"] == str(
            posttrain_environment.TEST_SEED
        )
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

    def test_failed_score_job_gives_zero_reward(self, env, api):
        env.reset(task_id="t0_artifact")
        job = self._finished(env, api)
        api.on_inspect["score"] = {"stage": "ERROR", "running_secs": 10, "logs": []}
        data, obs = call(env, "submit", job_id=job["job_id"])
        assert data["result"] == {"status": "failed", "score": 0.0}
        assert obs.reward == 0.0

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

    def test_a_failing_cancel_call_does_not_stop_the_others(
        self, env, api, monkeypatch
    ):
        env.reset(task_id="t0_artifact")
        first, second = submit_job(env), submit_job(env)
        cancel = api.cancel_job

        def flaky_cancel(*, job_id, namespace):
            if job_id == first["job_id"]:
                raise http_error(409)
            cancel(job_id=job_id, namespace=namespace)

        monkeypatch.setattr(api, "cancel_job", flaky_cancel)
        with pytest.raises(RuntimeError, match=first["job_id"]):
            env.reset(task_id="t0_artifact")
        assert api.jobs[second["job_id"]]["stage"] == "CANCELED"

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
