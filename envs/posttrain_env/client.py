# SPDX-License-Identifier: BSD-3-Clause

"""Client for the post-training environment."""

from openenv.core.mcp_client import MCPToolClient


class PostTrainEnv(MCPToolClient):
    """
    Client for the post-training environment.

    Dispatch tools with `step(CallToolAction(...))` to count episode steps and
    receive the final reward and `service_spend`. Direct `call_tool` bypasses
    episode step/time limits and does not return that final observation.
    Available tools are: `budget`, `jobs_submit`,
    `jobs_status`, `jobs_wait`, `jobs_logs`, `jobs_cancel`, `dev_evaluate` and
    `submit`. `submit` ends the episode and can take several minutes, so give
    the client a message timeout above the `score_timeout_s` reported by
    `reset`, with room for cancellation and transport overhead (2700 s for
    `t0_arithmetic_format`; the example uses 3600 s).

    Examples:

    ```python
    from openenv.core.env_server.mcp_types import CallToolAction
    from posttrain_env import PostTrainEnv

    with PostTrainEnv(base_url="http://localhost:8000", message_timeout_s=3600).sync() as env:
        result = env.reset(task_id="t0_artifact")
        print(result.observation.metadata["prompt"])
        result = env.step(CallToolAction(tool_name="jobs_submit", arguments={
            "script": script, "flavor": "cpu-basic", "timeout_minutes": 5,
        }))
        assert result.observation.error is None
        job = result.observation.result["data"]
        assert "error" not in job, job
        status = "queued"
        while status in {"queued", "running"}:
            result = env.step(CallToolAction(tool_name="jobs_wait", arguments={
                "job_id": job["job_id"], "timeout_s": 300,
            }))
            assert not result.done, result.observation.metadata
            status = result.observation.result["data"]["state"]
        assert status == "succeeded", result.observation.result
        final = env.step(CallToolAction(tool_name="submit", arguments={
            "job_id": job["job_id"],
        }))
        print(final.reward, final.observation.metadata["service_spend"])
    ```
    """

    pass
