# SPDX-License-Identifier: BSD-3-Clause

"""Client for the post-training environment."""

from openenv.core.mcp_client import MCPToolClient


class PostTrainEnv(MCPToolClient):
    """
    Client for the post-training environment.

    All interaction happens through MCP tools: `budget`, `jobs_submit`,
    `jobs_status`, `jobs_wait`, `jobs_logs`, `jobs_cancel`, `dev_evaluate` and
    `submit`. `submit` ends the episode and can take several minutes, so give
    the client a message timeout of at least the `score_timeout_s` reported by
    `reset`.

    Examples:

    ```python
    from posttrain_env import PostTrainEnv

    with PostTrainEnv(base_url="http://localhost:8000", message_timeout_s=3600).sync() as env:
        result = env.reset(task_id="t0_artifact")
        print(result.observation.metadata["prompt"])
        job = env.call_tool("jobs_submit", script=script, flavor="cpu-basic", timeout_minutes=5)
        env.call_tool("jobs_wait", job_id=job["job_id"], timeout_s=300)
        env.call_tool("submit", job_id=job["job_id"])
    ```
    """

    pass
