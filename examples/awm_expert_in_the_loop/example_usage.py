"""Minimal example: drive a single AWM task by hand, without any LLM.

This shows the raw environment API the expert-in-the-loop recipe is built on:
``reset`` -> ``list_tools`` -> ``call_tool`` -> ``verify`` -> ``done``. No model
credentials are required.

```bash
PYTHONPATH=src:envs uv run uvicorn \\
    envs.agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899 &
PYTHONPATH=src:envs uv run python examples/awm_expert_in_the_loop/example_usage.py
```
"""

from __future__ import annotations

import argparse
import asyncio
import os

from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


async def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal AWM env walkthrough (no LLM)")
    parser.add_argument("scenario", nargs="?", default="marketplace_1")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument(
        "--base-url", default=os.environ.get("AWM_BASE_URL", "http://127.0.0.1:8899")
    )
    args = parser.parse_args()

    async with AWMEnv(base_url=args.base_url) as env:
        reset = await env.reset(scenario=args.scenario, task_idx=args.task_idx)
        print(f"task: {reset.observation.task}")

        tools = (await env.step(ListToolsAction())).observation.tools
        print(
            f"discovered {len(tools)} tools; first few: {[t.name for t in tools[:5]]}"
        )

        profile = (
            await env.step(
                CallToolAction(tool_name="get_current_user_profile", arguments={})
            )
        ).observation
        print(f"current profile: {str(profile.tool_result)[:200]}")

        verify = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": ""},
            )
        )
        print(
            f"verify (no actions taken): reward_type={verify.observation.reward_type} reward={verify.reward}"
        )

        await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": False})
        )


if __name__ == "__main__":
    asyncio.run(main())
