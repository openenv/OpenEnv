"""Run a single AWM task with the agent, optionally with the verifier-informed expert.

Useful for inspecting one episode step by step.

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
PYTHONPATH=src:envs uv run python examples/awm_expert_in_the_loop/run_awm_task.py \\
    marketplace_1 --task-idx 1 --expert --agent-model gpt-5.4-mini
```
"""

from __future__ import annotations

import argparse
import asyncio

from agent_world_model_env import AWMEnv
from expert import VerifierInformedExpert
from policies import OpenAIChatPolicy
from rollout import run_task
from run_benchmark import _build_client, DEFAULT_BASE_URL, DEFAULT_MODEL


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AWM task (verbose)")
    parser.add_argument("scenario")
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument(
        "--expert", action="store_true", help="Enable the ask_expert tool"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--agent-model", default=DEFAULT_MODEL)
    parser.add_argument("--expert-model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iters", type=int, default=15)
    args = parser.parse_args()

    client = _build_client(
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", required=True
    )
    agent_policy = OpenAIChatPolicy(client, args.agent_model)
    expert = VerifierInformedExpert(client, args.expert_model) if args.expert else None

    async with AWMEnv(base_url=args.base_url) as env:
        result = await run_task(
            env,
            agent_policy,
            scenario=args.scenario,
            task_idx=args.task_idx,
            expert=expert,
            max_iters=args.max_iters,
            verbose=True,
        )

    print(
        f"\nresult: reward_type={result.reward_type} reward={result.reward} "
        f"steps={result.steps} errors={result.errors} expert_calls={result.expert_calls}"
    )


if __name__ == "__main__":
    asyncio.run(main())
