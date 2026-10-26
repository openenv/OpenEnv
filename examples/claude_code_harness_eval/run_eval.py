# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Claude Code on τ²-bench tasks through `HarnessEnvironment`.

Each task gets a fresh `tau2_env` (the simulated customer, the database and the
domain's tools) and a fresh Claude Code process with those tools injected. Every
customer message is one `step()`, and τ²-bench scores the conversation when the
customer is done.

    PYTHONPATH=src:envs:examples/claude_code_harness_eval \\
        python examples/claude_code_harness_eval/run_eval.py --domain airline --tasks 3

Requires the `claude` CLI (logged in, or `ANTHROPIC_API_KEY`), `HF_TOKEN` for the
simulated customer on Inference Providers, and `TAU2_DATA_DIR` outside the
`tau2_env` Docker image (see `envs/tau2_env/README.md`).
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Optional

from openenv.core.harness import HarnessAction, HarnessConfig
from tau2_env.server.tau2_environment import Tau2Environment
from tau2_harness import Tau2Harness


def run_task(task_id: str, args: argparse.Namespace) -> Optional[float]:
    """Play one task. Returns its reward, or `None` if Claude Code failed."""
    config = HarnessConfig(
        name="claude-code",
        command=[args.claude],
        # Claude Code has no tools of its own here, so a scratch directory is enough.
        working_directory=tempfile.mkdtemp(prefix="claude-code-tau2-"),
        model=args.model,
        session_timeout_s=args.turn_timeout,
    )
    tau2 = Tau2Environment(
        domain=args.domain, split=args.split, user_model=args.user_model
    )
    harness = Tau2Harness(tau2, task_id, config)
    print(f"\n=== {args.domain} task {task_id}")
    try:
        observation = harness.reset()
        while not observation.done:
            print(f"customer: {observation.metadata['customer']}")
            observation = harness.step(
                HarnessAction(message=observation.metadata["customer"])
            )
            for event in observation.metadata.get("turn_events", []):
                if event["type"] == "tool_call":
                    call = event["data"]
                    print(f"  -> {call['tool_name']}({call['arguments']})")
            agent = observation.metadata.get("response")
            print(f"agent: {agent or observation.metadata.get('error', '')}")
        if observation.metadata.get("customer"):  # the customer's last words
            print(f"customer: {observation.metadata['customer']}")
    finally:
        harness.close()
    if "error_type" in observation.metadata:
        print(f"error: {observation.metadata['error_type']}")
        return None
    breakdown = tau2.state.reward_info.get("reward_breakdown", {})
    print(f"reward: {observation.reward:.2f} {breakdown}")
    return observation.reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="airline", help="τ²-bench domain")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--tasks", type=int, default=3, help="how many tasks of the split to run"
    )
    parser.add_argument("--task-ids", nargs="*", help="run these tasks instead")
    parser.add_argument("--model", default="haiku", help="passed to `claude --model`")
    parser.add_argument(
        "--user-model",
        default=None,
        help="simulated customer model (Inference Providers)",
    )
    parser.add_argument(
        "--claude", default="claude", help="path to the Claude Code CLI"
    )
    parser.add_argument("--turn-timeout", type=float, default=180.0)
    args = parser.parse_args()

    task_ids = (
        args.task_ids
        or Tau2Environment(domain=args.domain, split=args.split).task_ids[: args.tasks]
    )
    rewards = {task_id: run_task(task_id, args) for task_id in task_ids}
    print()
    for task_id, reward in rewards.items():
        status = "ERROR" if reward is None else "PASS" if reward >= 1.0 else "FAIL"
        print(f"{status}  task {task_id}")
    # Tasks where Claude Code failed are left out: they measure the API, not the agent.
    scored = [r for r in rewards.values() if r is not None]
    print(f"pass^1: {sum(r >= 1.0 for r in scored)}/{len(scored)}")


if __name__ == "__main__":
    main()
