"""Benchmark the AWM expert-in-the-loop recipe.

Runs the no-expert baseline and the verifier-informed expert condition side by side
on one or more AWM scenarios and prints a comparison. This is an *evaluation* harness:
the trainable agent is replaced here by any OpenAI-compatible chat model so the loop
can be exercised without a training backend.

Setup:

```bash
# 1. Start the AWM server (from the repo root)
PYTHONPATH=src:envs uv run uvicorn \\
    envs.agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899

# 2. Configure the model endpoints (Azure OpenAI shown; OPENAI_* also works)
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<key>"

# 3. Run on the pinned, non-trivial validation split (from the repo root)
PYTHONPATH=src:envs uv run python examples/awm_expert_in_the_loop/run_benchmark.py \\
    --split examples/awm_expert_in_the_loop/splits/workflow_automation.json \\
    --split-section val \\
    --agent-model gpt-5.4-mini --expert-model gpt-5.4-mini
```

The expert can use a separate endpoint via ``AZURE_OPENAI_EXPERT_ENDPOINT`` /
``AZURE_OPENAI_EXPERT_API_KEY`` so the agent and the expert may be hosted apart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import task_select
from agent_world_model_env import AWMEnv
from expert import VerifierInformedExpert
from policies import OpenAIChatPolicy
from rollout import (
    CONTENT_FILTER_MARKER,
    RESET_ERROR,
    run_task,
    SERVER_ERROR,
    TaskResult,
)

DEFAULT_BASE_URL = os.environ.get("AWM_BASE_URL", "http://127.0.0.1:8899")
DEFAULT_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview")
DEFAULT_MODEL = os.environ.get("AZURE_OPENAI_REASONING_NAME", "gpt-5.4-mini")

# Outcomes excluded from completion-rate metrics: content filtered or an
# infrastructure error (the task never produced a genuine attempt).
_EXCLUDED = frozenset({CONTENT_FILTER_MARKER, RESET_ERROR, SERVER_ERROR})


def _build_client(endpoint_env: str, key_env: str, *, required: bool):
    """Create an Async chat client from env vars.

    Resolution order: Azure OpenAI with an API key, then Azure OpenAI with Entra ID
    (``DefaultAzureCredential``) when only the endpoint is set, then OpenAI via
    ``OPENAI_API_KEY``.
    """
    azure_endpoint = os.environ.get(endpoint_env)
    azure_key = os.environ.get(key_env)
    if azure_endpoint and azure_key:
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=DEFAULT_API_VERSION,
        )
    if azure_endpoint:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AsyncAzureOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=DEFAULT_API_VERSION,
        )
    if os.environ.get("OPENAI_API_KEY"):
        from openai import AsyncOpenAI

        return AsyncOpenAI()
    if required:
        raise SystemExit(
            f"No model credentials found. Set {endpoint_env} (+ optional {key_env}) "
            "for Azure OpenAI, or OPENAI_API_KEY for OpenAI."
        )
    return None


def _summary(label: str, results: list[TaskResult]) -> dict:
    valid = [r for r in results if r.reward_type not in _EXCLUDED]
    excluded = len(results) - len(valid)
    stats = {
        "label": label,
        "total": len(results),
        "excluded": excluded,
        "valid": len(valid),
        "avg_reward": 0.0,
        "complete": 0,
        "avg_steps": 0.0,
        "expert_calls": sum(r.expert_calls for r in valid),
    }
    if valid:
        stats["avg_reward"] = sum((r.reward or 0.0) for r in valid) / len(valid)
        stats["complete"] = sum(1 for r in valid if r.reward_type == "complete")
        stats["avg_steps"] = sum(r.steps for r in valid) / len(valid)
    return stats


def _print_summary(stats: dict) -> None:
    print(f"\n{'=' * 80}\nRESULTS: {stats['label']}\n{'=' * 80}")
    print(
        f"  tasks: {stats['total']}  valid: {stats['valid']}  excluded: {stats['excluded']}"
    )
    if not stats["valid"]:
        print("  no valid tasks")
        return
    pct = 100 * stats["complete"] / stats["valid"]
    print(f"  avg reward: {stats['avg_reward']:.3f}")
    print(f"  complete:   {stats['complete']}/{stats['valid']} ({pct:.0f}%)")
    print(f"  avg steps:  {stats['avg_steps']:.1f}")
    if stats["expert_calls"]:
        print(f"  expert calls (total): {stats['expert_calls']}")


async def _collect_tasks(base_url, scenarios, num_tasks, filter_trivial):
    """Build the `(scenario, task_idx)` list, optionally dropping trivial/unverified tasks."""
    tasks: list[tuple[str, int]] = []
    for scenario in scenarios:
        if not filter_trivial:
            tasks.extend((scenario, i) for i in range(num_tasks))
            continue
        kept = 0
        for task_idx in range(num_tasks * 4):
            if kept >= num_tasks:
                break
            async with AWMEnv(base_url=base_url) as env:
                if await task_select.is_usable_task(env, scenario, task_idx):
                    tasks.append((scenario, task_idx))
                    kept += 1
        if kept == 0:
            print(
                f"  [warn] no usable (code-verified, non-trivial) tasks in {scenario}"
            )
        elif kept < num_tasks:
            print(f"  [warn] only found {kept}/{num_tasks} usable tasks in {scenario}")
    return tasks


def _load_split(path: str, section: str) -> list[tuple[str, int]]:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if section not in manifest:
        raise SystemExit(
            f"Split manifest {path!r} has no section {section!r}. "
            "Expected one of: train, val."
        )
    return [(scenario, int(task_idx)) for scenario, task_idx in manifest[section]]


async def _run_suite(
    base_url, agent_policy, expert, tasks, max_iters, max_tool_response_chars, verbose
) -> list[TaskResult]:
    results: list[TaskResult] = []
    label = "expert" if expert is not None else "baseline"
    last_scenario = None
    for scenario, task_idx in tasks:
        if scenario != last_scenario:
            print(f"\n[{label}] scenario: {scenario}")
            last_scenario = scenario
        t0 = time.time()
        try:
            async with AWMEnv(base_url=base_url) as env:
                result = await run_task(
                    env,
                    agent_policy,
                    scenario=scenario,
                    task_idx=task_idx,
                    expert=expert,
                    max_iters=max_iters,
                    max_tool_response_chars=max_tool_response_chars,
                    verbose=verbose,
                )
        except Exception as exc:
            result = TaskResult(scenario, task_idx, "", 0, 1, 0, SERVER_ERROR, None)
            if verbose:
                print(
                    f"  task {task_idx}: [ERR] {type(exc).__name__}: {str(exc)[:120]}"
                )
        results.append(result)
        if result.reward_type == "complete":
            icon = "PASS"
        elif result.filtered:
            icon = "FILT"
        elif result.reward_type in (RESET_ERROR, SERVER_ERROR):
            icon = "ERR"
        else:
            icon = "FAIL"
        tail = f" expert={result.expert_calls}" if expert is not None else ""
        reward_str = f"{result.reward:.1f}" if result.reward is not None else "na"
        print(
            f"  task {task_idx}: [{icon}] reward={reward_str:>4}"
            f" steps={result.steps}{tail} ({time.time() - t0:.0f}s) {result.task[:48]}"
        )
    return results


def _print_comparison(baseline: dict, expert: dict) -> None:
    print(f"\n{'=' * 80}\nCOMPARISON: baseline vs expert\n{'=' * 80}")
    print(f"  {'metric':<22}{'baseline':>12}{'expert':>12}{'delta':>12}")
    print(f"  {'-' * 56}")
    dr = expert["avg_reward"] - baseline["avg_reward"]
    print(
        f"  {'avg reward':<22}{baseline['avg_reward']:>12.3f}{expert['avg_reward']:>12.3f}{dr:>+12.3f}"
    )
    print(
        f"  {'complete':<22}{baseline['complete']:>12}{expert['complete']:>12}{expert['complete'] - baseline['complete']:>+12}"
    )
    print(
        f"  {'avg steps':<22}{baseline['avg_steps']:>12.1f}{expert['avg_steps']:>12.1f}{expert['avg_steps'] - baseline['avg_steps']:>+12.1f}"
    )
    print(f"  {'expert calls':<22}{'n/a':>12}{expert['expert_calls']:>12}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="AWM expert-in-the-loop benchmark")
    parser.add_argument("scenarios", nargs="*", help="One or more scenario names")
    parser.add_argument("--tasks", type=int, default=5, help="Tasks per scenario")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="AWM server base URL"
    )
    parser.add_argument(
        "--agent-model", default=os.environ.get("AWM_AGENT_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--expert-model", default=os.environ.get("AWM_EXPERT_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--condition", choices=["baseline", "expert", "both"], default="both"
    )
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--max-tool-response-chars", type=int, default=3000)
    parser.add_argument(
        "--split",
        help="Pinned split manifest path. When set, scenarios/--tasks are ignored.",
    )
    parser.add_argument(
        "--split-section",
        choices=["train", "val"],
        default="val",
        help="Manifest section to run when --split is set",
    )
    parser.add_argument(
        "--no-filter-trivial",
        action="store_true",
        help="Do not skip verifier-less tasks or tasks that pass with no actions",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    agent_client = _build_client(
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", required=True
    )
    agent_policy = OpenAIChatPolicy(agent_client, args.agent_model)

    if args.split:
        tasks = _load_split(args.split, args.split_section)
    else:
        if not args.scenarios:
            raise SystemExit("Provide at least one scenario or pass --split.")
        tasks = await _collect_tasks(
            args.base_url, args.scenarios, args.tasks, not args.no_filter_trivial
        )
    if not tasks:
        raise SystemExit("No runnable tasks after filtering.")

    baseline_stats = expert_stats = None

    if args.condition in ("baseline", "both"):
        print("\n" + "#" * 80 + "\n# baseline (no expert)\n" + "#" * 80)
        baseline_results = await _run_suite(
            args.base_url,
            agent_policy,
            None,
            tasks,
            args.max_iters,
            args.max_tool_response_chars,
            args.verbose,
        )
        baseline_stats = _summary("baseline (no expert)", baseline_results)
        _print_summary(baseline_stats)

    if args.condition in ("expert", "both"):
        expert_client = (
            _build_client(
                "AZURE_OPENAI_EXPERT_ENDPOINT",
                "AZURE_OPENAI_EXPERT_API_KEY",
                required=False,
            )
            or agent_client
        )
        expert = VerifierInformedExpert(expert_client, args.expert_model)
        print(
            "\n" + "#" * 80 + "\n# expert-in-the-loop (verifier-informed)\n" + "#" * 80
        )
        expert_results = await _run_suite(
            args.base_url,
            agent_policy,
            expert,
            tasks,
            args.max_iters,
            args.max_tool_response_chars,
            args.verbose,
        )
        expert_stats = _summary("expert-in-the-loop", expert_results)
        _print_summary(expert_stats)

    if baseline_stats and expert_stats:
        _print_comparison(baseline_stats, expert_stats)


if __name__ == "__main__":
    asyncio.run(main())
