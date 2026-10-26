#!/usr/bin/env python3
"""End-to-end Terminal-Bench 2 evaluation on a Novita sandbox.

The other TB2 examples stop short of an actual episode: ``novita_tbench2_simple.py``
resets a task and runs ``ls``, and ``daytona_tbench2_concurrent.py`` measures
sandbox startup time. Neither scores anything. This one runs the whole loop on
one task in one sandbox — an LLM agent works the task over the environment's
``exec`` action, then ``evaluate`` scores it with the task's own canonical
verifier and returns the reward the environment produced.

The client owns the sandbox (``provider=provider``), so leaving the context
manager tears it down.

Usage:
    PYTHONPATH=src:envs uv run python examples/novita_tbench2_e2e_eval.py
    PYTHONPATH=src:envs uv run python examples/novita_tbench2_e2e_eval.py \
        --task regex-log --max-steps 20 --verbose

The TB2 repo is downloaded inside the sandbox on first ``reset``, so the task is
selected by id — a local checkout on this machine is not visible to it.

Requires:
    NOVITA_API_KEY, plus an OpenAI-compatible endpoint (``HF_TOKEN``, or
    ``API_KEY`` with optional ``API_BASE_URL`` / ``MODEL``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from openai import AsyncOpenAI
from openenv.core.containers.runtime.novita_provider import NovitaSandboxProvider
from tbench2_env import Tbench2Action, Tbench2Env

DOCKERFILE = "envs/tbench2_env/server/Dockerfile"

# fix-git needs only git, which the env-server image already has, so it scores
# meaningfully in TB2_MODE=local. Most TB2 tasks declare their own docker_image
# and only reach full fidelity in docker mode — see --tb2-mode.
DEFAULT_TASK = "fix-git"

MAX_OUTPUT_CHARS = 8000

SYSTEM_PROMPT = """You are an expert engineer working in a Linux terminal.

You are given a task to complete. Work step by step: inspect the environment,
make changes, and verify your work with commands.

Rules:
- Issue one shell command at a time via the `bash` tool.
- Commands run non-interactively and return combined stdout/stderr.
- Do not ask questions; no human is available to answer.
- Read files before editing them.
- When you believe the task is complete, call `submit` to end the episode.

Your work is scored afterwards by the task's own test suite, not by you."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the task container and return its "
                "combined stdout/stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Declare the task finished. Call this once you are confident "
                "the task is complete; the episode is then scored."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real TB2 task end to end on a Novita sandbox.",
    )
    parser.add_argument(
        "--task",
        default=os.getenv("TB2_TASK_ID", DEFAULT_TASK),
        help=f"TB2 task id to run. Defaults to {DEFAULT_TASK!r}.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max agent turns before scoring (default: 30).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "openai/gpt-oss-120b:novita"),
        help="Model name for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        help="Base URL of the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY") or os.getenv("HF_TOKEN"),
        help="API key. Defaults to API_KEY, then HF_TOKEN.",
    )
    parser.add_argument(
        "--tb2-mode",
        default="local",
        choices=("local", "docker"),
        help=(
            "Server execution mode. 'docker' needs a Docker daemon inside the "
            "sandbox and gives the task's own image (full TB2 fidelity) "
            "(default: local)."
        ),
    )
    parser.add_argument(
        "--command-timeout-s",
        type=float,
        default=300.0,
        help=(
            "Per-command budget inside the sandbox. The camel toolkit defaults "
            "to 20s, which truncates real agent commands (default: 300)."
        ),
    )
    parser.add_argument(
        "--message-timeout-s",
        type=float,
        default=1200.0,
        help=(
            "WebSocket response timeout. TB2 verifiers legitimately run for "
            "minutes (task.toml declares up to 900s), so the 60s client "
            "default would time out mid-scoring (default: 1200)."
        ),
    )
    parser.add_argument(
        "--sandbox-lifetime-s",
        type=int,
        default=3600,
        help=(
            "Hard sandbox lifetime. Counts down from creation regardless of "
            "activity and always fires, so it must exceed the whole episode "
            "(default: 3600)."
        ),
    )
    parser.add_argument(
        "--sandbox-timeout-s",
        type=int,
        default=300,
        help="Seconds to wait for the sandbox to become ready (default: 300).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every command the agent runs and its output.",
    )
    return parser.parse_args()


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Clip *text* to *limit* characters, noting how much was dropped.

    Args:
        text (`str`):
            The command output to clip.
        limit (`int`, *optional*, defaults to `8000`):
            Maximum characters to keep.

    Returns:
        `str` the clipped text.
    """
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n... [{dropped} more characters truncated]"


async def run_agent(
    env: Tbench2Env,
    *,
    instruction: str,
    llm: AsyncOpenAI,
    model: str,
    max_steps: int,
    verbose: bool,
) -> tuple[int, list[str]]:
    """Drive the task with a tool-calling agent until it submits or runs out of turns.

    Args:
        env (`Tbench2Env`):
            Connected client for the task session.
        instruction (`str`):
            The task instruction returned by ``reset``.
        llm (`AsyncOpenAI`):
            Client for the OpenAI-compatible endpoint.
        model (`str`):
            Model name to request.
        max_steps (`int`):
            Maximum agent turns before scoring.
        verbose (`bool`):
            Whether to echo each command and its output.

    Returns:
        `tuple[int, list[str]]`: turns used and the commands issued.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    commands: list[str] = []
    steps = 0

    for step in range(1, max_steps + 1):
        response = await llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.0,
        )
        message = response.choices[0].message
        steps = step

        assistant: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
        messages.append(assistant)

        if not message.tool_calls:
            # Model stopped acting without submitting; score what is there.
            break

        submitted = False
        for call in message.tool_calls:
            if call.function.name == "submit":
                submitted = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "Episode submitted for scoring.",
                    }
                )
                continue

            try:
                command = json.loads(call.function.arguments or "{}").get("command", "")
            except json.JSONDecodeError:
                command = ""

            result = await env.step(Tbench2Action(action_type="exec", command=command))
            obs = result.observation
            commands.append(command)

            output = f"ERROR: {obs.error}" if not obs.success else truncate(obs.output)

            if verbose:
                print(f"  $ {command}")
                for line in (output or "(no output)").splitlines():
                    print(f"    {line}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output or "(no output)",
                }
            )

        if submitted:
            break

    return steps, commands


async def main() -> int:
    args = parse_args()
    if not os.environ.get("NOVITA_API_KEY"):
        raise SystemExit("Set NOVITA_API_KEY to create the sandbox.")
    if not args.api_key:
        raise SystemExit("Set HF_TOKEN (or API_KEY) to query the model.")

    llm = AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key)
    env_vars = {
        "TB2_MODE": args.tb2_mode,
        "TB2_COMMAND_TIMEOUT_S": str(args.command_timeout_s),
    }

    image = NovitaSandboxProvider.image_from_dockerfile(str(_REPO_ROOT / DOCKERFILE))
    provider = NovitaSandboxProvider(
        env_vars=env_vars,
        timeout=args.sandbox_lifetime_s,
    )

    print(f"Starting Novita sandbox (TB2_MODE={args.tb2_mode})...")
    # The first start builds a Novita template from the Dockerfile, which can
    # take minutes; later starts reuse the cached template.
    base_url = await asyncio.to_thread(provider.start_container, image)
    reward: float | None = None
    try:
        await asyncio.to_thread(
            provider.wait_for_ready, base_url, args.sandbox_timeout_s
        )
        print("Novita sandbox server is ready.\n")

        # The client is given the provider, so exiting this block tears the
        # sandbox down — one task, one sandbox. The outer finally covers the
        # window before the client exists (a readiness timeout, say), where
        # nothing else would release it; stop_container is idempotent, so the
        # two paths overlapping is harmless.
        async with Tbench2Env(
            base_url=base_url,
            provider=provider,
            message_timeout_s=args.message_timeout_s,
        ) as env:
            result = await env.reset(task_id=args.task)
            instruction = result.observation.instruction
            print(f"Task {result.observation.task_id}:")
            print(f"  {instruction[:300]}\n")

            steps, commands = await run_agent(
                env,
                instruction=instruction,
                llm=llm,
                model=args.model,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
            print(f"Agent finished after {steps} step(s); scoring...")

            scored = await env.step(Tbench2Action(action_type="evaluate"))
            obs = scored.observation
            if not obs.success:
                print(f"Scoring failed: {obs.error}")
            else:
                reward = scored.reward
                if args.verbose or reward != 1.0:
                    print("\n--- verifier output (tail) ---")
                    for line in (obs.output or "").splitlines()[-40:]:
                        print(f"  {line}")
    finally:
        await asyncio.to_thread(provider.stop_container)

    if reward is None:
        print("\nNo reward: the verifier produced no verdict.")
        return 1

    print(f"\nReward: {reward:.1f} ({'PASS' if reward == 1.0 else 'FAIL'})")
    return 0 if reward == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
