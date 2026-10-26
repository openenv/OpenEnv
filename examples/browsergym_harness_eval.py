"""White-box BrowserGym evaluation example driven through the harness runtime.

This is a plain inference/evaluation script. It runs BrowserGym locally by
default, uses an OpenAI-compatible chat model to propose BrowserGym actions,
and executes those actions through the harness/session layer.

Example:
    python examples/browsergym_harness_eval.py --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from openenv.core.harness import HarnessRunLimits

from browsergym_harness_eval_common import (
    DEFAULT_BENCHMARK,
    DEFAULT_BROWSERGYM_IMAGE,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TASK_NAME,
    build_browsergym_session_factory,
    build_openai_model_step,
    create_openai_client,
    format_episode_summary,
    run_white_box_episode,
    start_browsergym_runtime,
    summarize_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BrowserGym task through the harness runtime.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Use an already-running BrowserGym server instead of starting Docker.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_BROWSERGYM_IMAGE,
        help="Docker image to start when --base-url is not provided.",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help="BrowserGym benchmark to run when starting Docker.",
    )
    parser.add_argument(
        "--task-name",
        default=DEFAULT_TASK_NAME,
        help="BrowserGym task name to evaluate.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI-compatible chat model name.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum tool/model turns per episode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the chat model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum completion tokens for each model step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = create_openai_client(api_base_url=args.api_base_url)

    with start_browsergym_runtime(
        base_url=args.base_url,
        image=args.image,
        benchmark=args.benchmark,
        task_name=args.task_name,
    ) as runtime:
        print(f"BrowserGym server: {runtime.base_url}")
        session_factory = build_browsergym_session_factory(
            base_url=runtime.base_url,
            task_name=args.task_name,
        )
        model_step = build_openai_model_step(
            client=client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        episodes = []
        limits = HarnessRunLimits(max_turns=args.max_steps)
        for index in range(1, args.episodes + 1):
            episode = run_white_box_episode(
                session_factory=session_factory,
                model_step=model_step,
                limits=limits,
                episode_id=f"browsergym-eval-{index}",
            )
            episodes.append(episode)
            print(format_episode_summary(index, episode))

        summary = summarize_episodes(episodes)
        print(
            "Aggregate: "
            f"episodes={int(summary['episodes'])} "
            f"avg_reward={summary['avg_reward']:.2f} "
            f"success_rate={summary['success_rate']:.2%} "
            f"avg_steps={summary['avg_steps']:.2f}"
        )


if __name__ == "__main__":
    main()
