"""Inspect one public ThinkingBox task through a running OpenEnv server."""

from __future__ import annotations

import argparse
import asyncio

from thinkingbox_env import ThinkingBoxEnv


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone public-client example parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("test_uid")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        help="Server-visible executable data root or dataset/ path.",
    )
    parser.add_argument(
        "--config",
        help="Server-visible native ThinkingBox configuration path.",
    )
    parser.add_argument(
        "--agent",
        help="Native agent definition; defaults to the server configuration.",
    )
    parser.add_argument(
        "--response",
        help="Optional assistant response to submit after inspecting the task.",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    """Reset one task and print only public observations."""
    async with ThinkingBoxEnv(base_url=args.base_url) as env:
        result = await env.reset(
            args.test_uid,
            dataset=args.dataset,
            agent=args.agent,
            config=args.config,
        )
        observation = result.observation
        print("kind:", observation.kind)
        print("task:", observation.task)
        print("tools:", [tool.name for tool in observation.tools or []])

        if not result.done:
            listed = await env.list_tools()
            print(
                "listed_tools:",
                [tool.name for tool in listed.observation.tools or []],
            )
        if not result.done and args.response is not None:
            result = await env.submit_message(args.response)

        print("done:", result.done)
        print("reward:", result.reward)
        print("finish_reason:", result.observation.finish_reason)
        print("system_error:", result.observation.system_error)


def main() -> None:
    """Run the standalone public-client example."""
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
