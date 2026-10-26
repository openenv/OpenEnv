"""Experimental BrowserGym evaluation example using Codex CLI as the harness.

This example is eval-only. Codex runs as a black-box CLI agent and connects to
BrowserGym through a local HTTP MCP bridge. It is not suitable for white-box RL
training or any setup that needs token IDs or logprobs from each model step.

Manual prerequisites:
- `codex` installed and authenticated
- BrowserGym Docker image available locally, or a running BrowserGym server
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from browsergym_harness_eval_common import (
    build_browsergym_session_factory,
    DEFAULT_BENCHMARK,
    DEFAULT_BROWSERGYM_IMAGE,
    DEFAULT_MAX_STEPS,
    DEFAULT_TASK_NAME,
    format_episode_summary,
    run_black_box_episode,
    SessionMCPHttpServer,
    start_browsergym_runtime,
    summarize_episodes,
)
from openenv.core.harness import (
    CLIHarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    RolloutEvent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental BrowserGym evaluation through Codex CLI + MCP.",
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
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum browser actions Codex should take.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Path to the Codex CLI binary.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory where stdout and the last-message artifact will be written.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Timeout for each Codex run.",
    )
    return parser.parse_args()


def build_codex_prompt(
    initial_messages: list[dict[str, object]], max_steps: int
) -> str:
    initial_context = "\n\n".join(
        str(message.get("content", "")).strip()
        for message in initial_messages
        if str(message.get("content", "")).strip()
    )
    return (
        "You are evaluating a BrowserGym task.\n"
        "Use only the MCP tools exposed by the 'browsergym' server.\n"
        "Do not use shell commands, files, or any other tool.\n"
        f"Take at most {max_steps} browser actions.\n"
        "Stop once the task is complete or no useful progress is possible.\n\n"
        "Initial BrowserGym context:\n"
        f"{initial_context}"
    )


def build_codex_runner(
    *,
    codex_bin: str,
    artifact_root: Path,
    timeout_seconds: int,
):
    def runner(bridge, session, limits):
        if shutil.which(codex_bin) is None:
            raise RuntimeError(f"Codex binary not found: {codex_bin}")

        initial_messages = session.initial_messages()
        prompt_text = build_codex_prompt(initial_messages, limits.max_turns)

        artifact_root.mkdir(parents=True, exist_ok=True)
        prompt_path = artifact_root / "codex_prompt.txt"
        stdout_path = artifact_root / "codex_exec.jsonl"
        stderr_path = artifact_root / "codex_exec.stderr"
        last_message_path = artifact_root / "codex_last_message.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="browsergym-codex-home-") as codex_home:
            with tempfile.TemporaryDirectory(
                prefix="browsergym-codex-cwd-"
            ) as codex_cwd:
                env = dict(os.environ)
                env["CODEX_HOME"] = codex_home

                with SessionMCPHttpServer(bridge).start() as mcp_server:
                    add_cmd = [
                        codex_bin,
                        "mcp",
                        "add",
                        "browsergym",
                        "--url",
                        mcp_server.url,
                    ]
                    add_result = subprocess.run(
                        add_cmd,
                        capture_output=True,
                        text=True,
                        env=env,
                        check=False,
                        timeout=timeout_seconds,
                    )
                    if add_result.returncode != 0:
                        raise RuntimeError(
                            "Failed to register BrowserGym MCP server with Codex:\n"
                            f"{add_result.stderr.strip() or add_result.stdout.strip()}"
                        )

                    exec_cmd = [
                        codex_bin,
                        "exec",
                        "--json",
                        "--skip-git-repo-check",
                        "-s",
                        "read-only",
                        "-C",
                        codex_cwd,
                        "--output-last-message",
                        str(last_message_path),
                        prompt_text,
                    ]
                    exec_result = subprocess.run(
                        exec_cmd,
                        capture_output=True,
                        text=True,
                        env=env,
                        check=False,
                        timeout=timeout_seconds,
                    )

        stdout_path.write_text(exec_result.stdout, encoding="utf-8")
        stderr_path.write_text(exec_result.stderr, encoding="utf-8")

        last_message = ""
        if last_message_path.exists():
            last_message = last_message_path.read_text(encoding="utf-8").strip()

        return HarnessRolloutResult(
            messages=[
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": last_message},
            ],
            events=[
                RolloutEvent(
                    type="codex_exec",
                    payload={
                        "returncode": exec_result.returncode,
                        "artifacts": {
                            "prompt": str(prompt_path),
                            "stdout": str(stdout_path),
                            "stderr": str(stderr_path),
                            "last_message": str(last_message_path),
                        },
                    },
                )
            ],
            done=exec_result.returncode == 0,
            metrics={
                "mode": "black_box",
                "codex_returncode": exec_result.returncode,
                "artifact_dir": str(artifact_root),
            },
        )

    return runner


def main() -> None:
    args = parse_args()
    artifact_root = Path(
        args.artifact_dir or tempfile.mkdtemp(prefix="browsergym-codex-artifacts-")
    ).resolve()

    with start_browsergym_runtime(
        base_url=args.base_url,
        image=args.image,
        benchmark=args.benchmark,
        task_name=args.task_name,
    ) as runtime:
        print(f"BrowserGym server: {runtime.base_url}")
        print(f"Artifact dir: {artifact_root}")

        session_factory = build_browsergym_session_factory(
            base_url=runtime.base_url,
            task_name=args.task_name,
        )
        episodes = []
        limits = HarnessRunLimits(max_turns=args.max_steps)
        for index in range(1, args.episodes + 1):
            episode_artifact_root = artifact_root / f"episode-{index}"
            episode_adapter = CLIHarnessAdapter(
                runner=build_codex_runner(
                    codex_bin=args.codex_bin,
                    artifact_root=episode_artifact_root,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            episode = run_black_box_episode(
                session_factory=session_factory,
                harness_adapter=episode_adapter,
                limits=limits,
                episode_id=f"browsergym-codex-eval-{index}",
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
