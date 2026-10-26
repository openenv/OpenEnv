#!/usr/bin/env python3
"""Import and use a tiny ORS/OpenReward-style environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


ENV_NAME = "imported_ors_demo"
EXAMPLE_DIR = Path(__file__).resolve().parent


def import_environment(source_dir: Path, output_dir: Path) -> Path:
    """Run the OpenEnv importer and return the generated package directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / ENV_NAME
    if generated_dir.exists():
        shutil.rmtree(generated_dir)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "openenv.cli",
            "import",
            str(source_dir),
            "--name",
            ENV_NAME,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    return generated_dir


def _clear_demo_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == ENV_NAME or module_name.startswith(f"{ENV_NAME}."):
            sys.modules.pop(module_name, None)
        if module_name == "ors" or module_name.startswith("ors."):
            sys.modules.pop(module_name, None)


def run_imported_environment(output_dir: Path) -> dict[str, Any]:
    """Use the generated OpenEnv wrapper directly."""
    _clear_demo_modules()
    sys.path.insert(0, str(output_dir))
    try:
        from imported_ors_demo.server.imported_ors_demo_environment import (  # type: ignore
            ImportedOrsDemoEnvironment,
        )

        env = ImportedOrsDemoEnvironment()
        reset_observation = env.reset(split="train", index=0)
        tools_observation = env.step(ListToolsAction())
        answer_observation = env.step(
            CallToolAction(tool_name="answer", arguments={"value": "4"})
        )

        return {
            "prompt": reset_observation.metadata["prompt"][0]["text"],
            "tools": [tool.name for tool in tools_observation.tools],
            "result": answer_observation.result["blocks"][0]["text"],
            "reward": answer_observation.reward,
            "done": answer_observation.done,
        }
    finally:
        try:
            sys.path.remove(str(output_dir))
        except ValueError:
            pass
        _clear_demo_modules()


def _print_summary(generated_dir: Path, summary: dict[str, Any]) -> None:
    print(f"Generated environment: {generated_dir}")
    print(f"Prompt: {summary['prompt']}")
    print(f"Tools: {', '.join(summary['tools'])}")
    print(f"Tool result: {summary['result']}")
    print(f"Reward: {summary['reward']}")
    print(f"Done: {summary['done']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the generated OpenEnv package. Defaults to a temp dir.",
    )
    args = parser.parse_args(argv)

    source_dir = EXAMPLE_DIR / "source"
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        generated_dir = import_environment(source_dir, output_dir)
        summary = run_imported_environment(output_dir)
        _print_summary(generated_dir, summary)
        return

    with TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        generated_dir = import_environment(source_dir, output_dir)
        summary = run_imported_environment(output_dir)
        _print_summary(generated_dir, summary)


if __name__ == "__main__":
    main()
