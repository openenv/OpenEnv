#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke-check a real OpenEnv server through the HF sandbox provider."""

# /// script
# dependencies = [
#     "huggingface_hub>=1.22.0",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.coding_env import CodeAction, CodingEnv
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider


IMAGE = "hf.co/spaces/openenv/coding_env"


def run_code(base_url: str, code: str) -> str:
    with CodingEnv(base_url=base_url).sync() as env:
        env.reset()
        result = env.step(CodeAction(code=code))
    observation = result.observation
    if observation.exit_code != 0:
        raise RuntimeError(observation.stderr)
    return observation.stdout.strip()


def main() -> None:
    with HFSandboxProvider(image=IMAGE) as provider:
        base_url = provider.start_container()
        print(f"provider URL: {base_url}")
        provider.wait_for_ready(base_url, timeout_s=300.0)

        first_output = run_code(base_url, "answer = 40 + 2\nprint(answer)")
        print(f"first connection output: {first_output!r}")
        if first_output != "42":
            raise RuntimeError("first HF sandbox connection returned unexpected output")

        second_output = run_code(
            base_url, "message = 'second connection ok'\nprint(message)"
        )
        print(f"second connection output: {second_output!r}")
        if second_output != "second connection ok":
            raise RuntimeError(
                "second HF sandbox connection returned unexpected output"
            )

    print("HF sandbox coding_env check passed")


if __name__ == "__main__":
    main()
