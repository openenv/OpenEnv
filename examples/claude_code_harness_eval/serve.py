# SPDX-License-Identifier: BSD-3-Clause

"""Serve Claude Code as a τ²-bench agent in production mode: you are the customer, on `WS /harness`.

    PYTHONPATH=src:envs:examples/claude_code_harness_eval \\
        python examples/claude_code_harness_eval/serve.py --domain airline --task-id 2
    python examples/claude_code_harness_eval/chat.py ws://localhost:8000/harness

The server takes one connection at a time, each with a fresh copy of the task's
database and its own Claude Code process. You play the customer, so nothing is
scored. It has no authentication, and every connection spends your credentials, so
keep it on localhost or behind your own auth.

`tau2_env` loads a task's database and policy in `reset()`, which also opens the
conversation with its simulated customer. That message is not used here, but it
needs `HF_TOKEN` and costs one customer call per connection.
"""

from __future__ import annotations

import argparse
import tempfile

import uvicorn
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.types import Observation
from openenv.core.harness import HarnessAction, HarnessConfig, HarnessEnvironment
from tau2_env.server.tau2_environment import Tau2Environment
from tau2_harness import Tau2Harness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="airline")
    parser.add_argument(
        "--task-id", default="2", help="the task whose database the agent works on"
    )
    parser.add_argument("--model", default="haiku", help="passed to `claude --model`")
    parser.add_argument(
        "--claude", default="claude", help="path to the Claude Code CLI"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 inside a container"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    def make_env() -> HarnessEnvironment:
        config = HarnessConfig(
            name="claude-code",
            command=[args.claude],
            working_directory=tempfile.mkdtemp(prefix="claude-code-serve-"),
            model=args.model,
        )
        tau2 = Tau2Environment(domain=args.domain, split="base")
        return Tau2Harness(tau2, args.task_id, config)

    app = create_fastapi_app(make_env, HarnessAction, Observation, mode="production")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
