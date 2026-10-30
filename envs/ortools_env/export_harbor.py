# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a dataset built by `build_dataset.py` as Harbor tasks, to run frontier agents (claude-code, codex, ...) on the
same episodes through `harbor run` or the OpenEnv `harbor_env`.

Each task gets the instance in `/app/instance.json` and asks for `/app/solution.json` (level 1) or `/app/solve.py`
defining `solve(instance)` (level 2). The verifier, the problem module and, at level 2, the hidden evaluation
instances live in `tests/`, which Harbor only uploads after the agent has finished.

    python -m ortools_env.export_harbor --dataset rcpsp_l1.jsonl --out tasks/
"""

import argparse
import json
import os
import shutil

try:
    from .problems import PROBLEMS
    from .prompts import task_prompt
except ImportError:
    from problems import PROBLEMS
    from prompts import task_prompt

HERE = os.path.dirname(os.path.abspath(__file__))

SUBMIT_HELP = {
    1: "## Submission\n\nThe instance is in `/app/instance.json`. Write your final solution as JSON to "
    "`/app/solution.json`.",
    2: "## Submission\n\nThe example instance is in `/app/instance.json`. Write your final algorithm to "
    "`/app/solve.py`, defining `solve(instance: dict) -> dict`. It is imported and called in a fresh process for each "
    "evaluation instance.",
}

DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy{ortools}
# The verifier runs the submitted `solve` as this user, which cannot read `/tests` or write `/logs`.
RUN useradd --system --no-create-home sandbox
WORKDIR /app
COPY instance.json /app/instance.json
"""

TASK_TOML = """\
version = "1.0"

[metadata]
problem = "{problem}"
size = "{size}"
level = {level}
instance_seed = {instance_seed}

[verifier]
timeout_sec = {verifier_timeout}

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = 600.0
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", required=True, help="JSONL file from build_dataset.py."
    )
    parser.add_argument(
        "--out", required=True, help="Directory to write the tasks into."
    )
    parser.add_argument(
        "--allow-ortools",
        action="store_true",
        help="Install OR-Tools in the agent's image.",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=3600.0,
        help="Agent time per task, in seconds.",
    )
    parser.add_argument(
        "--solve-time-limit",
        type=float,
        default=30.0,
        help="Per-instance limit of `solve` (level 2).",
    )
    args = parser.parse_args()

    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f]
    for i, row in enumerate(rows):
        module = PROBLEMS[row["problem"]]
        level = row["level"]
        instance = module.generate(row["instance_seed"], row["size"])
        task_dir = os.path.join(
            args.out, f"{row['problem']}-{row['size']}-l{level}-{i:04d}"
        )
        os.makedirs(os.path.join(task_dir, "environment"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "tests"), exist_ok=True)

        prompt = task_prompt(
            module,
            level,
            row["size"],
            instance,
            args.solve_time_limit,
            args.allow_ortools,
        )
        budget = (
            f"## Compute budget\n\nYou have {args.agent_timeout:g} seconds in total."
        )
        with open(os.path.join(task_dir, "instruction.md"), "w") as f:
            f.write("\n\n".join([prompt, budget, SUBMIT_HELP[level]]) + "\n")
        with open(os.path.join(task_dir, "environment", "instance.json"), "w") as f:
            json.dump(instance, f)
        with open(os.path.join(task_dir, "environment", "Dockerfile"), "w") as f:
            f.write(DOCKERFILE.format(ortools=" ortools" if args.allow_ortools else ""))
        num_evals = len(row.get("eval_seeds", [])) or 1
        with open(os.path.join(task_dir, "task.toml"), "w") as f:
            f.write(
                TASK_TOML.format(
                    problem=row["problem"],
                    size=row["size"],
                    level=level,
                    instance_seed=row["instance_seed"],
                    verifier_timeout=num_evals * args.solve_time_limit + 300,
                    agent_timeout=args.agent_timeout,
                )
            )

        task = {"level": level, "solve_time_limit_s": args.solve_time_limit}
        if level == 1:
            task["reference"] = row["reference"]
            task["baseline"] = row["baseline"]
        else:
            task["eval_instances"] = [
                module.generate(s, row["size"]) for s in row["eval_seeds"]
            ]
            task["eval_references"] = row["eval_references"]
            task["eval_baselines"] = row["eval_baselines"]
        with open(os.path.join(task_dir, "tests", "task.json"), "w") as f:
            json.dump(task, f)
        shutil.copy(module.__file__, os.path.join(task_dir, "tests", "problem.py"))
        shutil.copy(
            os.path.join(HERE, "harbor_verify.py"),
            os.path.join(task_dir, "tests", "verify.py"),
        )
        with open(os.path.join(task_dir, "tests", "test.sh"), "w") as f:
            f.write("#!/bin/bash\npython3 /tests/verify.py\n")
        os.chmod(os.path.join(task_dir, "tests", "test.sh"), 0o755)
    print(f"Wrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
