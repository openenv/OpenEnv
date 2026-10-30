# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task prompts shared by the OpenEnv server and the Harbor exporter.

The two levels follow "LLMs Can Design Near-Optimal OR Algorithms" (https://huggingface.co/papers/2608.27296):
at level 1 the model sees one instance and returns a solution for it; at level 2 it sees the problem class and returns
an algorithm that is then run on unseen instances. Each front end appends its own section on how to run code and
submit.
"""

import json

# Above this many characters the instance is only given as a file / sandbox variable, not inline.
MAX_INLINE_INSTANCE_CHARS = 20000


def task_prompt(
    module,
    level: int,
    size: str,
    instance: dict,
    solve_time_limit_s: float,
    allow_ortools: bool,
) -> str:
    libraries = "the Python standard library, numpy and scipy" + (
        " and OR-Tools (`ortools`)" if allow_ortools else ""
    )
    parts = [f"# {module.NAME} ({size})", module.STATEMENT]
    scoring = (
        "Scoring: `(naive - yours) / (naive - reference)`, where `reference` is the objective of an OR-Tools "
        f"reference solution and `naive` the objective of the naive solution that {module.BASELINE}. Matching the "
        "reference scores 1.0 and beating it scores above 1.0; a solution no better than the naive one scores 0 "
        "or less, down to -1.0, which is also the score of an infeasible solution."
    )
    if level == 1:
        parts.append(
            "## Task\n\nFind the best solution you can for the instance below. You may use "
            f"{libraries}.\n\n{scoring}"
        )
    else:
        parts.append(
            "## Task\n\nDesign an algorithm for this problem class. Write Python code defining "
            "`solve(instance: dict) -> dict` that returns a solution in the format above. It will be run on unseen "
            f"instances of size {size} drawn from the same distribution as the example instance below, with "
            f"{solve_time_limit_s:g} seconds per instance; it may only use {libraries}. Your score is the mean of "
            f"the per-instance scores (a crash or a timeout counts as infeasible).\n\n{scoring}"
        )
    text = json.dumps(instance, separators=(",", ":"))
    name = "Instance" if level == 1 else "Example instance"
    if len(text) <= MAX_INLINE_INSTANCE_CHARS:
        parts.append(f"## {name}\n\n```json\n{text}\n```")
    else:
        parts.append(
            f"## {name}\n\nThe instance is too large to show here ({len(text)} characters of JSON)."
        )
    return "\n\n".join(parts)
