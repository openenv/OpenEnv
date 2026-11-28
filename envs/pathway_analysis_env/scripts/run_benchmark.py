#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run a fixed policy on pathway_analysis_env cases and report accuracy / tool usage.

Example:
  PYTHONPATH=src:envs python envs/pathway_analysis_env/scripts/run_benchmark.py \\
      --case toy_case_001.json --guess 'MAPK signaling'
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.pathway_environment import (
    PathwayEnvironment,
    load_case,
)


def run_episode(case_file: str, guess: str, strict: bool) -> Dict[str, Any]:
    case = load_case(case_file)
    ref = (case.get("default_contrast") or {}).get("reference", "control")
    alt = (case.get("default_contrast") or {}).get("alternate", "treated")

    env = PathwayEnvironment(case_file=case_file)
    env.reset(strict=strict)
    actions: List[str] = []
    total_reward = 0.0

    def go(kind: str, **kw: Any) -> Any:
        actions.append(kind)
        nonlocal total_reward
        o = env.step(PathwayAction(action_type=kind, **kw))
        total_reward += float(o.reward or 0.0)
        return o

    go("inspect_dataset")
    o_de = go(
        "run_differential_expression",
        condition_a=ref,
        condition_b=alt,
    )
    if o_de.done:
        return {
            "correct": False,
            "actions": actions,
            "reward": total_reward,
            "strict_failure": True,
        }
    go("run_pathway_enrichment")
    pnames = list((case.get("pathway_genes") or {}).keys())
    if len(pnames) >= 2:
        go("compare_pathways", pathway_a=pnames[0], pathway_b=pnames[1])
    o_fin = go("submit_answer", hypothesis=guess)
    correct = bool(o_fin.metadata.get("correct")) if o_fin.metadata else False
    return {
        "correct": correct,
        "actions": actions,
        "reward": total_reward,
        "strict_failure": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pathway env episodes.")
    parser.add_argument("--case", default="toy_case_001.json")
    parser.add_argument("--guess", default="MAPK signaling")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", action="store_true", help="Print JSON line")
    args = parser.parse_args()

    stats = run_episode(args.case, args.guess, args.strict)
    if args.json_out:
        print(json.dumps(stats))
    else:
        print("correct:", stats["correct"])
        print("actions:", stats["actions"])
        print("reward (partial sum):", stats["reward"])
        print("strict_failure:", stats["strict_failure"])


if __name__ == "__main__":
    main()
