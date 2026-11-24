# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Build a JSONL dataset of episodes with precomputed OR-Tools references.

Each row holds the `reset` arguments of one episode: problem, size, level, instance seed, reference and naive
baseline (level 1), or evaluation seeds, references and baselines (level 2). Sizes are sampled per row from `--sizes`, so one dataset can mix sizes.

    python -m ortools_env.build_dataset --problem rcpsp --sizes j60 j120 --level 1 --num-rows 256 \
        --reference-time-limit 60 --out rcpsp_l1.jsonl
"""

import argparse
import json
import random

try:
    from .problems import PROBLEMS
except ImportError:
    from problems import PROBLEMS


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="Sizes to sample from (default: the problem's default).",
    )
    parser.add_argument("--level", type=int, choices=[1, 2], default=1)
    parser.add_argument("--num-rows", type=int, default=64)
    parser.add_argument(
        "--num-eval-instances",
        type=int,
        default=4,
        help="Hidden evaluation instances per row (level 2).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-time-limit", type=float, default=60.0)
    parser.add_argument(
        "--num-workers", type=int, default=8, help="CP-SAT workers for the references."
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    module = PROBLEMS[args.problem]
    sizes = args.sizes or [module.DEFAULT_SIZE]
    rng = random.Random(args.seed)
    with open(args.out, "w") as f:
        for i in range(args.num_rows):
            size = rng.choice(sizes)
            row = {
                "problem": args.problem,
                "size": size,
                "level": args.level,
                "instance_seed": rng.randrange(2**31),
            }
            if args.level == 1:
                instance = module.generate(row["instance_seed"], size)
                _, row["reference"], row["lower_bound"] = module.reference(
                    instance, args.reference_time_limit, args.num_workers
                )
                row["baseline"] = module.baseline(instance)
            else:
                row["eval_seeds"] = [
                    rng.randrange(2**31) for _ in range(args.num_eval_instances)
                ]
                instances = [module.generate(s, size) for s in row["eval_seeds"]]
                row["eval_references"] = [
                    module.reference(i, args.reference_time_limit, args.num_workers)[1]
                    for i in instances
                ]
                row["eval_baselines"] = [module.baseline(i) for i in instances]
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i + 1}/{args.num_rows}] {json.dumps(row)}", flush=True)


if __name__ == "__main__":
    main()
