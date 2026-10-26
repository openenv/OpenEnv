"""Discover and save an honest AWM task split.

Scans candidate tasks (those with a ``code`` verifier) in one or more scenario
domains, drops trivially-passing tasks (a no-op ``reset -> verify`` already
completes), and writes a train/val manifest. The manifest pins exactly which
``(scenario, task_idx)`` pairs were used so the benchmark and the GRPO trainer
are reproducible.

```bash
PYTHONPATH=src:envs uv run uvicorn \\
    envs.agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899 &
PYTHONPATH=src:envs uv run python examples/awm_expert_in_the_loop/build_split.py \\
    --prefixes workflow_automation --out examples/awm_expert_in_the_loop/splits/workflow_automation.json
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import awm_data
import task_select
from run_benchmark import DEFAULT_BASE_URL


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build an honest AWM task split")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["workflow_automation"],
        help="Scenario name prefixes to include",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tasks", type=int, default=130)
    parser.add_argument("--val-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output manifest path (.json)")
    args = parser.parse_args()

    split = await task_select.build_split(
        args.base_url,
        scenario_prefixes=tuple(args.prefixes),
        max_tasks=args.max_tasks,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    manifest = {
        "meta": {
            "prefixes": args.prefixes,
            "verifier_mode": "code",
            "hf_repo_id": awm_data.HF_REPO_ID,
            "hf_repo_type": awm_data.HF_REPO_TYPE,
            "verifier_file": "gen_verifier.pure_code.jsonl",
            "max_tasks": args.max_tasks,
            "filtered": "no-op trivial passes and verifier-less tasks excluded",
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "num_train": len(split.train),
            "num_val": len(split.val),
        },
        "train": [[s, i] for s, i in split.train],
        "val": [[s, i] for s, i in split.val],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"wrote {args.out}: {len(split.train)} train + {len(split.val)} val "
        f"usable tasks across {args.prefixes}"
    )


if __name__ == "__main__":
    asyncio.run(main())
