#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run timed pathway_analysis_env episodes on a case file and write a markdown report.

Example:
  PYTHONPATH=src:envs python envs/pathway_analysis_env/scripts/run_performance_report.py
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.analysis import pydeseq2_available
from pathway_analysis_env.server.pathway_environment import (
    PathwayEnvironment,
    load_case,
)


def timed_step(
    env: PathwayEnvironment,
    kind: str,
    **kw: Any,
) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    obs = env.step(PathwayAction(action_type=kind, **kw))
    return obs, time.perf_counter() - t0


def run_profiled_episode(
    case_file: str,
    guess: str,
    *,
    strict: bool,
) -> Dict[str, Any]:
    case = load_case(case_file)
    ref = (case.get("default_contrast") or {}).get("reference", "control")
    alt = (case.get("default_contrast") or {}).get("alternate", "treated")
    true_path = case.get("true_pathway", "")

    env = PathwayEnvironment(case_file=case_file)
    t_reset0 = time.perf_counter()
    env.reset(strict=strict)
    t_reset = time.perf_counter() - t_reset0

    timings: List[Dict[str, Any]] = []
    actions: List[str] = []

    def record(name: str, dt: float, obs: Any) -> None:
        timings.append({"action": name, "seconds": dt})
        actions.append(name)
        total_reward_local = float(obs.reward or 0.0)
        timings[-1]["reward"] = total_reward_local

    o, dt = timed_step(env, "inspect_dataset")
    record("inspect_dataset", dt, o)

    o, dt = timed_step(
        env,
        "run_differential_expression",
        condition_a=ref,
        condition_b=alt,
    )
    record("run_differential_expression", dt, o)
    de_genes = o.de_genes or []
    top5 = [r.get("gene") for r in de_genes[:5]]

    if o.done:
        partial_reward = sum(t.get("reward", 0.0) for t in timings)
        return {
            "strict_failure": True,
            "reset_seconds": t_reset,
            "timings": timings,
            "actions": actions,
            "total_reward": partial_reward,
            "true_pathway": true_path,
            "guess": guess,
            "correct": False,
            "top5_de_genes": top5,
            "ora_top": None,
            "rank_true_pathway": None,
        }

    o, dt = timed_step(env, "run_pathway_enrichment")
    record("run_pathway_enrichment", dt, o)
    enrichment = o.pathway_enrichment or []
    ora_names = [r.get("pathway") for r in enrichment]
    rank = ora_names.index(true_path) + 1 if true_path in ora_names else None

    pnames = list((case.get("pathway_genes") or {}).keys())
    if len(pnames) >= 2:
        o, dt = timed_step(
            env,
            "compare_pathways",
            pathway_a=pnames[0],
            pathway_b=pnames[1],
        )
        record("compare_pathways", dt, o)

    o, dt = timed_step(env, "submit_answer", hypothesis=guess)
    record("submit_answer", dt, o)
    correct = bool(o.metadata.get("correct")) if o.metadata else False

    total_reward = sum(t.get("reward", 0.0) for t in timings)

    return {
        "strict_failure": False,
        "reset_seconds": t_reset,
        "timings": timings,
        "actions": actions,
        "total_reward": total_reward,
        "true_pathway": true_path,
        "guess": guess,
        "correct": correct,
        "top5_de_genes": top5,
        "ora_top": ora_names[:5] if ora_names else [],
        "rank_true_pathway": rank,
        "n_genes": len(case.get("counts") or {}),
        "n_samples": len(case.get("sample_ids") or []),
        "experiment_metadata": case.get("experiment_metadata"),
    }


def format_report(
    case_file: str,
    correct_run: Dict[str, Any],
    wrong_run: Dict[str, Any],
) -> str:
    lines = [
        "# Pathway OpenEnv — performance report (synthetic RNA-seq)",
        "",
        f"- **Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"- **Case file:** `{case_file}`",
        f"- **PyDESeq2 available:** {pydeseq2_available()}",
        "",
        "## Synthetic database summary",
        "",
    ]
    em = correct_run.get("experiment_metadata") or {}
    if em:
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for k, v in em.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    lines.extend(
        [
            f"- **Genes in count matrix:** {correct_run.get('n_genes', '—')}",
            f"- **Samples:** {correct_run.get('n_samples', '—')}",
            "",
            "## Task outcome",
            "",
            f"- **Ground-truth pathway:** `{correct_run.get('true_pathway')}`",
            f"- **ORA rank of ground truth:** {correct_run.get('rank_true_pathway')}",
            f"- **Top ORA pathways (first 5):** {correct_run.get('ora_top')}",
            f"- **Top 5 DE genes (by padj ordering):** {correct_run.get('top5_de_genes')}",
            "",
            "### Policy A — submit correct pathway",
            "",
            f"- **Correct:** {correct_run.get('correct')}",
            f"- **Cumulative reward (sum of step rewards):** {correct_run.get('total_reward'):.4f}",
            f"- **Strict failure:** {correct_run.get('strict_failure')}",
            "",
            "### Policy B — submit wrong pathway (`WNT signaling`)",
            "",
            f"- **Correct:** {wrong_run.get('correct')}",
            f"- **Cumulative reward:** {wrong_run.get('total_reward'):.4f}",
            "",
            "## Latency (Policy A, correct guess)",
            "",
            "| Step | Seconds | Step reward |",
            "|------|---------|-------------|",
        ]
    )
    for t in correct_run.get("timings") or []:
        lines.append(
            f"| {t['action']} | {t['seconds']:.4f} | {t.get('reward', 0):.4f} |"
        )
    total_s = correct_run.get("reset_seconds", 0) + sum(
        x["seconds"] for x in (correct_run.get("timings") or [])
    )
    lines.extend(
        [
            "",
            f"- **Reset:** {correct_run.get('reset_seconds', 0):.4f} s",
            f"- **Total wall (reset + steps):** {total_s:.4f} s",
            "",
            "## Notes",
            "",
            "- Latency is measured on the local `PathwayEnvironment` (no HTTP/WebSocket).",
            "- DESeq2 variance estimates are noisy with small *n*; this benchmark uses 4 vs 4 samples.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="fake_rnaseq_detailed.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "outputs"
        / "pathway_performance_report.md",
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Run build_fake_rnaseq_detailed_case.py first",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    case_path = data_dir / args.case

    if args.build_first or not case_path.is_file():
        import subprocess
        import sys

        build = Path(__file__).resolve().parent / "build_fake_rnaseq_detailed_case.py"
        subprocess.check_call([sys.executable, str(build), "--out", str(case_path)])

    correct = run_profiled_episode(args.case, guess="MAPK signaling", strict=False)
    wrong = run_profiled_episode(args.case, guess="WNT signaling", strict=False)

    report = format_report(args.case, correct, wrong)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[Written] {args.out}")


if __name__ == "__main__":
    main()
