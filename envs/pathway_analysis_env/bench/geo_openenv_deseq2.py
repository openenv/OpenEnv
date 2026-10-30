#!/usr/bin/env python3
"""
Run PyDESeq2 + Enrichr ORA for a pathway_analysis_env case JSON (counts on disk).

Invoked from the repo (host) with PYTHONPATH set by the caller, or standalone:

  PYTHONPATH=src:envs python envs/pathway_analysis_env/bench/geo_openenv_deseq2.py \\
    --repo-root . --env-dir envs/pathway_analysis_env \\
    --case-json envs/pathway_analysis_env/data/geo_eval/foo/case.json \\
    --out-dir envs/pathway_analysis_env/data/geo_eval/foo \\
    --summary-extra '{"study":"GSE…"}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--env-dir", type=Path, required=True)
    ap.add_argument("--case-json", type=Path, required=True, help="Absolute path to case JSON")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for summary.json, de_all.json, enrichment.json")
    ap.add_argument(
        "--summary-extra",
        default="{}",
        help="JSON object merged into summary.json (e.g. study label, contrast string).",
    )
    args = ap.parse_args()

    repo_root: Path = args.repo_root.resolve()
    env_dir: Path = args.env_dir.resolve()
    case_path: Path = args.case_json.resolve()
    out_dir: Path = args.out_dir.resolve()

    extra: Dict[str, Any]
    try:
        extra = json.loads(args.summary_extra)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid --summary-extra JSON: {e}") from e
    if not isinstance(extra, dict):
        raise SystemExit("--summary-extra must be a JSON object")

    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "envs"))

    mpl_dir = env_dir / "outputs" / ".mplcache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    os.chdir(env_dir)

    from pathway_analysis_env.server.analysis import (
        build_sample_metadata,
        enrichr_ora,
        filter_counts_by_minimum_total,
        load_counts_csv_as_samples_by_genes,
        merge_analysis_options,
        pick_de_query_genes,
        run_deseq2_contrast,
    )

    case = json.loads(case_path.read_text(encoding="utf-8"))
    opts = merge_analysis_options(case)
    counts_df = load_counts_csv_as_samples_by_genes(
        Path("data") / case["counts_file"],
        sample_ids=case["sample_ids"],
    )
    counts_df, n_genes_in, n_genes_filt = filter_counts_by_minimum_total(
        counts_df, int(opts["min_total_count"])
    )
    if n_genes_filt < 5:
        raise RuntimeError(
            f"After min_total_count={opts['min_total_count']} prefilter, "
            f"only {n_genes_filt} genes remain (need ≥5 for stable DESeq2)."
        )
    meta_df = build_sample_metadata(case["sample_ids"], case["sample_metadata"])
    ref = case["default_contrast"]["reference"]
    alt = case["default_contrast"]["alternate"]
    rows, err = run_deseq2_contrast(
        counts_df,
        meta_df,
        alt,
        ref,
        padj_alpha=float(opts["padj_alpha"]),
    )
    if err:
        raise RuntimeError(err)
    query = pick_de_query_genes(
        rows,
        padj_alpha=float(opts["padj_alpha"]),
        direction=str(opts["de_query_direction"]),
        min_abs_log2fc=float(opts["min_abs_log2fc"]),
    )
    if not query:
        query = [r["gene"] for r in rows[:50]]
    ora, oerr = enrichr_ora(
        query,
        libraries=list(case.get("enrichr_libraries") or []),
        background=list(counts_df.columns),
        top_k=100,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **extra,
        "genes_in_matrix": n_genes_in,
        "genes_after_prefilter": n_genes_filt,
        "n_de_rows_returned": len(rows),
        "n_sig_padj_0.05": sum(1 for r in rows if r.get("significant")),
        "enrichr_error": oerr,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "de_all.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "enrichment.json").write_text(json.dumps(ora, indent=2), encoding="utf-8")
    print("[OK] wrote", out_dir)


if __name__ == "__main__":
    main()
