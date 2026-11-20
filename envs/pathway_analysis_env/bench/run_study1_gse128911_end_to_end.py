#!/usr/bin/env python3
"""
Study 1 (GSE128911): Dataset 2 integer counts → subset CSV → OpenEnv DESeq2 + ORA.

Contrast: fulvestrant vs dmso (reference = dmso).
Samples: SAM24360838, SAM24360839 (DMSO); SAM24360844, SAM24360845 (Fulvestrant).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _openenv_python(repo_root: Path, env_dir: Path) -> tuple[list[str], str]:
    if shutil.which("uv"):
        return ["uv", "run", "python"], str(env_dir)
    return [sys.executable], str(env_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-openenv", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    env_dir = repo_root / "envs" / "pathway_analysis_env"
    bench_dir = env_dir / "bench"
    geo_eval = env_dir / "data" / "geo_eval"
    src_matrix = geo_eval / "GSE128911_RNA-seq_counts_Dataset2.tsv.gz"
    out_dir = geo_eval / "gse128911_mda_mb_134_vi_fulvestrant_vs_dmso"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = ["SAM24360838", "SAM24360839", "SAM24360844", "SAM24360845"]
    df = pd.read_csv(src_matrix, sep="\t", header=2, compression="infer", low_memory=False)
    if "symbol" not in df.columns:
        raise SystemExit(f"Expected 'symbol' column; got {df.columns[:15].tolist()}")
    missing = [s for s in samples if s not in df.columns]
    if missing:
        raise SystemExit(f"Missing sample columns: {missing}")

    sub = df[["symbol"] + samples].copy()
    sub = sub.rename(columns={"symbol": "gene"})
    sub["gene"] = sub["gene"].astype(str)
    for c in samples:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0).round().astype(int)
    sub = sub.groupby("gene", as_index=False).sum()

    counts_path = out_dir / "gse128911_dataset2_subset_counts.csv.gz"
    sub.set_index("gene").to_csv(counts_path)

    meta = {
        "SAM24360838": "dmso",
        "SAM24360839": "dmso",
        "SAM24360844": "fulvestrant",
        "SAM24360845": "fulvestrant",
    }
    case = {
        "case_id": "GSE128911_mda_mb_134_vi_fulvestrant_vs_dmso",
        "strict_mode": False,
        "experiment_metadata": {
            "accession": "GSE128911",
            "reference": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128911",
            "summary": "Dataset 2 count matrix subset (MDA-MB-134-VI; 2×2 DMSO vs Fulvestrant).",
        },
        "counts_file": "geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_dataset2_subset_counts.csv.gz",
        "sample_ids": samples,
        "sample_metadata": meta,
        "conditions": sorted(set(meta.values())),
        "default_contrast": {"reference": "dmso", "alternate": "fulvestrant"},
        "analysis_options": {"min_total_count": 10, "padj_alpha": 0.05, "de_query_direction": "both"},
        "enrichr_libraries": ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"],
        "true_pathway": "Unknown (GEO benchmark)",
    }
    case_path = out_dir / "gse128911_case.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    if args.skip_openenv:
        print("[DONE] counts + case only:", out_dir)
        return

    py, cwd = _openenv_python(repo_root, env_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + ":" + str(repo_root / "envs")
    env.setdefault("MPLCONFIGDIR", str(env_dir / "outputs" / ".mplcache"))
    subprocess.check_call(
        py
        + [
            str(bench_dir / "geo_openenv_deseq2.py"),
            "--repo-root",
            str(repo_root),
            "--env-dir",
            str(env_dir),
            "--case-json",
            str(case_path),
            "--out-dir",
            str(out_dir),
            "--summary-extra",
            json.dumps(
                {
                    "study": "GSE128911",
                    "contrast": "fulvestrant vs dmso (reference=dmso)",
                    "cell_line": "MDA-MB-134-VI",
                    "samples": samples,
                }
            ),
        ],
        cwd=cwd,
        env=env,
    )
    print("[DONE] Study 1 outputs in:", out_dir)


if __name__ == "__main__":
    main()
