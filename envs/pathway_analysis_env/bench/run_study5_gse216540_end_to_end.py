#!/usr/bin/env python3
"""
Study 5 (GSE216540): GEO supplement is TPM-only (`GSE216540_allData_tpm_human.txt.gz`).

This script builds **TPM-derived pseudo-counts** (rounded TPM × scale) for an OpenEnv
**workflow / plumbing test only** — not statistically equivalent to true raw counts.

Default ``--fast-subset`` uses one 28_CM block (8 samples: DMSO vs FULV) for a quick run.
Use ``--full`` to load all sample columns (slow, large memory).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

FTP_TPM = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE216nnn/GSE216540/suppl/"
    "GSE216540_allData_tpm_human.txt.gz"
)

# Fast default: 4 DMSO + 4 FULV replicates in the 28_CM arm (see matrix header).
FAST_SUBSET_COLUMNS: List[str] = [
    "28_CM_DMSO_1",
    "28_CM_DMSO_2",
    "28_CM_DMSO_3",
    "28_CM_DMSO_4",
    "28_CM_FULV_1",
    "28_CM_FULV_2",
    "28_CM_FULV_3",
    "28_CM_FULV_4",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=300) as r, dest.open("wb") as out:
        out.write(r.read())


def _strip_ensembl_version(ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in ids:
        s = str(x).strip('"')
        out.append(re.sub(r"\.\d+$", "", s))
    return out


def _openenv_python(repo_root: Path, env_dir: Path) -> Tuple[List[str], str]:
    if shutil.which("uv"):
        return ["uv", "run", "python"], str(env_dir)
    return [sys.executable], str(repo_root)


def _condition_for_sample(col: str) -> str:
    if "_DMSO_" in col:
        return "DMSO"
    if "_FULV_" in col:
        return "FULV"
    raise ValueError(f"Cannot infer condition from column name: {col}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="Load all sample columns from the TPM matrix (slow, high memory). "
        "Default: fast 28_CM 8-sample DMSO vs FULV subset.",
    )
    ap.add_argument("--tpm-scale", type=float, default=100.0, help="pseudo_count = round(TPM * scale).")
    ap.add_argument("--skip-openenv", action="store_true")
    args = ap.parse_args()
    fast_subset = not bool(args.full)

    repo_root = Path(__file__).resolve().parents[3]
    env_dir = repo_root / "envs" / "pathway_analysis_env"
    bench_dir = env_dir / "bench"
    out_dir = env_dir / "data" / "geo_eval" / "gse216540_tpm_pseudo_benchmark"
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    tpm_path = work / "GSE216540_allData_tpm_human.txt.gz"
    _download(FTP_TPM, tpm_path)

    if fast_subset:
        df = pd.read_csv(tpm_path, sep="\t", usecols=["Geneid"] + FAST_SUBSET_COLUMNS)
        sample_cols = FAST_SUBSET_COLUMNS
    else:
        df = pd.read_csv(tpm_path, sep="\t")
        meta = {"Geneid", "GeneSymbol", "Chromosome", "Start", "End", "Class", "Strand", "Length"}
        sample_cols = [c for c in df.columns if c not in meta]

    tpm = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pseudo = np.round(tpm * float(args.tpm_scale)).clip(lower=0).astype(int)
    pseudo.index = _strip_ensembl_version(df["Geneid"].tolist())
    # Collapse version-stripped duplicate Ensembl IDs
    pseudo = pseudo.groupby(pseudo.index).sum()
    pseudo.index.name = "gene"
    counts_path = out_dir / "gse216540_pseudo_counts.csv.gz"
    pseudo.to_csv(counts_path)

    sample_metadata = {c: _condition_for_sample(c) for c in sample_cols}
    case = {
        "case_id": "GSE216540_28CM_FULV_vs_DMSO_tpm_pseudo" if fast_subset else "GSE216540_FULV_vs_DMSO_tpm_pseudo_full",
        "strict_mode": False,
        "experiment_metadata": {
            "accession": "GSE216540",
            "reference": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE216540",
            "summary": (
                "TPM matrix from GEO supplement; pseudo-counts = round(TPM * "
                f"{args.tpm_scale}). For OpenEnv pipeline testing only, not DESeq2-ground-truth counts."
            ),
        },
        "counts_file": "geo_eval/gse216540_tpm_pseudo_benchmark/gse216540_pseudo_counts.csv.gz",
        "sample_ids": sample_cols,
        "sample_metadata": sample_metadata,
        "conditions": sorted(set(sample_metadata.values())),
        "default_contrast": {"reference": "DMSO", "alternate": "FULV"},
        "analysis_options": {"min_total_count": 10, "padj_alpha": 0.05, "de_query_direction": "both"},
        "enrichr_libraries": ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"],
        "true_pathway": "Unknown (GEO benchmark)",
    }
    case_path = out_dir / "gse216540_case.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    if args.skip_openenv:
        print("[DONE] pseudo-counts + case only:", out_dir)
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
                    "study": "GSE216540",
                    "mode": "tpm_pseudo_counts",
                    "tpm_scale": float(args.tpm_scale),
                    "fast_subset": fast_subset,
                    "contrast": "FULV vs DMSO (reference=DMSO)",
                }
            ),
        ],
        cwd=cwd,
        env=env,
    )
    print("[DONE] Study 5 outputs in:", out_dir)


if __name__ == "__main__":
    main()
