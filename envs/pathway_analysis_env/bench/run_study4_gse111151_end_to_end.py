#!/usr/bin/env python3
"""
Study 4 (GSE111151): merge per-sample GEO supplement count tables → OpenEnv DESeq2 + ORA.

Uses legacy FTP-visible supplements ``GSE111151_GSM1417177.txt.gz`` … ``…7184`` (8 samples),
mapped to current GSM accessions ``GSM3024053`` … ``GSM3024060`` (MCF-7 … ZR-75-1 Tam2).

Default contrast: tamoxifen_resistant vs parental (reference = parental).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/GSE111151/suppl"

# GEO ``filelist.txt`` lists newer ``GSM3024053_*.txt.gz`` names, but this FTP path
# currently serves the legacy ``GSE111151_GSM1417177.txt.gz`` … ``1417184`` set (8 files).
# Those eight files align with the first eight series-matrix columns (MCF-7 … ZR-75-1 Tam2).
# BT-474 arm samples are not included here (no matching legacy per-sample supplements on this mirror).
LEGACY_SUPPLEMENT_FILES: List[str] = [f"GSE111151_GSM{1417177 + i}.txt.gz" for i in range(8)]

CURRENT_GSM_ORDER: List[str] = [
    "GSM3024053",
    "GSM3024054",
    "GSM3024055",
    "GSM3024056",
    "GSM3024057",
    "GSM3024058",
    "GSM3024059",
    "GSM3024060",
]

# Parental vs resistant (cell-line–aware grouping for one pooled contrast).
GSM_CONDITION: Dict[str, str] = {
    "GSM3024053": "parental",
    "GSM3024054": "tamoxifen_resistant",
    "GSM3024055": "parental",
    "GSM3024056": "tamoxifen_resistant",
    "GSM3024057": "tamoxifen_resistant",
    "GSM3024058": "parental",
    "GSM3024059": "tamoxifen_resistant",
    "GSM3024060": "tamoxifen_resistant",
    "GSM3024061": "parental",
    "GSM3024062": "tamoxifen_resistant",
    "GSM3024063": "tamoxifen_resistant",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as out:
        out.write(r.read())


def _load_one_counts(path: Path, colname: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", compression="infer")
    if "gene_name" not in df.columns or "counts" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    s = df.groupby(df["gene_name"].astype(str).str.strip())["counts"].sum()
    s = s.astype(float).round().astype(int)
    s.name = colname
    return s


def _openenv_python(repo_root: Path, env_dir: Path) -> Tuple[List[str], str]:
    if shutil.which("uv"):
        return ["uv", "run", "python"], str(env_dir)
    return [sys.executable], str(repo_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fetch-parallelism",
        type=int,
        default=5,
        help="Concurrent supplement downloads (default: 5).",
    )
    ap.add_argument("--skip-openenv", action="store_true", help="Only build counts + case JSON.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    env_dir = repo_root / "envs" / "pathway_analysis_env"
    bench_dir = env_dir / "bench"
    out_dir = env_dir / "data" / "geo_eval" / "gse111151_tamoxifen_benchmark"
    dl_dir = out_dir / "work" / "suppl"
    out_dir.mkdir(parents=True, exist_ok=True)

    par = max(1, min(int(args.fetch_parallelism), len(LEGACY_SUPPLEMENT_FILES)))

    def _dl(pair: Tuple[str, str]) -> Tuple[Path, str]:
        fn, gsm = pair
        url = f"{FTP_BASE}/{fn}"
        dest = dl_dir / fn
        _download(url, dest)
        return dest, gsm

    pairs = list(zip(LEGACY_SUPPLEMENT_FILES, CURRENT_GSM_ORDER, strict=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=par) as ex:
        resolved = list(ex.map(_dl, pairs))

    series_list = [_load_one_counts(p, gsm) for p, gsm in resolved]
    counts = pd.concat(series_list, axis=1).fillna(0).astype(int)
    counts.index.name = "gene"
    counts_path = out_dir / "gse111151_counts.csv.gz"
    counts.to_csv(counts_path)

    sample_ids = list(CURRENT_GSM_ORDER)
    sample_metadata = {g: GSM_CONDITION[g] for g in sample_ids}

    case = {
        "case_id": "GSE111151_tamoxifen_resistance_parental",
        "strict_mode": False,
        "experiment_metadata": {
            "accession": "GSE111151",
            "reference": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111151",
            "summary": "Merged author per-sample raw counts (GEO legacy supplement files GSE111151_GSM1417177–84 → GSM3024053–60); 8/11 matrix columns (BT-474 arm not on this FTP mirror).",
        },
        "counts_file": "geo_eval/gse111151_tamoxifen_benchmark/gse111151_counts.csv.gz",
        "sample_ids": sample_ids,
        "sample_metadata": sample_metadata,
        "conditions": sorted(set(sample_metadata.values())),
        "default_contrast": {"reference": "parental", "alternate": "tamoxifen_resistant"},
        "analysis_options": {"min_total_count": 10, "padj_alpha": 0.05, "de_query_direction": "both"},
        "enrichr_libraries": ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"],
        "true_pathway": "Unknown (GEO benchmark)",
    }
    case_path = out_dir / "gse111151_case.json"
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
                    "study": "GSE111151",
                    "contrast": "tamoxifen_resistant vs parental (reference=parental)",
                }
            ),
        ],
        cwd=cwd,
        env=env,
    )
    print("[DONE] Study 4 outputs in:", out_dir)


if __name__ == "__main__":
    main()
