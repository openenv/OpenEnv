#!/usr/bin/env python3
"""
End-to-end runner for Study 3 (GSE235350):

  FASTQ (SRA) → hg38/GENCODE → Salmon → OpenEnv DESeq2 + enrichment

Designed to run on the **repo host** with ``uv`` (PyDESeq2 / gseapy), while FASTQ
steps are often run in ``bench/Dockerfile`` where those conda tools exist.

Outputs:
  envs/pathway_analysis_env/data/geo_eval/gse235350_true_counts/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.check_call(cmd, cwd=str(cwd))


def _openenv_python(repo_root: Path, env_dir: Path) -> tuple[list[str], str]:
    if shutil.which("uv"):
        return ["uv", "run", "python"], str(env_dir)
    return [sys.executable], str(repo_root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--fetch-parallelism",
        type=int,
        default=3,
        help="Concurrent fasterq-dump jobs (passed to run_true_counts_pipeline.py).",
    )
    ap.add_argument(
        "--quant-parallelism",
        type=int,
        default=2,
        help="Concurrent salmon quant jobs (passed to run_true_counts_pipeline.py).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only the first N samples from the study sheet (smoke test).",
    )
    ap.add_argument("--skip-openenv", action="store_true", help="Stop after counts + case JSON in work dir.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    env_dir = repo_root / "envs" / "pathway_analysis_env"
    bench_dir = env_dir / "bench"

    out_dir = env_dir / "data" / "geo_eval" / "gse235350_true_counts"
    ref_dir = out_dir / "ref" / "hg38_gencode_v47"
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    samples_json = bench_dir / "studies" / "gse235350_samples.json"
    if int(args.max_samples) > 0:
        raw = json.loads(samples_json.read_text(encoding="utf-8"))
        sub = raw[: int(args.max_samples)]
        samples_json = work_dir / "gse235350_samples_subset.json"
        samples_json.write_text(json.dumps(sub), encoding="utf-8")

    _run(
        [
            "python",
            str(bench_dir / "download_hg38_gencode.py"),
            "--out-dir",
            str(ref_dir),
        ],
        cwd=repo_root,
    )

    gtf = ref_dir / "gencode.v47.annotation.gtf.gz"
    tx = ref_dir / "gencode.v47.transcripts.fa.gz"

    counts_out = work_dir / "gse235350_counts.csv.gz"
    case_out = work_dir / "gse235350_case.json"

    _run(
        [
            "python",
            str(bench_dir / "run_true_counts_pipeline.py"),
            "--method",
            "salmon",
            "--samples-json",
            str(samples_json),
            "--out-dir",
            str(work_dir / "pipeline"),
            "--threads",
            str(int(args.threads)),
            "--fetch-parallelism",
            str(int(args.fetch_parallelism)),
            "--quant-parallelism",
            str(int(args.quant_parallelism)),
            "--gtf",
            str(gtf),
            "--transcripts-fasta",
            str(tx),
            "--counts-csv-gz",
            str(counts_out),
            "--write-case-json",
            "--case-id",
            "GSE235350_MCF7_palbociclib_true_counts",
            "--case-json-path",
            str(case_out),
            "--counts-file-rel",
            "geo_eval/gse235350_true_counts/gse235350_counts.csv.gz",
            "--reference-condition",
            "control",
            "--alternate-condition",
            "palbociclib",
        ],
        cwd=repo_root,
    )

    target_counts_path = out_dir / "gse235350_counts.csv.gz"
    target_case_path = out_dir / "gse235350_case.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_counts_path.write_bytes(counts_out.read_bytes())
    target_case_path.write_text(case_out.read_text(encoding="utf-8"), encoding="utf-8")

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
            str(target_case_path),
            "--out-dir",
            str(out_dir),
            "--summary-extra",
            json.dumps(
                {
                    "study": "GSE235350",
                    "contrast": "palbociclib vs control (reference=control)",
                }
            ),
        ],
        cwd=cwd,
        env=env,
    )
    print("[DONE] Study 3 outputs in:", out_dir)


if __name__ == "__main__":
    main()
