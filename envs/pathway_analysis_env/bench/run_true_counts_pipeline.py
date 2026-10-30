#!/usr/bin/env python3

"""
True-counts RNA-seq pipeline for OpenEnv benchmarks.

This script is designed to run **inside** the benchmark container in `bench/`.

It supports two quantification methods:
  - STAR + featureCounts (alignment-based; default, maximum credibility)
  - Salmon (alignment-free; faster)

Outputs:
  - a gene × sample count matrix (CSV/CSV.GZ) compatible with OpenEnv `counts_file`
  - optional OpenEnv case JSON scaffold (counts_file + sample_ids + sample_metadata)

Notes:
  - Reference genomes/indexes are intentionally user-provided to keep this repo small.
  - This script does not attempt to be a full LIMS/QC system; it’s a reproducible benchmark runner.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import shutil
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    merged = None
    if env is not None:
        merged = os.environ.copy()
        merged.update(env)
    subprocess.check_call(list(cmd), cwd=str(cwd) if cwd else None, env=merged)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_gz_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


def _maybe_decompress_to(
    src: Path,
    *,
    out_dir: Path,
) -> Path:
    """
    STAR genomeGenerate expects uncompressed FASTA/GTF.
    If `src` ends with `.gz`, we write an uncompressed copy in `out_dir` and return it.
    """
    if not str(src).endswith(".gz"):
        return src
    _ensure_dir(out_dir)
    dst = out_dir / src.name[: -len(".gz")]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dst

@dataclass(frozen=True)
class Sample:
    sample_id: str
    sra: str
    condition: str


def parse_samples_json(path: Path) -> List[Sample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("samples.json must be a non-empty JSON list")
    out: List[Sample] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        sid = str(r.get("sample_id") or "").strip()
        sra = str(r.get("sra") or "").strip()
        cond = str(r.get("condition") or "").strip()
        if not sid or not sra or not cond:
            raise ValueError(f"Invalid sample entry: {r}")
        out.append(Sample(sample_id=sid, sra=sra, condition=cond))
    return out


def build_tx2gene_from_gtf(*, gtf_gz: Path, out_tsv: Path) -> Path:
    """
    Build a transcript→gene mapping for Salmon `--geneMap` from plain or gzip GTF.

    Output format (TSV):
      transcript_id <tab> gene_name

    We prefer gene_name for downstream interpretability; fallback to gene_id when gene_name missing.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        return out_tsv

    # GTF attributes: key "value";
    re_tid = re.compile(r'transcript_id "([^"]+)"')
    re_gn = re.compile(r'gene_name "([^"]+)"')
    re_gid = re.compile(r'gene_id "([^"]+)"')

    seen = set()
    opener = gzip.open if str(gtf_gz).endswith(".gz") else open
    with (
        opener(gtf_gz, "rt", encoding="utf-8", errors="replace") as f,
        out_tsv.open("w", encoding="utf-8") as out,
    ):
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("transcript", "exon"):
                continue
            attrs = parts[8]
            m_tid = re_tid.search(attrs)
            if not m_tid:
                continue
            tid = m_tid.group(1)
            if tid in seen:
                continue
            m_gn = re_gn.search(attrs)
            m_gid = re_gid.search(attrs)
            gene = (m_gn.group(1) if m_gn else None) or (
                m_gid.group(1) if m_gid else None
            )
            if not gene:
                continue
            out.write(f"{tid}\t{gene}\n")
            seen.add(tid)
    return out_tsv


def _fasterq_dump_one(
    s: Sample, *, out_dir: Path, threads_per_job: int
) -> Tuple[str, str, Tuple[Path, Optional[Path]]]:
    """Run fasterq-dump for one SRR if outputs are missing; return (sample_id, sra, (fq1, fq2_or_none))."""
    fq1 = out_dir / f"{s.sra}_1.fastq"
    fq2 = out_dir / f"{s.sra}_2.fastq"
    fq_single = out_dir / f"{s.sra}.fastq"
    if not (fq1.exists() or fq2.exists() or fq_single.exists()):
        _run(
            [
                "fasterq-dump",
                "-e",
                str(max(1, threads_per_job)),
                "-O",
                str(out_dir),
                s.sra,
            ]
        )
    if fq1.exists() or fq2.exists():
        if not (fq1.exists() and fq2.exists()):
            raise RuntimeError(
                f"Incomplete paired FASTQ for {s.sra}; restore both mates or "
                "remove the partial files and rerun fasterq-dump."
            )
        return s.sample_id, s.sra, (fq1, fq2)
    if fq_single.exists():
        return s.sample_id, s.sra, (fq_single, None)
    raise RuntimeError(f"FASTQ not found after fasterq-dump for {s.sra}")


def download_fastq(
    samples: Sequence[Sample],
    *,
    out_dir: Path,
    threads: int,
    fetch_parallelism: int = 1,
) -> Dict[str, Tuple[Path, Optional[Path]]]:
    """
    Download FASTQs using sra-tools (fasterq-dump).

    When ``fetch_parallelism`` > 1, multiple SRRs are fetched concurrently (bounded
    per-job thread counts to avoid oversubscribing the machine).

    Returns mapping: sample_id -> (fastq1, fastq2_or_none)
    """
    _ensure_dir(out_dir)
    n = max(1, len(samples))
    par = max(1, min(int(fetch_parallelism), n))
    threads_per = max(1, int(threads) // par)

    fastqs: Dict[str, Tuple[Path, Optional[Path]]] = {}
    if par <= 1:
        for s in samples:
            sid, _sra, pair = _fasterq_dump_one(s, out_dir=out_dir, threads_per_job=int(threads))
            fastqs[sid] = pair
        return fastqs

    with concurrent.futures.ThreadPoolExecutor(max_workers=par) as ex:
        futs = [
            ex.submit(_fasterq_dump_one, s, out_dir=out_dir, threads_per_job=threads_per)
            for s in samples
        ]
        for fut in concurrent.futures.as_completed(futs):
            sid, _sra, pair = fut.result()
            fastqs[sid] = pair

    # Preserve stable ordering for downstream column order
    return {s.sample_id: fastqs[s.sample_id] for s in samples}


def build_salmon_index(*, transcripts_fasta: Path, out_dir: Path, threads: int) -> None:
    _ensure_dir(out_dir)
    # Salmon can read gz in many builds, but we keep behavior consistent.
    tx_dir = out_dir.parent / "ref_unzipped"
    transcripts_fasta = _maybe_decompress_to(transcripts_fasta, out_dir=tx_dir)
    # If partial index exists, clear it.
    # Salmon ≥1.10 expects a finished index to contain version metadata.
    marker = out_dir / "versionInfo.json"
    if out_dir.exists() and not marker.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    if marker.exists():
        return
    env = os.environ.copy()
    # Avoid flaky version-server checks in CI / restricted networks.
    env.setdefault("SALMON_NO_VERSION_CHECK", "1")
    _run(
        [
            "salmon",
            "index",
            "-t",
            str(transcripts_fasta),
            "-i",
            str(out_dir),
            "-p",
            str(max(1, threads)),
            "--gencode",
        ],
        env=env,
    )


def quantify_salmon(
    *,
    salmon_index: Path,
    sample_id: str,
    fastq1: Path,
    fastq2: Optional[Path],
    out_dir: Path,
    threads: int,
    tx2gene_tsv: Path,
) -> Path:
    """
    Run Salmon quant and return path to `quant.genes.sf`.
    """
    sdir = out_dir / sample_id
    if (sdir / "quant.genes.sf").exists():
        return sdir / "quant.genes.sf"
    _ensure_dir(sdir)

    env = os.environ.copy()
    env.setdefault("SALMON_NO_VERSION_CHECK", "1")

    cmd = [
        "salmon",
        "quant",
        "-i",
        str(salmon_index),
        "-l",
        "A",
        "--validateMappings",
        "--gcBias",
        "--geneMap",
        str(tx2gene_tsv),
        "-p",
        str(max(1, threads)),
        "--numBootstraps",
        "0",
        "-o",
        str(sdir),
    ]
    if fastq2 is None:
        cmd += ["-r", str(fastq1)]
    else:
        cmd += ["-1", str(fastq1), "-2", str(fastq2)]
    _run(cmd, env=env)
    qg = sdir / "quant.genes.sf"
    if not qg.exists():
        raise RuntimeError(f"Salmon did not produce {qg}")
    return qg


def salmon_genes_to_openenv_counts(
    *,
    quant_genes_by_sample: Dict[str, Path],
    out_counts_csv_gz: Path,
) -> None:
    """
    Build OpenEnv counts (genes × samples) from Salmon `quant.genes.sf` NumReads.
    """
    dfs = []
    for sid, p in quant_genes_by_sample.items():
        df = pd.read_csv(p, sep="\t")
        if "Name" not in df.columns or "NumReads" not in df.columns:
            raise ValueError(f"Unexpected salmon quant.genes.sf schema: {p}")
        d = df[["Name", "NumReads"]].copy()
        d = d.rename(columns={"Name": "gene", "NumReads": sid}).set_index("gene")
        dfs.append(d)
    out = pd.concat(dfs, axis=1).fillna(0.0)
    out = out.round().astype(int)
    _ensure_dir(out_counts_csv_gz.parent)
    out.to_csv(out_counts_csv_gz)


def build_star_index(*, genome_fasta: Path, gtf: Path, out_dir: Path, threads: int) -> None:
    _ensure_dir(out_dir)
    # STAR requires uncompressed inputs.
    ref_dir = out_dir.parent / "ref_unzipped"
    genome_fasta = _maybe_decompress_to(genome_fasta, out_dir=ref_dir)
    gtf = _maybe_decompress_to(gtf, out_dir=ref_dir)

    # If a previous attempt created a partial index directory, clear it.
    marker = out_dir / "genomeParameters.txt"
    if out_dir.exists() and not marker.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass
    cmd: List[str] = [
        "STAR",
        "--runThreadN",
        str(max(1, threads)),
        "--runMode",
        "genomeGenerate",
        "--genomeDir",
        str(out_dir),
        "--genomeFastaFiles",
        str(genome_fasta),
        "--sjdbGTFfile",
        str(gtf),
    ]

    # On machines with limited RAM (common with Docker Desktop), hg38 genomeGenerate can OOM-kill.
    # These parameters reduce STAR index memory footprint while preserving alignment correctness.
    try:
        fasta_size = genome_fasta.stat().st_size
    except FileNotFoundError:
        fasta_size = 0
    if fasta_size > 500_000_000:  # ~hg38 scale
        cmd += ["--genomeSAindexNbases", "14"]

    _run(cmd)


def align_star(
    *,
    star_index: Path,
    sample_id: str,
    fastq1: Path,
    fastq2: Optional[Path],
    out_dir: Path,
    threads: int,
) -> Path:
    """
    Returns path to coordinate-sorted BAM.
    """
    _ensure_dir(out_dir)
    prefix = out_dir / f"{sample_id}."
    cmd = [
        "STAR",
        "--runThreadN",
        str(max(1, threads)),
        "--genomeDir",
        str(star_index),
        "--readFilesIn",
        str(fastq1),
    ]
    if fastq2 is not None:
        cmd.append(str(fastq2))
    cmd += [
        "--outFileNamePrefix",
        str(prefix),
        "--outSAMtype",
        "BAM",
        "SortedByCoordinate",
    ]
    _run(cmd)
    bam = out_dir / f"{sample_id}.Aligned.sortedByCoord.out.bam"
    if not bam.exists():
        raise RuntimeError(f"STAR did not produce BAM for {sample_id}: expected {bam}")
    return bam


def quantify_featurecounts(
    *,
    gtf: Path,
    bams: Dict[str, Path],
    out_path: Path,
    threads: int,
    paired_end: bool,
) -> Path:
    """
    Run featureCounts on all BAMs and return the raw featureCounts output path.
    """
    _ensure_dir(out_path.parent)
    bam_paths = [bams[sid] for sid in sorted(bams.keys())]
    cmd = [
        "featureCounts",
        "-T",
        str(max(1, threads)),
        "-a",
        str(gtf),
        "-o",
        str(out_path),
        "-g",
        "gene_id",
    ]
    if paired_end:
        cmd += ["-p", "--countReadPairs"]
    cmd += [str(p) for p in bam_paths]
    _run(cmd)
    if not out_path.exists():
        raise RuntimeError(f"featureCounts did not write {out_path}")
    return out_path


def featurecounts_to_openenv_counts(
    *,
    featurecounts_txt: Path,
    sample_ids: List[str],
    out_counts_csv_gz: Path,
) -> None:
    """
    Convert featureCounts output into OpenEnv-compatible counts:
      - rows: genes
      - columns: sample_ids
    """
    df = pd.read_csv(featurecounts_txt, sep="\t", comment="#")
    if "Geneid" not in df.columns:
        raise ValueError("featureCounts output missing 'Geneid' column")

    # featureCounts writes one column per BAM path; we assume order matches sample_ids sort.
    count_cols = [c for c in df.columns if c not in ("Geneid", "Chr", "Start", "End", "Strand", "Length")]
    if len(count_cols) != len(sample_ids):
        raise ValueError(f"featureCounts produced {len(count_cols)} sample columns, expected {len(sample_ids)}")

    out = df[["Geneid"] + count_cols].copy()
    out.columns = ["gene"] + sample_ids
    out = out.set_index("gene")
    _ensure_dir(out_counts_csv_gz.parent)
    out.to_csv(out_counts_csv_gz)


def write_openenv_case(
    *,
    case_id: str,
    counts_file_rel: str,
    samples: Sequence[Sample],
    reference_condition: str,
    alternate_condition: str,
    out_case_json: Path,
) -> None:
    sample_ids = [s.sample_id for s in samples]
    sample_metadata = {s.sample_id: s.condition for s in samples}
    case = {
        "case_id": case_id,
        "strict_mode": False,
        "experiment_metadata": {
            "source": "SRA",
            "summary": "True-counts benchmark generated from FASTQs using STAR+featureCounts or Salmon (see bench/run_true_counts_pipeline.py).",
        },
        "counts_file": counts_file_rel,
        "sample_ids": sample_ids,
        "sample_metadata": sample_metadata,
        "conditions": sorted(set(sample_metadata.values())),
        "default_contrast": {"reference": reference_condition, "alternate": alternate_condition},
        "analysis_options": {"min_total_count": 10, "padj_alpha": 0.05, "de_query_direction": "both"},
        "enrichr_libraries": ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"],
        "true_pathway": "Unknown (true-counts benchmark)",
    }
    _ensure_dir(out_case_json.parent)
    out_case_json.write_text(json.dumps(case, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["star", "salmon"], default="star")
    ap.add_argument("--samples-json", required=True, type=Path, help="JSON list of {sample_id,sra,condition}")
    ap.add_argument("--out-dir", required=True, type=Path, help="Benchmark working/output directory")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument(
        "--fetch-parallelism",
        type=int,
        default=3,
        help="Concurrent fasterq-dump jobs (I/O bound). Per-job threads ~= --threads / this value.",
    )
    ap.add_argument(
        "--quant-parallelism",
        type=int,
        default=2,
        help="Concurrent salmon quant (or STAR align) jobs. Per-job threads ~= --threads / this value.",
    )

    # STAR+featureCounts inputs
    ap.add_argument("--genome-fasta", type=Path, help="Genome FASTA (required for --method star if --star-index not provided)")
    ap.add_argument("--gtf", type=Path, help="Gene annotation GTF (required for --method star)")
    ap.add_argument("--star-index", type=Path, help="STAR index directory (if missing, it will be built)")

    # Salmon inputs
    ap.add_argument("--transcripts-fasta", type=Path, help="Transcriptome FASTA (required for --method salmon if --salmon-index not provided)")
    ap.add_argument("--salmon-index", type=Path, help="Salmon index directory (if missing, it will be built)")

    # Outputs
    ap.add_argument("--counts-csv-gz", type=Path, help="Where to write OpenEnv counts CSV.GZ")
    ap.add_argument("--write-case-json", action="store_true")
    ap.add_argument("--case-id", default="true_counts_case")
    ap.add_argument("--case-json-path", type=Path, help="Where to write the OpenEnv case JSON (if --write-case-json)")
    ap.add_argument("--counts-file-rel", default="geo_eval/true_counts.csv.gz", help="counts_file path relative to env data dir")
    ap.add_argument("--reference-condition", default="control")
    ap.add_argument("--alternate-condition", default="treated")

    args = ap.parse_args()

    samples = parse_samples_json(args.samples_json)
    out_dir: Path = args.out_dir
    _ensure_dir(out_dir)

    # Download FASTQs
    fq_dir = out_dir / "fastq"
    fastqs = download_fastq(
        samples,
        out_dir=fq_dir,
        threads=int(args.threads),
        fetch_parallelism=int(args.fetch_parallelism),
    )

    qpar = max(1, min(int(args.quant_parallelism), len(samples)))
    threads_per_quant = max(1, int(args.threads) // qpar)

    if args.method == "salmon":
        if args.gtf is None:
            raise SystemExit("--gtf is required for --method salmon (to build transcript→gene map)")
        salmon_index = args.salmon_index or (out_dir / "salmon_index")
        if not (salmon_index / "versionInfo.json").exists():
            if args.transcripts_fasta is None:
                raise SystemExit("--transcripts-fasta is required to build Salmon index (or provide --salmon-index)")
            build_salmon_index(
                transcripts_fasta=args.transcripts_fasta,
                out_dir=salmon_index,
                threads=int(args.threads),
            )

        tx2gene = build_tx2gene_from_gtf(
            gtf_gz=args.gtf,
            out_tsv=out_dir / "tx2gene.tsv",
        )

        qdir = out_dir / "salmon"
        qgenes: Dict[str, Path] = {}
        if qpar <= 1:
            for s in samples:
                fq1, fq2 = fastqs[s.sample_id]
                qgenes[s.sample_id] = quantify_salmon(
                    salmon_index=salmon_index,
                    sample_id=s.sample_id,
                    fastq1=fq1,
                    fastq2=fq2,
                    out_dir=qdir,
                    threads=int(args.threads),
                    tx2gene_tsv=tx2gene,
                )
        else:

            def _salmon_job(s: Sample) -> Tuple[str, Path]:
                fq1, fq2 = fastqs[s.sample_id]
                pth = quantify_salmon(
                    salmon_index=salmon_index,
                    sample_id=s.sample_id,
                    fastq1=fq1,
                    fastq2=fq2,
                    out_dir=qdir,
                    threads=threads_per_quant,
                    tx2gene_tsv=tx2gene,
                )
                return s.sample_id, pth

            with concurrent.futures.ThreadPoolExecutor(max_workers=qpar) as ex:
                futs = [ex.submit(_salmon_job, s) for s in samples]
                for fut in concurrent.futures.as_completed(futs):
                    sid, pth = fut.result()
                    qgenes[sid] = pth
            qgenes = {s.sample_id: qgenes[s.sample_id] for s in samples}

        counts_out = args.counts_csv_gz or (out_dir / "openenv_counts.csv.gz")
        salmon_genes_to_openenv_counts(
            quant_genes_by_sample=qgenes,
            out_counts_csv_gz=counts_out,
        )

        if args.write_case_json:
            case_json = args.case_json_path or (out_dir / "openenv_case.json")
            write_openenv_case(
                case_id=str(args.case_id),
                counts_file_rel=str(args.counts_file_rel),
                samples=samples,
                reference_condition=str(args.reference_condition),
                alternate_condition=str(args.alternate_condition),
                out_case_json=case_json,
            )

        print("[DONE] counts:", counts_out)
        return

    # STAR + featureCounts (Subread >=2.0.2).
    layouts = {fastqs[s.sample_id][1] is not None for s in samples}
    if len(layouts) != 1:
        raise ValueError(
            "STAR featureCounts requires a uniform single-end or paired-end batch."
        )
    paired_end = layouts.pop()
    if args.gtf is None:
        raise SystemExit("--gtf is required for --method star")

    star_index = args.star_index or (out_dir / "star_index")
    marker = star_index / "genomeParameters.txt"
    if not marker.exists():
        if args.genome_fasta is None:
            raise SystemExit("--genome-fasta is required to build STAR index (or provide --star-index)")
        build_star_index(
            genome_fasta=args.genome_fasta,
            gtf=args.gtf,
            out_dir=star_index,
            threads=int(args.threads),
        )

    bam_dir = out_dir / "bam"
    bams: Dict[str, Path] = {}
    if qpar <= 1:
        for s in samples:
            fq1, fq2 = fastqs[s.sample_id]
            bams[s.sample_id] = align_star(
                star_index=star_index,
                sample_id=s.sample_id,
                fastq1=fq1,
                fastq2=fq2,
                out_dir=bam_dir,
                threads=int(args.threads),
            )
    else:

        def _star_job(s: Sample) -> Tuple[str, Path]:
            fq1, fq2 = fastqs[s.sample_id]
            bam = align_star(
                star_index=star_index,
                sample_id=s.sample_id,
                fastq1=fq1,
                fastq2=fq2,
                out_dir=bam_dir,
                threads=threads_per_quant,
            )
            return s.sample_id, bam

        with concurrent.futures.ThreadPoolExecutor(max_workers=qpar) as ex:
            futs = [ex.submit(_star_job, s) for s in samples]
            for fut in concurrent.futures.as_completed(futs):
                sid, bam = fut.result()
                bams[sid] = bam
        bams = {s.sample_id: bams[s.sample_id] for s in samples}

    fc_out = out_dir / "featureCounts.txt"
    quantify_featurecounts(
        gtf=args.gtf,
        bams=bams,
        out_path=fc_out,
        threads=int(args.threads),
        paired_end=paired_end,
    )

    counts_out = args.counts_csv_gz or (out_dir / "openenv_counts.csv.gz")
    # featureCounts sample column order matches sorted BAM list in quantify_featurecounts
    sample_ids_sorted = sorted(bams.keys())
    featurecounts_to_openenv_counts(
        featurecounts_txt=fc_out,
        sample_ids=sample_ids_sorted,
        out_counts_csv_gz=counts_out,
    )

    if args.write_case_json:
        case_json = args.case_json_path or (out_dir / "openenv_case.json")
        write_openenv_case(
            case_id=str(args.case_id),
            counts_file_rel=str(args.counts_file_rel),
            samples=samples,
            reference_condition=str(args.reference_condition),
            alternate_condition=str(args.alternate_condition),
            out_case_json=case_json,
        )

    print("[DONE] counts:", counts_out)


if __name__ == "__main__":
    main()

