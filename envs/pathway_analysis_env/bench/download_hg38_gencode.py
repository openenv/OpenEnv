#!/usr/bin/env python3

"""
Download a standard hg38/GRCh38 reference (GENCODE) for RNA-seq benchmarks.

Why GENCODE?
  - widely used in publications
  - consistent gene models for human RNA-seq

This script downloads:
  - Genome FASTA (primary assembly) (for STAR)
  - Annotation GTF (for featureCounts and transcript→gene mapping)
  - Transcriptome FASTA (for Salmon)

It writes into an output directory and prints the resulting paths.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
import tempfile
import zlib
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_gzip(path: Path) -> None:
    """Read through EOF to verify gzip CRC/trailer without holding a genome in RAM."""
    with gzip.open(path, "rb") as stream:
        if not stream.read(1):
            raise ValueError(f"Empty reference: {path}")
        while stream.read(1024 * 1024):
            pass


def _download_reference(url: str, path: Path) -> None:
    if path.exists():
        try:
            _validate_gzip(path)
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            raise ValueError(
                f"Invalid cached reference {path}; move this file aside or remove "
                "it explicitly before rerunning. Other files are preserved."
            ) from exc
        return

    # Use the destination filesystem so replacement is atomic. Only this run's
    # temporary file is removed on failure; never publish HTTP bodies/partials.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        _run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "-o",
                str(temporary),
                url,
            ]
        )
        _validate_gzip(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gencode-release", type=str, default="47")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rel = str(args.gencode_release).strip().lstrip("v")
    # GENCODE paths are stable; use the "current" release number.
    fasta_url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{rel}/"
        f"GRCh38.primary_assembly.genome.fa.gz"
    )
    gtf_url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{rel}/"
        f"gencode.v{rel}.annotation.gtf.gz"
    )
    tx_url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{rel}/"
        f"gencode.v{rel}.transcripts.fa.gz"
    )

    fasta_path = out_dir / "GRCh38.primary_assembly.genome.fa.gz"
    gtf_path = out_dir / f"gencode.v{rel}.annotation.gtf.gz"
    tx_path = out_dir / f"gencode.v{rel}.transcripts.fa.gz"

    for url, path in ((fasta_url, fasta_path), (gtf_url, gtf_path), (tx_url, tx_path)):
        _download_reference(url, path)

    print("[OK] FASTA:", fasta_path)
    print("[OK]  GTF:", gtf_path)
    print("[OK]   TX:", tx_path)
    # Lightweight integrity signal (not an official checksum).
    print("[SHA256] FASTA:", _sha256(fasta_path))
    print("[SHA256]  GTF:", _sha256(gtf_path))
    print("[SHA256]   TX:", _sha256(tx_path))


if __name__ == "__main__":
    main()
