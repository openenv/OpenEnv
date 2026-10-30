# Benchmark toolchain (STAR/featureCounts/Salmon)

This folder provides an **optional** “gold-standard” RNA-seq toolchain for benchmarking OpenEnv on GEO studies that only provide FASTQs (or when you want to validate counts-derived DESeq2 from first principles).

It is intentionally separated from the normal server image to keep the environment lightweight for typical use.

## Build the benchmark image

From the repo root:

```bash
# Apple Silicon note: build/run as linux/amd64 for bioconda bioinformatics tools.
docker build --platform linux/amd64 -t openenv-pathway-bench:latest -f envs/pathway_analysis_env/bench/Dockerfile .
```

## What’s inside

The benchmark image installs (via bioconda):

- `STAR` (alignment)
- `featureCounts` (from `subread`)
- `salmon` (alignment-free quantification)
- `sra-tools` (download FASTQs)
- `samtools`, `fastqc`, `multiqc` (QC + utilities)

## Next step (workflow)

To run a true-counts benchmark you typically:

1. Download FASTQs for your study’s samples (SRA)
2. Quantify expression into a **gene-level count matrix** (STAR+featureCounts or Salmon)
3. Feed the counts into OpenEnv (case JSON with `counts_file` + `sample_ids` + `sample_metadata`)
4. Compare OpenEnv DE to author DE (if available)

## STAR vs Salmon (which to choose?)

If your priority is **maximum credibility** (paper-style “raw counts → DESeq2”), use:

- **STAR + featureCounts** (alignment-based; **recommended default**)

Use **Salmon** when you need speed / many studies and are comfortable with transcript-first quantification and gene-level summarization.

## `run_true_counts_pipeline.py`

Implements **`--method star`** (STAR + featureCounts) and **`--method salmon`** (Salmon → gene-level counts).

Speed-oriented flags (defaults tuned for laptops):

- **`--fetch-parallelism`** (default `3`): concurrent `fasterq-dump` jobs; per-job threads ≈ `--threads / fetch-parallelism`.
- **`--quant-parallelism`** (default `2`): concurrent `salmon quant` or STAR per-sample alignments; per-job threads ≈ `--threads / quant-parallelism`.
- Salmon passes **`--numBootstraps 0`** for faster quantification.

## Getting hg38 FASTA + GTF (best default)

For maximum credibility, match the common “hg38/GRCh38 + GENCODE annotation” setup:

- Download reference files:

```bash
python envs/pathway_analysis_env/bench/download_hg38_gencode.py --out-dir /work/ref/hg38_gencode_v47
```

The downloader retries HTTP failures and validates the full gzip stream before atomically publishing each reference. Valid cached files are reused without downloading. An invalid cached file stops the run and is preserved: move aside or explicitly remove only the file named in the error before rerunning. Temporary downloads from the current run are cleaned up on failure. The pipeline accepts both plain `.gtf` and compressed `.gtf.gz` annotations for STAR and Salmon.

This writes:
- `GRCh38.primary_assembly.genome.fa.gz` (FASTA)
- `gencode.v47.annotation.gtf.gz` (GTF)

## Study 3 (GSE235350) note

GSE235350’s GEO supplements contain **bigWig tracks** (`*.bw`) but **no gene count matrix**.
To run DESeq2 in OpenEnv you must generate counts from SRA FASTQs.

A ready sample sheet is provided:

- `bench/studies/gse235350_samples.json`

To run Study 3 end-to-end (downloads reference + FASTQs + counts + OpenEnv outputs):

- `bench/run_study3_gse235350_end_to_end.py`

Example (FASTQ + Salmon inside Docker; **OpenEnv DESeq2 step expects `uv` on the host** unless you extend the image):

```bash
docker run --platform linux/amd64 --rm -v "$PWD":/work openenv-pathway-bench:latest \
  python envs/pathway_analysis_env/bench/run_study3_gse235350_end_to_end.py --skip-openenv
```

Then on the host (repo root), run DESeq2 + Enrichr:

```bash
cd envs/pathway_analysis_env && PYTHONPATH="../../src:../../envs" uv run python bench/geo_openenv_deseq2.py \
  --repo-root ../.. --env-dir . \
  --case-json data/geo_eval/gse235350_true_counts/gse235350_case.json \
  --out-dir data/geo_eval/gse235350_true_counts \
  --summary-extra '{"study":"GSE235350","contrast":"palbociclib vs control (reference=control)"}'
```

Smaller smoke test (first *N* samples in the sheet):

```bash
python envs/pathway_analysis_env/bench/run_study3_gse235350_end_to_end.py --max-samples 2 --threads 8 --fetch-parallelism 2 --quant-parallelism 2
```

## Study 4 / Study 5 (counts or TPM on GEO)

These runners download GEO supplements and call **`bench/geo_openenv_deseq2.py`** (needs **`uv`** in `PATH`, cwd `envs/pathway_analysis_env`):

- **Study 4 — GSE111151:** `bench/run_study4_gse111151_end_to_end.py` (legacy per-sample count supplements; 8 samples on the FTP mirror).
- **Study 5 — GSE216540:** `bench/run_study5_gse216540_end_to_end.py` (TPM matrix → **pseudo-counts** for pipeline testing; default fast 8-sample subset; add `--full` for all columns).

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/bench/run_study4_gse111151_end_to_end.py
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/bench/run_study5_gse216540_end_to_end.py
```


### Counting and filtering contract

The STAR path requires Subread **2.0.2 or newer** (also enforced by the benchmark image's package constraint). Paired FASTQs use `featureCounts -p --countReadPairs` to count fragments; single-end FASTQs use neither flag. Mixed single/paired batches are rejected before alignment. See the [Subread release notes](https://subread.sourceforge.net/) and [user guide](https://subread.sourceforge.net/SubreadUsersGuide.pdf). Earlier versions gave `-p` different semantics and are unsupported here.

The GEO DE runner applies the case's `min_total_count` filter before fitting and uses the retained genes for its ORA background, matching the environment. Fewer than five retained genes aborts analysis. Summaries distinguish `genes_in_matrix` from `genes_after_prefilter`.

Gene identifiers remain the annotation's `gene_id`; no Ensembl-to-symbol mapping is implied. Annotation provenance, identifier mapping, strandedness, and biological validation remain study-specific responsibilities.
