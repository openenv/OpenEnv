# GEO benchmark studies (OpenEnv `pathway_analysis_env`)

This document tracks **external RNA-seq benchmarks** used to sanity-check OpenEnv’s **DESeq2 + ORA** pipeline. Studies are ordered **easiest first** (processed counts available on GEO → harder jobs).

## Shared pipeline settings (benchmark runs)

| Setting | Value |
|--------|--------|
| Differential expression | PyDESeq2 (`run_deseq2_contrast` in `server/analysis.py`) |
| Significance | Benjamini–Hochberg **padj ≤ 0.05** |
| Count prefilter | `filter_counts_by_minimum_total` (minimum counts across samples; exact cutoff recorded per study in `summary.json`) |
| Pathway ORA | **Enrichr** via **gseapy** (`enrichr_ora`), multiple libraries merged and sorted by adjusted *P* |
| ORA query genes | Typically **padj ≤ 0.05** DE genes (see `pick_de_query_genes`) |

Artifacts for each completed study live under  
`envs/pathway_analysis_env/data/geo_eval/<study_folder>/`.

---

## Study 1 — GSE128911 (easiest; **complete**)

**Biology:** MDA-MB-134-VI — **Fulvestrant vs DMSO** (Dataset 2 in the submission).

**Why easiest:** GEO ships a **gene × sample integer count matrix**:  
`data/geo_eval/GSE128911_RNA-seq_counts_Dataset2.tsv.gz`.

**Contrast**

- **Reference:** DMSO — samples `SAM24360838`, `SAM24360839`
- **Alternate:** Fulvestrant — samples `SAM24360844`, `SAM24360845`
- **DESeq2 contrast:** `fulvestrant` vs `dmso` (reference = DMSO)

**OpenEnv outputs** (`data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/`)

| File | Contents |
|------|-----------|
| `summary.json` | Matrix dimensions, prefilter counts, **n_sig_padj_0.05** |
| `de_all.json` | Full DE table |
| `de_top50.csv` | Top 50 by significance |
| `enrichment.json` | Top pathway hits (Enrichr), up to 50 rows after merge/sort |

**Numbers (from `summary.json`)**

| Metric | Value |
|--------|------:|
| Genes in matrix | 30,590 |
| Genes after prefilter | 17,540 |
| DE rows returned | 17,540 |
| **Significant padj ≤ 0.05** | **310** |

**Biological face validity**

- **`ESR1`** is strongly **down** in Fulvestrant vs vehicle (top DE table), consistent with ER antagonism/degradation.
- Other top downregulated genes include estrogen pathway-linked transcripts (**`TFF1`**, **`TFF3`**, **`SERPINA5`**, **`FKBP4`**, **`AREG`**, **`NRIP1`** context in Hallmark overlaps).

**Pathway ORA (top themes, from `enrichment.json`)**

Strong enrichment for **Hallmark Estrogen Response Early / Late**, **E2F / Myc proliferation programs**, **KEGG Cell cycle**, and **Reactome mitotic / G1–S** transitions — coherent with estrogen-pathway inhibition plus proliferation/cell-cycle shifts in a short drug treatment.

**Interpretation for robustness**

- With **only 2×2** biological replicates, DESeq2 estimates are **noisy**; treat **exact ranks and padj** as indicative, not publication-grade replication.
- The run is still a strong **integration test**: counts ingestion → DESeq2 → sensible biology + enrichment.

---

## Study 2 — GSE227102 (**blocked for DESeq2 parity**)

**Goal:** Compare OpenEnv DESeq2 to author-reported DE for **MCF7 vs MCF7-FAR** (and similarly **T47D**).

**What GEO provides**

FTP listing `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE227nnn/GSE227102/suppl/` includes **only**:

- `GSE227102_Differential_expression_analysis_MCF7.csv.gz`
- `GSE227102_Differential_expression_analysis_T47D.csv.gz`

These are **already differential-expression tables** (gene-level statistics), **not** raw or normalized integer counts.

**What we can run today (author-DE mode)**

Because GEO does not provide a count matrix for this series, OpenEnv can still run a meaningful benchmark by:

- loading the **author DE table** (`log2FoldChange`, `pvalue`, `padj`)
- selecting significant genes (padj threshold)
- running **Enrichr ORA** on the resulting gene list

This tests OpenEnv’s **ingestion + enrichment** stack and yields interpretable pathway-level results, but it is **not** a re-fit of DESeq2 on raw counts.

**Counts-like DESeq2 parity test (approximate)**

If the author DE table includes **per-sample numeric columns** (as GSE227102 does), OpenEnv can run an *approximate* DESeq2 parity test by rounding those values to integers and running DESeq2. This yields **quantitative agreement metrics** (correlations, top-N overlaps) against the author DE table, but should be described as **counts-like**, not FASTQ-derived counts.

**Author table sanity check** (local copy: `data/geo_eval/gse227102_author/MCF7_author.csv.gz`)

- **15,384** genes (semicolon-separated; European decimal commas normalized for parsing).
- **64.1%** of genes have **padj ≤ 0.05** under the author’s reported padj column — useful as an external reference distribution, but **not** directly runnable through PyDESeq2.

**What is needed to continue Study 2**

1. **Gene-level count matrix** for the same samples (author supplement elsewhere, SRA → alignment → quantification, or contacting authors).
2. Then rerun OpenEnv with the **same contrast** as the author table and compute:
   - Spearman correlation of **log2FC** on intersection
   - Overlap of top *N* genes
   - Sign concordance for significant sets

Until counts exist, **Study 2 remains documentation + author-table profiling only**.

---

## Study 3 — GSE235350 (pending)

GEO supplements contain **bigWig tracks** but **no gene count matrix** (the `RAW.tar` includes `*.bw`).

To run this study as a true-counts benchmark, use the optional `bench/` toolchain to download FASTQs from SRA and generate a counts matrix (authors used **hg38 + STAR + Subread/featureCounts** per the series matrix).

---

## Study 4 — GSE111151 (runner added)

Supplementary per-sample **`counts`** files exist. The NCBI FTP mirror currently exposes the legacy names `GSE111151_GSM1417177.txt.gz` … `…7184.gz` (eight samples), which map to current GSM accessions `GSM3024053` … `GSM3024060` (MCF-7 through ZR-75-1 Tam2); the BT-474 arm is not in that legacy set on the mirror.

**Runner:** `envs/pathway_analysis_env/bench/run_study4_gse111151_end_to_end.py`  
**Contrast:** tamoxifen-resistant vs parental (reference = parental).

---

## Study 5 — GSE216540 (runner added; TPM → pseudo-counts)

GEO supplement is TPM-only (`GSE216540_allData_tpm_human.txt.gz`). The benchmark script builds **rounded TPM × scale** pseudo-counts for an OpenEnv **pipeline test** (not statistically equivalent to raw-count DESeq2).

**Runner:** `envs/pathway_analysis_env/bench/run_study5_gse216540_end_to_end.py`  
**Default:** fast subset (28_CM: 4× DMSO vs 4× FULV). Use `--full` for all sample columns (slow).

---

## How to cite these benchmarks

Point reviewers to:

- This file for **methods and study status**
- Per-study `summary.json` + `enrichment.json` under `data/geo_eval/` for **reproducible numbers**
