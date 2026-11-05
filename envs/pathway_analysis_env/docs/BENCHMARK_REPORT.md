# Benchmark report (OpenEnv `pathway_analysis_env`)

This report is meant to be readable by a biologist with minimal coding background. It summarizes what we ran, what outputs exist, and how well OpenEnv agrees with external references.

## Modes (important)

OpenEnv can evaluate GEO studies in three different ways depending on what GEO provides:

- **True-counts DESeq2 (gold standard)**: integer read counts per gene per sample → DESeq2 → enrichment.  
  **Best for accuracy comparisons**, but requires a count matrix.

- **Author-DE mode (when GEO lacks counts)**: load author-provided DE table (log2FC/pvalue/padj) → enrichment.  
  **Tests ingestion + enrichment** and yields biology, but **does not** re-fit DESeq2.

- **Counts-like DESeq2 (approximate parity test)**: reconstruct an integer “count-like” matrix by rounding the author per-sample columns → DESeq2 → compare DE table to the author DE table.  
  This can give a **quantitative agreement score**, but it is **not equivalent** to FASTQ → featureCounts.

---

## Study 1 — GSE128911 (true-counts DESeq2) ✅

**Question:** Fulvestrant vs DMSO in ER+ breast cancer line MDA-MB-134-VI (2 replicates per arm).

**Inputs (GEO provides integer counts):**

- `envs/pathway_analysis_env/data/geo_eval/GSE128911_RNA-seq_counts_Dataset2.tsv.gz`

**OpenEnv outputs (reproducible artifacts):**

- `envs/pathway_analysis_env/data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/summary.json`
- `envs/pathway_analysis_env/data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/de_top50.csv`
- `envs/pathway_analysis_env/data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/de_all.json`
- `envs/pathway_analysis_env/data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/enrichment.json`

**Headline results:**

- **310 genes** significant at **padj ≤ 0.05**
- **`ESR1`** is strongly **down** with fulvestrant (expected for a SERD)
- Pathway enrichment highlights **Estrogen Response Early/Late**, plus proliferation/cell-cycle programs

**Robustness note:** 2×2 replicates is small; treat exact gene ranks/padj as **indicative**, while pathway-level themes are more stable.

---

## Study 2 — GSE227102 (resistance models) ✅✅

**Question:** transcriptional changes in resistant derivatives:

- **MCF7** vs **MCF7-FAR** (fulvestrant+abemaciclib resistant)
- **T47D** vs **T47D-FAR**

GEO provides **author DE tables** (and sample-level columns), but **no raw count matrix**.

### Study 2A — Author-DE mode (enrichment on author DE) ✅

**Outputs:**

- `envs/pathway_analysis_env/data/geo_eval/GSE227102_MCF7_vs_MCF7-FAR_author_DE/summary.json`
- `envs/pathway_analysis_env/data/geo_eval/GSE227102_MCF7_vs_MCF7-FAR_author_DE/enrichment.json`
- `envs/pathway_analysis_env/data/geo_eval/GSE227102_T47D_vs_T47D-FAR_author_DE/summary.json`
- `envs/pathway_analysis_env/data/geo_eval/GSE227102_T47D_vs_T47D-FAR_author_DE/enrichment.json`

**Biology summary:**

- MCF7 arm: strong enrichment of **cell-cycle / DNA replication** programs and ER program changes.
- T47D arm: strong **interferon response / EMT** themes (plus additional programs).

### Study 2B — Counts-like DESeq2 parity test (quantitative agreement) ✅

We reconstructed a “count-like” matrix by rounding the author per-sample columns to integers, ran DESeq2, then compared the resulting DE table to the author DE statistics.

**Outputs:**

- `envs/pathway_analysis_env/data/geo_eval/gse227102_counts_like_runs/mcf7_compare_to_author.json`
- `envs/pathway_analysis_env/data/geo_eval/gse227102_counts_like_runs/t47d_compare_to_author.json`
- plus the full OpenEnv outputs:
  - `mcf7_de_all.json`, `mcf7_enrichment.json`, `mcf7_openenv.json`
  - `t47d_de_all.json`, `t47d_enrichment.json`, `t47d_openenv.json`

**Agreement metrics (OpenEnv vs author DE table):**

- **MCF7**
  - Spearman(log2FC): ~**0.9996**
  - Spearman(padj): ~**0.9965**
  - Top-50 overlap: **38/50**
  - Top-200 overlap: **179/200**
  - Top-500 overlap: **450/500**

- **T47D**
  - Spearman(log2FC): ~**0.9970**
  - Spearman(padj): ~**0.9506**
  - Top-50 overlap: **32/50**
  - Top-200 overlap: **140/200**
  - Top-500 overlap: **362/500**

**How to interpret:** these numbers show OpenEnv’s DE + ranking behavior is highly consistent with the author table *under this reconstruction*, but the reconstruction is **not a substitute** for raw-count processing from FASTQs.

---

## What we fixed to make OpenEnv more robust

- **Matplotlib cache permissions:** the pathway env now sets `MPLCONFIGDIR` to a writable cache under `envs/pathway_analysis_env/outputs/.mplcache`, avoiding runtime warnings/failures in restricted environments.

