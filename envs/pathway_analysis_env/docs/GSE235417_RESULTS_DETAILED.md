# GSE235417 — detailed DGE + pathway analysis breakdown (OpenEnv run)

Source study: [GSE235417](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235417) (HNSCC PDX UCLHN04; baseline vs acquired cetuximab-resistant; 3 biological replicates per condition).

This analysis was executed **through the OpenEnv `pathway_analysis_env` actions** (DESeq2 via PyDESeq2, then ORA enrichment).

## 1) Experimental design (what was compared)

- **Conditions**: `baseline` (n=3) vs `resistant` (n=3)
- **Contrast**: `resistant` vs `baseline`
  - **Interpretation**: log2 fold-change \(>0\) means **higher in resistant**; \(<0\) means **lower in resistant**

## 2) Input + preprocessing

- **Counts source**: `envs/pathway_analysis_env/data/geo/GSE235417_counts.csv.gz`
- **Genes in matrix (after ID normalization & deduping)**: **28,623**
- **Prefilter**: removed genes with total counts across all samples < 10
  - **Genes after prefilter**: **17,976**

## 3) Differential gene expression (DGE / DESeq2)

Environment output files:
- **Top 200 DE rows** (gene, baseMean, log2FC, lfcSE, pvalue, padj): `envs/pathway_analysis_env/outputs/gse235417/de_top200.json`

### Most increased in resistant (top by log2FC among exported hits)

Each item is: gene — log2FC — q-value (FDR)

- PCK1 — 8.038 — 2.75e-28
- POF1B — 6.487 — 1.11e-09
- OTOP3 — 6.019 — 6.50e-08
- GPD1 — 5.875 — 1.23e-34
- ARHGEF26 — 5.832 — 6.32e-08
- KY — 5.667 — 4.05e-11
- TMEM255A — 5.443 — 7.82e-08
- PAX6 — 5.194 — 2.72e-20
- SECTM1 — 5.146 — 4.09e-08
- FOLR1 — 4.841 — 8.84e-09

### Most decreased in resistant (top by negative log2FC among exported hits)

- COL22A1 — -10.029 — 1.01e-20
- CCNA1 — -8.272 — 3.54e-26
- FABP4 — -7.898 — 7.76e-08
- TFPI2 — -6.998 — 7.60e-25
- GUCY1A2 — -6.598 — 2.46e-24
- TAGLN3 — -5.497 — 1.47e-08
- TREM1 — -5.445 — 2.08e-12
- C11orf87 — -5.415 — 2.24e-16
- HAS2 — -5.361 — 1.30e-22
- VNN1 — -5.186 — 7.46e-11

### Top-ranked DE genes (smallest q-values)

From the run summary (`summary.json`) top list:

DAPK1, TLE4, MCTP1, CXCL8, GPD1, SOX2, FKBP5, PCK1, SSC5D, CCNA1, ODC1, SYT7, TFPI2, GUCY1A2, SPTBN5, AKR1B10, SEMA3A, HAS2, IGFN1, GSAP

## 4) Pathway enrichment (ORA)

Environment output file:
- **Top 50 enriched terms** (p-value, q-value, odds ratio, overlap genes): `envs/pathway_analysis_env/outputs/gse235417/enrichment_top50.json`

Configured libraries (Enrichr via `gseapy`):
- `MSigDB_Hallmark_2020`
- `KEGG_2021_Human`
- `Reactome_2022`

### Top enriched pathways (lowest q-values)

Each item is: pathway — q-value — overlap gene count (within this run’s query set)

1. MSigDB_Hallmark_2020: Estrogen Response Late — 5.33e-06 — 13
2. MSigDB_Hallmark_2020: Estrogen Response Early — 5.33e-06 — 13
3. MSigDB_Hallmark_2020: Epithelial Mesenchymal Transition — 3.02e-05 — 12
4. Reactome_2022: Interleukin-10 Signaling R-HSA-6783783 — 4.59e-05 — 7
5. MSigDB_Hallmark_2020: Interferon Gamma Response — 6.89e-05 — 11
6. MSigDB_Hallmark_2020: Inflammatory Response — 2.26e-04 — 10
7. Reactome_2022: Cytokine Signaling In Immune System R-HSA-1280215 — 3.07e-04 — 22
8. MSigDB_Hallmark_2020: Allograft Rejection — 3.44e-04 — 9
9. MSigDB_Hallmark_2020: TNF-alpha Signaling via NF-kB — 3.57e-04 — 10
10. MSigDB_Hallmark_2020: Interferon Alpha Response — 4.12e-04 — 7

### “Cross-pathway support” genes (appear in multiple top pathways)

Among the **top 10 pathways**, these genes show up repeatedly (gene — number of top-10 pathways containing it):

- CSF1 (6)
- IL1B (5)
- CXCL1 (4)
- CXCL8 (4)
- IL15RA (4)
- IL1A (4)
- INHBA (4)
- TAP1 (4)
- HLA-A (3)
- HLA-DQA1 (3)
- ISG15 (3)
- LCK (3)
- SERPINE1 (3)
- USP18 (3)

This is useful when communicating a story: instead of “one pathway name,” you can show the **recurring evidence genes** that drive several related immune/inflammatory pathway terms.

## 5) Reproducibility artifacts

- **Run summary**: `envs/pathway_analysis_env/outputs/gse235417/summary.json`
- **DE table export**: `envs/pathway_analysis_env/outputs/gse235417/de_top200.json`
- **Enrichment export**: `envs/pathway_analysis_env/outputs/gse235417/enrichment_top50.json`
- **HTML episode trace**: path is stored in `summary.json` (`trace_path`)

## 6) Important notes (interpretation)

- The exported DE table is currently the **top 200 rows** from the run (for UI responsiveness). If you want the *entire* DESeq2 results exported, we can add an option to write all rows to disk.
- ORA here uses Enrichr libraries (fast, convenient). For publication-grade work, you may want to add **rank-based GSEA** and use pinned gene sets (GMT) for fully offline reproducibility.

