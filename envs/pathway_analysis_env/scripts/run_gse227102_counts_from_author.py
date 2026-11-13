#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GEO_EVAL_DIR = DATA_DIR / "geo_eval"


def _read_author_de(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", compression="gzip")
    if df.empty:
        raise ValueError(f"Empty author DE table: {path}")
    # normalize decimal commas in numeric columns we care about
    for c in ("log2FoldChange", "pvalue", "padj"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _infer_sample_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No sample columns found with prefix {prefix!r}. Columns={df.columns.tolist()}")
    return cols


def build_counts_matrix_from_author(
    df: pd.DataFrame,
    *,
    gene_id_col: str,
    sample_cols: List[str],
) -> pd.DataFrame:
    """
    Build a **genes × samples** matrix from author table numeric columns.

    Note: author values are not guaranteed to be raw integer counts. We coerce to numeric,
    fill missing with 0, clip negatives, and round to int so PyDESeq2 can run.
    """
    if gene_id_col not in df.columns:
        raise ValueError(f"Missing gene id column {gene_id_col!r}")
    for c in sample_cols:
        if c not in df.columns:
            raise ValueError(f"Missing sample column {c!r}")

    out = df[[gene_id_col] + sample_cols].copy()
    out = out.rename(columns={gene_id_col: "gene_id"})
    out["gene_id"] = out["gene_id"].astype(str)
    for c in sample_cols:
        out[c] = out[c].astype(str).str.replace(",", ".", regex=False)
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[c] = out[c].clip(lower=0.0)
        out[c] = out[c].round().astype(int)

    # Aggregate duplicate gene IDs (can occur with repeated symbols).
    out = out.groupby("gene_id", as_index=True).sum()
    return out


def write_counts_csv_gz(counts_gene_by_sample: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts_gene_by_sample.to_csv(out_path)


def run_openenv_case(case_file: str) -> Dict:
    """
    Run DE + enrichment using the same underlying analysis functions as the environment,
    but keep **full** DE rows (the env truncates `de_genes` for payload size).
    """
    from pathway_analysis_env.server.analysis import (
        build_sample_metadata,
        enrichr_ora,
        load_counts_csv_as_samples_by_genes,
        merge_analysis_options,
        pick_de_query_genes,
        run_deseq2_contrast,
    )
    from pathway_analysis_env.server.pathway_environment import load_case

    case = load_case(case_file)
    opts = merge_analysis_options(case)

    counts_df = load_counts_csv_as_samples_by_genes(
        DATA_DIR / str(case["counts_file"]),
        sample_ids=list(case["sample_ids"]),
    )
    meta_df = build_sample_metadata(list(case["sample_ids"]), dict(case["sample_metadata"]))
    ref = str(case["default_contrast"]["reference"])
    alt = str(case["default_contrast"]["alternate"])

    de_rows, err = run_deseq2_contrast(
        counts_df,
        meta_df,
        alt,
        ref,
        padj_alpha=float(opts["padj_alpha"]),
    )
    if err:
        raise RuntimeError(err)

    query = pick_de_query_genes(
        de_rows,
        padj_alpha=float(opts["padj_alpha"]),
        direction=str(opts["de_query_direction"]),
        min_abs_log2fc=float(opts["min_abs_log2fc"]),
    )
    if not query:
        query = [r["gene"] for r in de_rows[:50]]

    enr_libs = list(case.get("enrichr_libraries") or [])
    ora_rows, ora_err = enrichr_ora(
        query,
        libraries=enr_libs,
        background=list(counts_df.columns),
        top_k=100,
    )
    if ora_err:
        ora_rows = []

    return {
        "case_id": case.get("case_id"),
        "de_n_rows": len(de_rows),
        "de_n_sig": sum(1 for r in de_rows if r.get("significant")),
        "de_top_genes": [r["gene"] for r in de_rows[:50]],
        "de_all": de_rows,
        "ora_top_pathways": [r["pathway"] for r in ora_rows[:20]],
        "ora_rows": ora_rows,
        "ora_error": ora_err,
        "query_n_genes": len(query),
    }


def compare_to_author(
    *,
    author_df: pd.DataFrame,
    openenv_de_rows: List[Dict],
    author_gene_col: str = "Gene,name",
) -> Dict:
    """
    Compare OpenEnv DESeq2 results to author DE statistics (table-to-table).
    """
    # OpenEnv DESeq2 output uses the count-matrix column names as gene IDs.
    # In our reconstructed count-like matrices, the gene IDs are Ensembl IDs from the author table.
    a = author_df.copy()
    if author_gene_col not in a.columns:
        raise ValueError(f"Missing author gene col {author_gene_col!r}")
    a[author_gene_col] = a[author_gene_col].astype(str)
    a = a.dropna(subset=[author_gene_col]).drop_duplicates(subset=[author_gene_col])
    a = a.set_index(author_gene_col)

    oe = pd.DataFrame(openenv_de_rows)
    if oe.empty:
        return {"error": "openenv_de_empty"}
    oe = oe.dropna(subset=["gene"]).drop_duplicates(subset=["gene"]).set_index("gene")

    joined = oe.join(
        a[["log2FoldChange", "padj"]].rename(
            columns={"log2FoldChange": "author_log2fc", "padj": "author_padj"}
        ),
        how="inner",
    )
    if joined.empty:
        return {"error": "no_gene_overlap"}

    # Spearman correlations for effect size and significance
    out: Dict = {"n_overlap_genes": int(joined.shape[0])}
    out["spearman_log2fc"] = float(joined["log2FoldChange"].corr(joined["author_log2fc"], method="spearman"))
    out["spearman_padj"] = float(joined["padj"].corr(joined["author_padj"], method="spearman"))

    # Top-N overlap by smallest padj
    for n in (50, 200, 500):
        top_oe = set(joined.sort_values("padj").head(n).index)
        top_a = set(joined.sort_values("author_padj").head(n).index)
        inter = len(top_oe & top_a)
        out[f"top{n}_overlap"] = int(inter)
        out[f"top{n}_jaccard"] = float(inter / max(1, len(top_oe | top_a)))

    # Sign concordance for genes significant in either table
    sig = joined[(joined["padj"] <= 0.05) | (joined["author_padj"] <= 0.05)]
    if sig.empty:
        out["sign_concordance_sig_union"] = None
    else:
        s1 = (sig["log2FoldChange"] >= 0).astype(int)
        s2 = (sig["author_log2fc"] >= 0).astype(int)
        out["sign_concordance_sig_union"] = float((s1 == s2).mean())
        out["n_sig_union"] = int(sig.shape[0])
    return out


def main() -> None:
    # Inputs (author DE tables)
    mcf7_path = GEO_EVAL_DIR / "gse227102_author" / "MCF7_author.csv.gz"
    t47d_path = GEO_EVAL_DIR / "gse227102_author" / "T47D_author.csv.gz"

    # Build count-like matrices
    mcf7_df = _read_author_de(mcf7_path)
    t47d_df = _read_author_de(t47d_path)

    mcf7_samples = _infer_sample_cols(mcf7_df, "MCF7-")
    t47d_samples = _infer_sample_cols(t47d_df, "T47D-")

    # Prefer gene symbols so enrichment libraries match (Enrichr uses symbols).
    mcf7_counts = build_counts_matrix_from_author(
        mcf7_df, gene_id_col="Gene,name", sample_cols=mcf7_samples
    )
    t47d_counts = build_counts_matrix_from_author(
        t47d_df, gene_id_col="Gene,name", sample_cols=t47d_samples
    )

    # Write as counts files compatible with OpenEnv loader (genes in rows, samples in columns)
    mcf7_counts_path = GEO_EVAL_DIR / "gse227102_counts_like_MCF7.csv.gz"
    t47d_counts_path = GEO_EVAL_DIR / "gse227102_counts_like_T47D.csv.gz"
    write_counts_csv_gz(mcf7_counts, mcf7_counts_path)
    write_counts_csv_gz(t47d_counts, t47d_counts_path)

    # Build case files for DESeq2
    def make_case(
        case_id: str,
        counts_file_rel: str,
        sample_ids: List[str],
        sample_metadata: Dict[str, str],
        pubmed_id: str = "37258566",
    ) -> Dict:
        return {
            "case_id": case_id,
            "strict_mode": False,
            "experiment_metadata": {
                "source": "GEO",
                "accession": "GSE227102",
                "pubmed_id": pubmed_id,
                "summary": "Count-like matrix reconstructed from author supplement (rounded to integers) to enable DESeq2 run in OpenEnv. Use for approximate parity testing only.",
            },
            "counts_file": counts_file_rel,
            "sample_ids": sample_ids,
            "sample_metadata": sample_metadata,
            "conditions": sorted(set(sample_metadata.values())),
            "default_contrast": {"reference": "control", "alternate": "FAR"},
            "analysis_options": {"min_total_count": 10, "padj_alpha": 0.05, "de_query_direction": "both"},
            "enrichr_libraries": ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"],
            "true_pathway": "Unknown (counts-like benchmark)",
        }

    mcf7_case = make_case(
        "GSE227102_MCF7_counts_like_DESeq2",
        "geo_eval/gse227102_counts_like_MCF7.csv.gz",
        ["MCF7-WT-01", "MCF7-WT-02", "MCF7-WT-03", "MCF7-FAR-01", "MCF7-FAR-02", "MCF7-FAR-03"],
        {
            "MCF7-WT-01": "control",
            "MCF7-WT-02": "control",
            "MCF7-WT-03": "control",
            "MCF7-FAR-01": "FAR",
            "MCF7-FAR-02": "FAR",
            "MCF7-FAR-03": "FAR",
        },
    )
    t47d_case = make_case(
        "GSE227102_T47D_counts_like_DESeq2",
        "geo_eval/gse227102_counts_like_T47D.csv.gz",
        ["T47D-WT-01", "T47D-WT-02", "T47D-WT-03", "T47D-FAR-01", "T47D-FAR-02", "T47D-FAR-03"],
        {
            "T47D-WT-01": "control",
            "T47D-WT-02": "control",
            "T47D-WT-03": "control",
            "T47D-FAR-01": "FAR",
            "T47D-FAR-02": "FAR",
            "T47D-FAR-03": "FAR",
        },
    )

    (DATA_DIR / "gse227102_mcf7_counts_like_case.json").write_text(json.dumps(mcf7_case, indent=2), encoding="utf-8")
    (DATA_DIR / "gse227102_t47d_counts_like_case.json").write_text(json.dumps(t47d_case, indent=2), encoding="utf-8")

    # Run OpenEnv DESeq2 + enrichment on both
    mcf7_res = run_openenv_case("gse227102_mcf7_counts_like_case.json")
    t47d_res = run_openenv_case("gse227102_t47d_counts_like_case.json")

    mcf7_cmp = compare_to_author(author_df=mcf7_df, openenv_de_rows=list(mcf7_res["de_all"]))
    t47d_cmp = compare_to_author(author_df=t47d_df, openenv_de_rows=list(t47d_res["de_all"]))

    out_dir = GEO_EVAL_DIR / "gse227102_counts_like_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mcf7_openenv.json").write_text(
        json.dumps({k: v for k, v in mcf7_res.items() if k != "de_all"}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "t47d_openenv.json").write_text(
        json.dumps({k: v for k, v in t47d_res.items() if k != "de_all"}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "mcf7_de_all.json").write_text(json.dumps(mcf7_res["de_all"], indent=2), encoding="utf-8")
    (out_dir / "t47d_de_all.json").write_text(json.dumps(t47d_res["de_all"], indent=2), encoding="utf-8")
    (out_dir / "mcf7_enrichment.json").write_text(json.dumps(mcf7_res["ora_rows"], indent=2), encoding="utf-8")
    (out_dir / "t47d_enrichment.json").write_text(json.dumps(t47d_res["ora_rows"], indent=2), encoding="utf-8")
    (out_dir / "mcf7_compare_to_author.json").write_text(json.dumps(mcf7_cmp, indent=2), encoding="utf-8")
    (out_dir / "t47d_compare_to_author.json").write_text(json.dumps(t47d_cmp, indent=2), encoding="utf-8")

    print("[WROTE]", out_dir)
    print("MCF7 compare:", json.dumps(mcf7_cmp, indent=2))
    print("T47D compare:", json.dumps(t47d_cmp, indent=2))


if __name__ == "__main__":
    main()

