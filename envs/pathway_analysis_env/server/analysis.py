# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Differential expression (PyDESeq2) and over-representation analysis (ORA).

Counts matrices use **samples × genes** layout for PyDESeq2 ≥ 0.5.
"""

from __future__ import annotations

import io
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    _PYDESQ2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy dep
    DeseqDataSet = None  # type: ignore[misc, assignment]
    DeseqStats = None  # type: ignore[misc, assignment]
    _PYDESQ2_AVAILABLE = False

try:
    import gseapy as gp

    _GSEAPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional extra dep
    gp = None  # type: ignore[assignment]
    _GSEAPY_AVAILABLE = False


def pydeseq2_available() -> bool:
    return _PYDESQ2_AVAILABLE


def gseapy_available() -> bool:
    return _GSEAPY_AVAILABLE


def default_analysis_options() -> Dict[str, Any]:
    """Defaults aligned with common RNA-seq practice (DESeq2 prefilter, directional ORA)."""
    return {
        "min_total_count": 10,
        "padj_alpha": 0.05,
        "ora_min_pathway_genes": 3,
        # Use "up" for treated-vs-control activation screens; "both" is the safe default.
        "de_query_direction": "both",
        "min_abs_log2fc": 0.0,
    }


def merge_analysis_options(case: Dict[str, Any]) -> Dict[str, Any]:
    out = default_analysis_options()
    raw = case.get("analysis_options")
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if k not in out or v is None:
            continue
        if k in ("min_total_count", "ora_min_pathway_genes"):
            out[k] = int(v)
        elif k in ("padj_alpha", "min_abs_log2fc"):
            out[k] = float(v)
        elif k == "de_query_direction":
            out[k] = str(v).lower().strip()
        else:
            out[k] = v
    return out


def filter_counts_by_minimum_total(
    counts_df: pd.DataFrame,
    min_total: int,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Remove genes with summed counts below ``min_total`` (DESeq2-style prefilter).

    Returns:
        (filtered_df, n_before, n_after)
    """
    if min_total <= 0:
        return counts_df, counts_df.shape[1], counts_df.shape[1]
    totals = counts_df.sum(axis=0)
    keep = totals >= min_total
    n_before = int(counts_df.shape[1])
    filtered = counts_df.loc[:, keep]
    n_after = int(filtered.shape[1])
    return filtered, n_before, n_after


def normalize_gene_ids(raw: Sequence[str]) -> List[str]:
    """
    Normalize gene identifiers for downstream gene set matching.

    GEO count tables often use a combined key like ``ENSG...__TP53``.
    We keep the symbol suffix when present.
    """

    out: List[str] = []
    for g in raw:
        s = str(g)
        if "__" in s:
            s = s.split("__", 1)[1]
        out.append(s)
    return out


def load_counts_csv_as_samples_by_genes(
    path: str | Path,
    *,
    sample_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load a counts table from CSV/CSV.GZ and return **samples × genes** DataFrame.

    Expected file layout:
      - rows: genes
      - columns: sample IDs
      - first column: gene identifier (may be unnamed)
    """

    p = Path(path)
    df = pd.read_csv(p, index_col=0)
    if df.empty:
        raise ValueError(f"Counts file is empty: {p}")

    df.index = normalize_gene_ids(df.index.tolist())
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Aggregate duplicate symbols (common when collapsing Ensembl->symbol).
    if df.index.has_duplicates:
        df = df.groupby(df.index).sum()

    # genes × samples -> samples × genes
    counts_df = df.T

    if sample_ids is not None:
        missing = [s for s in sample_ids if s not in counts_df.index]
        if missing:
            raise ValueError(
                f"Counts file missing sample columns/rows for: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
        counts_df = counts_df.loc[list(sample_ids)]

    return counts_df


def counts_dict_to_samples_by_genes(
    counts: Dict[str, Sequence[int]],
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    """Build a samples × genes count matrix from gene → per-sample counts."""
    sid_to_i = {sid: i for i, sid in enumerate(sample_ids)}
    rows = []
    for sid in sample_ids:
        j = sid_to_i[sid]
        rows.append([int(counts[g][j]) for g in counts])
    return pd.DataFrame(rows, index=list(sample_ids), columns=list(counts.keys()))


def build_sample_metadata(
    sample_ids: Sequence[str],
    condition_by_sample: Dict[str, str],
) -> pd.DataFrame:
    missing = [s for s in sample_ids if s not in condition_by_sample]
    if missing:
        raise ValueError(
            f"sample_metadata missing entries for sample_ids: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    conds = [condition_by_sample[s] for s in sample_ids]
    return pd.DataFrame({"condition": conds}, index=list(sample_ids))


def validate_counts_case(case: Dict[str, Any]) -> Optional[str]:
    """Return an error message if pipeline case JSON is inconsistent, else None."""
    counts = case.get("counts")
    sample_ids = case.get("sample_ids")
    if not isinstance(counts, dict) or not sample_ids:
        return None
    n = len(sample_ids)
    for gene, vals in counts.items():
        if len(vals) != n:
            return (
                f"Gene {gene!r} has {len(vals)} count values but "
                f"sample_ids has length {n}."
            )
    return None


def _finite_float_or_none(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def load_author_de_table_csv(
    path: str | Path,
    *,
    gene_column: str | None = None,
    log2fc_column: str = "log2FoldChange",
    pvalue_column: str = "pvalue",
    padj_column: str = "padj",
) -> List[Dict[str, Any]]:
    """
    Load a precomputed differential expression (DE) table (author-provided).

    Supports the common GEO supplement format used in GSE227102:
      - semicolon-delimited
      - decimal comma in numeric columns (e.g. ``0,12``) and scientific like ``1,47E-18``
      - gene symbol in a column like ``Gene,name`` and/or an Ensembl ``ID``

    Returns:
        DE rows in the same schema as ``run_deseq2_contrast`` output, sorted by ascending padj.
    """

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"DE table file not found: {p}")

    df = pd.read_csv(p, sep=";")
    if df.empty:
        raise ValueError(f"DE table is empty: {p}")

    # Pick gene column.
    if gene_column is None:
        for cand in ("Gene,name", "gene", "symbol", "Gene", "gene_name"):
            if cand in df.columns:
                gene_column = cand
                break
    if gene_column is None or gene_column not in df.columns:
        raise ValueError(
            "Could not infer gene column. Available columns: "
            + ", ".join(map(str, df.columns.tolist()))
        )

    # Normalize numeric strings (decimal commas).
    for c in (log2fc_column, pvalue_column, padj_column):
        if c not in df.columns:
            raise ValueError(f"Missing required column {c!r} in DE table: {p}")
        df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[gene_column] = df[gene_column].astype(str)

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        gene = str(r.get(gene_column, "")).strip()
        if not gene or gene.lower() in ("nan", "none"):
            continue
        padj = _safe_padj_value(r.get(padj_column))
        rows.append(
            {
                "gene": gene,
                "baseMean": None,  # unknown for author tables
                "log2FoldChange": _finite_float_or_none(r.get(log2fc_column)),
                "lfcSE": None,
                "pvalue": _finite_float_or_none(r.get(pvalue_column)),
                "padj": padj,
                "significant": False,  # filled by caller using chosen alpha
            }
        )

    rows.sort(
        key=lambda x: (
            _safe_padj_value(x.get("padj")),
            -abs(float(x.get("log2FoldChange") or 0.0)),
        )
    )
    return rows


def run_deseq2_contrast(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    alt_level: str,
    ref_level: str,
    *,
    padj_alpha: float = 0.05,
    min_replicates: int = 2,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Run PyDESeq2 Wald test for ``alt_level`` vs ``ref_level`` on column ``condition``.

    Returns:
        (de_rows, error_message). ``de_rows`` are sorted by ascending adjusted p-value.
    """
    if not _PYDESQ2_AVAILABLE:
        return [], "PyDESeq2 is not installed."

    levels = set(metadata_df["condition"].tolist())
    if ref_level not in levels or alt_level not in levels:
        return [], (
            f"Contrast invalid: need both reference {ref_level!r} and "
            f"alternate {alt_level!r} in sample metadata; got {sorted(levels)}."
        )

    try:
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata_df,
            design="~condition",
            refit_cooks=True,
            min_replicates=min_replicates,
            quiet=True,
        )
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=["condition", alt_level, ref_level])
            stat_res.summary()
        res = stat_res.results_df
    except Exception as exc:  # pragma: no cover - fitting failures
        return [], f"DESeq2 failed: {exc}"

    de_rows: List[Dict[str, Any]] = []
    for gene, row in res.iterrows():
        padj = _safe_padj_value(row["padj"])
        de_rows.append(
            {
                "gene": str(gene),
                "baseMean": _finite_float_or_none(row.get("baseMean")),
                "log2FoldChange": _finite_float_or_none(row.get("log2FoldChange")),
                "lfcSE": _finite_float_or_none(row.get("lfcSE")),
                "pvalue": _finite_float_or_none(row.get("pvalue")),
                "padj": padj,
                "significant": padj <= padj_alpha,
            }
        )
    de_rows.sort(key=lambda r: (r["padj"], -abs(r["log2FoldChange"] or 0.0)))
    return de_rows, None


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg FDR; returns q-values in original order (fallback)."""
    m = len(p_values)
    if m == 0:
        return []
    p_arr = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    adj_sorted = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(sorted_p[i] * m / (i + 1), running)
        adj_sorted[i] = running
    out = np.empty(m)
    out[order] = adj_sorted
    return np.clip(out, 0.0, 1.0).tolist()


def adjust_pvalues_bh(p_values: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg adjusted p-values using SciPy (preferred)."""
    m = len(p_values)
    if m == 0:
        return []
    p_arr = np.clip(
        np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0), 1e-300, 1.0
    )
    try:
        adj = false_discovery_control(p_arr, method="bh")
        return np.clip(adj, 0.0, 1.0).tolist()
    except Exception:
        return benjamini_hochberg(p_values)


def ora_fisher(
    de_genes: Sequence[str],
    pathway_genes: Dict[str, Sequence[str]],
    universe_genes: Sequence[str],
    *,
    min_pathway_genes: int = 3,
) -> List[Dict[str, Any]]:
    """
    Over-representation analysis (one-sided Fisher exact, greater overlap).

    ``universe_genes`` should be the **same gene set** used for DESeq2 (prefiltered).

    Pathways smaller than ``min_pathway_genes`` in the universe are skipped (reduces
    noise from tiny sets).
    """
    u: Set[str] = set(universe_genes)
    de: Set[str] = {g for g in de_genes if g in u}
    results: List[Dict[str, Any]] = []
    de_n = len(de)

    for pname, pgenes in pathway_genes.items():
        pset = {g for g in pgenes if g in u}
        if len(pset) < min_pathway_genes:
            continue
        overlap = sorted(de & pset)
        a = len(overlap)
        b = len(de - pset)
        c = len(pset - de)
        d = len(u) - a - b - c
        if d < 0:
            d = 0
        oddsr, p_raw = fisher_exact([[a, b], [c, d]], alternative="greater")
        p_f = float(p_raw) if math.isfinite(float(p_raw)) else 1.0
        results.append(
            {
                "pathway": pname,
                "p_value": p_f,
                "odds_ratio": float(oddsr) if np.isfinite(oddsr) else None,
                "overlap_genes": overlap,
                "overlap_count": a,
                "pathway_size": len(pset),
                "de_in_universe": de_n,
                "gene_ratio": f"{a}/{len(pset)}",
            }
        )

    qvals = adjust_pvalues_bh([r["p_value"] for r in results])
    for r, q in zip(results, qvals):
        r["q_value"] = q
    results.sort(key=lambda x: (x["p_value"], -x["overlap_count"]))
    return results


def enrichr_ora(
    query_genes: Sequence[str],
    *,
    libraries: Sequence[str],
    background: Optional[Sequence[str]] = None,
    top_k: int = 50,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Enrichr-based ORA using gseapy (requires network for most libraries).

    Returns:
        (rows, error_message)
    """

    if not _GSEAPY_AVAILABLE:
        return [], "gseapy is not installed."
    q = [str(g) for g in query_genes if g]
    if not q:
        return [], "Empty query gene list."
    libs = [str(x) for x in libraries if x]
    if not libs:
        return [], "No Enrichr libraries configured."

    rows: List[Dict[str, Any]] = []
    try:
        for lib in libs:
            enr = gp.enrichr(  # type: ignore[union-attr]
                gene_list=q,
                gene_sets=lib,
                background=list(background) if background is not None else None,
                outdir=None,
                no_plot=True,
            )
            res = getattr(enr, "results", None)
            if res is None or res.empty:
                continue
            for _, r in res.head(top_k).iterrows():
                genes = []
                raw = r.get("Genes")
                if isinstance(raw, str):
                    genes = [g.strip() for g in raw.replace(";", ",").split(",") if g.strip()]
                rows.append(
                    {
                        "pathway": f"{lib}: {r.get('Term')}",
                        "p_value": float(r.get("P-value", 1.0)),
                        "q_value": float(r.get("Adjusted P-value", 1.0)),
                        "odds_ratio": float(r.get("Odds Ratio"))
                        if pd.notna(r.get("Odds Ratio"))
                        else None,
                        "overlap_genes": genes,
                        "overlap_count": int(r.get("Overlap", "0/0").split("/")[0])
                        if isinstance(r.get("Overlap"), str)
                        else None,
                        "pathway_size": int(r.get("Overlap", "0/0").split("/")[1])
                        if isinstance(r.get("Overlap"), str)
                        else None,
                    }
                )
    except Exception as exc:  # pragma: no cover
        return [], f"Enrichr failed: {exc}"

    rows.sort(key=lambda x: (x.get("q_value", 1.0), x.get("p_value", 1.0)))
    return rows, None


def _safe_padj_value(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 1.0
    return x if math.isfinite(x) else 1.0


def pick_de_query_genes(
    de_rows: Sequence[Dict[str, Any]],
    *,
    padj_alpha: float = 0.05,
    max_genes: int = 200,
    direction: str = "both",
    min_abs_log2fc: float = 0.0,
) -> List[str]:
    """
    Genes for ORA query: significant by ``padj`` and optional **direction** (activation).

    ``direction``: ``\"up\"`` (alt > ref), ``\"down\"`` (alt < ref), or ``\"both\"``.
    """
    dir_norm = direction.lower().strip()
    if dir_norm not in ("up", "down", "both"):
        dir_norm = "both"

    def lfc_ok(r: Dict[str, Any]) -> bool:
        try:
            lfc = float(r.get("log2FoldChange", 0.0))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(lfc):
            return False
        if dir_norm == "both":
            return abs(lfc) >= min_abs_log2fc
        if dir_norm == "up":
            return lfc >= min_abs_log2fc
        return lfc <= -min_abs_log2fc

    sig: List[str] = []
    for r in de_rows:
        if _safe_padj_value(r.get("padj", 1.0)) > padj_alpha:
            continue
        if not lfc_ok(r):
            continue
        sig.append(r["gene"])

    if not sig:
        for r in de_rows[:max_genes]:
            if lfc_ok(r):
                sig.append(r["gene"])
    if not sig:
        sig = [r["gene"] for r in de_rows[:max_genes]]
    return sig[:max_genes]


def compare_pathways_detail(
    pathway_a: str,
    pathway_b: str,
    pathway_genes: Dict[str, Sequence[str]],
    de_genes: Sequence[str],
) -> Dict[str, Any]:
    """Exclusive vs shared DE support between two pathways."""
    pa = set(pathway_genes.get(pathway_a, []))
    pb = set(pathway_genes.get(pathway_b, []))
    de = set(de_genes)
    only_a = sorted((pa - pb) & de)
    only_b = sorted((pb - pa) & de)
    shared = sorted((pa & pb) & de)
    return {
        "pathway_a": pathway_a,
        "pathway_b": pathway_b,
        "exclusive_to_a": only_a,
        "exclusive_to_b": only_b,
        "shared_de_support": shared,
        "pathway_a_size": len(pa),
        "pathway_b_size": len(pb),
        "overlap_pathway_genes": sorted(pa & pb),
    }


def overlap_genes_across_top_pathways(
    ora_rows: Sequence[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """DE genes that appear in more than one of the top-k pathways by p-value."""
    top = [r for r in ora_rows[:top_k] if r.get("overlap_genes")]
    gene_to_paths: Dict[str, List[str]] = {}
    for row in top:
        p = row["pathway"]
        for g in row.get("overlap_genes", []):
            gene_to_paths.setdefault(g, []).append(p)
    multi = {g: paths for g, paths in gene_to_paths.items() if len(paths) > 1}
    return {
        "genes_supporting_multiple_top_pathways": sorted(multi.keys()),
        "gene_to_pathways": {g: multi[g] for g in sorted(multi)},
    }


def top_hits_statistically_close(
    ora_rows: Sequence[Dict[str, Any]],
    *,
    ratio_threshold: float = 10.0,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Flag when the top two enriched pathways have similar p-values (ratio bound)."""
    if len(ora_rows) < 2:
        return {
            "close_top_hits": False,
            "p_ratio": None,
            "note": "fewer than 2 pathways",
        }
    p1 = ora_rows[0]["p_value"]
    p2 = ora_rows[1]["p_value"]
    if p1 <= 0 or p2 <= 0:
        ratio = None
        close = False
    else:
        ratio = max(p1, p2) / min(p1, p2)
        close = ratio <= ratio_threshold
    return {
        "close_top_hits": close,
        "p_ratio": ratio,
        "p_top1": p1,
        "p_top2": p2,
    }
