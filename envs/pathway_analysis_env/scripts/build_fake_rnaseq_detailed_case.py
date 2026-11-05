#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Build a detailed synthetic RNA-seq-style case: many genes, 8 samples (4 vs 4),
pathway gene sets with overlap, and experiment metadata for auditing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

RNG = np.random.default_rng(42)

# Biologically plausible MAPK / stress-response genes (up in treated)
MAPK_CORE = [
    "DUSP6",
    "FOS",
    "JUN",
    "EGR1",
    "MAPK1",
    "MAPK3",
    "MAP2K1",
    "MAP2K2",
    "ELK1",
    "MYC",
    "CCND1",
    "CDKN1A",
    "GADD45A",
    "ATF3",
    "NR4A1",
    "HSPB1",
    "HSPA1A",
    "HSPA1B",
    "DNAJB1",
    "PPP1R15A",
]

# Decoy pathway (flat or weak)
WNT_SET = [
    "WNT3A",
    "WNT5A",
    "WNT11",
    "CTNNB1",
    "APC",
    "AXIN1",
    "TCF7",
    "LEF1",
    "DVL1",
    "DVL2",
    "FZD1",
    "LRP5",
]

PI3K_SET = [
    "AKT1",
    "AKT2",
    "PIK3CA",
    "PIK3CB",
    "PTEN",
    "MTOR",
    "RPS6KB1",
    "EIF4EBP1",
    "BAD",
    "FOXO1",
    "GSK3B",
    "TSC2",
]

# Housekeeping / structural background
HOUSEKEEPING = [
    "GAPDH",
    "ACTB",
    "TUBB",
    "TUBA1A",
    "RPLP0",
    "RPS18",
    "RPL13A",
    "EEF1A1",
    "EEF2",
    "HPRT1",
    "PPIA",
    "UBC",
    "B2M",
    "TBP",
    "GUSB",
    "YWHAZ",
    "PGK1",
    "TFRC",
    "POLR2A",
    "SDHA",
    "TPT1",
    "RPS6",
    "RPL7",
    "RPL11",
    "RPL23",
    "RPS3",
    "RPS14",
    "RPS27A",
    "RPL18",
    "RPL22",
    "RPS20",
    "RPS24",
    "RPS25",
    "RPS27",
    "RPL10",
    "RPL15",
    "RPL19",
    "RPL26",
    "RPL27",
    "RPL30",
    "RPLP1",
    "RPS2",
    "RPS4X",
    "RPS5",
    "RPS6KB2",
    "RPS8",
    "RPS9",
    "RPS10",
    "RPS11",
    "RPS12",
    "RPS13",
    "RPS15",
    "RPS16",
]

# Extra random genes to fill universe
NOISE_GENES = [f"LOC{i:05d}" for i in range(1, 121)]


def _sample_counts(base: float, n: int, scale: float = 0.08) -> List[int]:
    x = RNG.normal(loc=base, scale=base * scale, size=n)
    return [max(1, int(round(v))) for v in x]


def build_case() -> Dict[str, Any]:
    sample_ids = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"]
    sample_metadata = {s: "control" for s in sample_ids[:4]}
    sample_metadata.update({s: "treated" for s in sample_ids[4:]})

    counts: Dict[str, List[int]] = {}
    n_ctrl, n_trt = 4, 4

    # MAPK: strong up in treated
    for g in MAPK_CORE:
        ctrl = _sample_counts(120.0, n_ctrl, 0.12)
        trt = _sample_counts(2200.0, n_trt, 0.1)
        counts[g] = ctrl + trt

    # WNT: flat
    for g in WNT_SET:
        counts[g] = _sample_counts(400.0, 8, 0.15)

    # PI3K: mild bump treated (decoy)
    for g in PI3K_SET:
        ctrl = _sample_counts(350.0, n_ctrl, 0.12)
        trt = _sample_counts(520.0, n_trt, 0.12)
        counts[g] = ctrl + trt

    # Housekeeping
    for g in HOUSEKEEPING:
        counts[g] = _sample_counts(900.0, 8, 0.1)

    # Noise genes
    for g in NOISE_GENES:
        counts[g] = _sample_counts(180.0, 8, 0.2)

    pathway_genes: Dict[str, List[str]] = {
        "MAPK signaling": list(MAPK_CORE),
        "ERK cascade": [
            "DUSP6",
            "FOS",
            "JUN",
            "ELK1",
            "MAPK1",
            "MAPK3",
            "MAP2K1",
            "MAP2K2",
        ],
        "WNT signaling": WNT_SET,
        "PI3K-Akt signaling": PI3K_SET,
        "Stress response": ["HSPB1", "HSPA1A", "HSPA1B", "DNAJB1", "PPP1R15A"],
    }

    # Deduplicate overlap lists for JSON readability
    pathway_genes["MAPK signaling"] = list(
        dict.fromkeys(pathway_genes["MAPK signaling"])
    )

    gene_annotations = {
        g: f"ENSG{100000000 + i:011d}" for i, g in enumerate(list(counts.keys())[:200])
    }

    experiment_metadata = {
        "organism": "Homo sapiens",
        "assay": "poly-A RNA-seq",
        "library_layout": "paired-end",
        "read_length_bp": 101,
        "strand_specificity": "reverse",
        "reference_genome": "GRCh38",
        "quantification": "gene-level counts (synthetic)",
        "batch_note": "fake single-batch dataset for OpenEnv benchmarking",
        "n_samples": len(sample_ids),
        "n_genes_quantified": len(counts),
    }

    return {
        "case_id": "fake_rnaseq_detailed",
        "conditions": ["control", "treated"],
        "true_pathway": "MAPK signaling",
        "sample_ids": sample_ids,
        "sample_metadata": sample_metadata,
        "default_contrast": {"reference": "control", "alternate": "treated"},
        "analysis_options": {
            "min_total_count": 10,
            "padj_alpha": 0.05,
            "de_query_direction": "up",
            "min_abs_log2fc": 0.25,
            "ora_min_pathway_genes": 5,
        },
        "counts": counts,
        "pathway_genes": pathway_genes,
        "experiment_metadata": experiment_metadata,
        "gene_annotations_ensembl_style": gene_annotations,
        "expert_budget": 2,
        "expert_penalty": 0.25,
        "expert_hint": "Strongest coordinated DE should align with MAPK/ERK-associated genes; check overlap across top pathways.",
        "strict_mode": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "fake_rnaseq_detailed.json",
    )
    args = p.parse_args()
    case = build_case()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(case, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.out} ({len(case['counts'])} genes, {len(case['sample_ids'])} samples)"
    )


if __name__ == "__main__":
    main()
