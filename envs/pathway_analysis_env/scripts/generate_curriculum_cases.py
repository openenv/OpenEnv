#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate synthetic case JSON files with increasing difficulty.

Levels:
  1 — clean MAPK vs PI3K separation
  2 — shared hub gene between top pathways (overlap-heavy)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = {
    "sample_ids": ["C1", "C2", "T1", "T2"],
    "sample_metadata": {
        "C1": "control",
        "C2": "control",
        "T1": "treated",
        "T2": "treated",
    },
    "conditions": ["control", "treated"],
    "default_contrast": {"reference": "control", "alternate": "treated"},
    "expert_budget": 1,
    "expert_penalty": 0.3,
    "strict_mode": False,
}


def case_clean_mapk(out: Path) -> None:
    data = {
        **BASE,
        "case_id": "curriculum_clean_mapk",
        "true_pathway": "MAPK signaling",
        "counts": {
            "DUSP6": [100, 110, 800, 820],
            "FOS": [90, 100, 750, 760],
            "JUN": [80, 85, 700, 710],
            "EGR1": [70, 75, 600, 620],
            "MAPK1": [60, 65, 500, 520],
            "AKT1": [200, 210, 205, 215],
            "PIK3CA": [150, 155, 152, 158],
            "GAPDH": [1000, 1050, 1020, 1010],
            "ACTB": [900, 920, 910, 905],
        },
        "pathway_genes": {
            "MAPK signaling": ["DUSP6", "FOS", "JUN", "EGR1", "MAPK1"],
            "PI3K-Akt signaling": ["AKT1", "PIK3CA"],
        },
        "expert_hint": "Contrast treated vs control; strongest ORA should match the pathway with coherent DE support.",
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def case_hub_overlap(out: Path) -> None:
    """DUSP6 appears in both MAPK and ERK; agent should use compare_pathways."""
    data = {
        **BASE,
        "case_id": "curriculum_hub_overlap",
        "true_pathway": "MAPK signaling",
        "counts": {
            "DUSP6": [100, 110, 750, 760],
            "FOS": [90, 100, 700, 710],
            "JUN": [80, 85, 650, 660],
            "EGR1": [70, 75, 600, 610],
            "MAPK1": [60, 65, 500, 510],
            "ELK1": [50, 55, 480, 490],
            "AKT1": [200, 210, 205, 215],
            "GAPDH": [1000, 1050, 1020, 1010],
            "ACTB": [900, 920, 910, 905],
        },
        "pathway_genes": {
            "MAPK signaling": ["DUSP6", "FOS", "JUN", "EGR1", "MAPK1"],
            "ERK cascade": ["DUSP6", "FOS", "ELK1", "MAPK1"],
            "PI3K-Akt signaling": ["AKT1"],
        },
        "expert_hint": "Hub genes can appear in multiple pathways; compare exclusive DE support.",
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Write curriculum case JSON files.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "curriculum",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_clean_mapk(args.out_dir / "curriculum_clean_mapk.json")
    case_hub_overlap(args.out_dir / "curriculum_hub_overlap.json")
    print(f"Wrote curriculum cases under {args.out_dir}")


if __name__ == "__main__":
    main()
