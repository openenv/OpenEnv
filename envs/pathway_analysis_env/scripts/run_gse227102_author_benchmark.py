#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.pathway_environment import PathwayEnvironment, load_case


def run_case(case_file: str) -> Dict[str, Any]:
    case = load_case(case_file)
    env = PathwayEnvironment(case_file=case_file)
    env.reset(strict=bool(case.get("strict_mode", False)))

    def step(kind: str, **kw: Any) -> Any:
        return env.step(PathwayAction(action_type=kind, **kw))

    step("inspect_dataset")
    o_de = step("run_differential_expression")
    if o_de.done:
        raise RuntimeError(f"DE step failed: {o_de.message}")
    o_enr = step("run_pathway_enrichment")
    if o_enr.done:
        raise RuntimeError(f"Enrichment step failed: {o_enr.message}")

    return {
        "case_id": case.get("case_id"),
        "de_n_rows": len(o_de.de_genes or []),
        "de_top_genes": o_de.top_genes or [],
        "ora_top_pathways": o_enr.top_pathways or [],
        "ora_rows": o_enr.pathway_enrichment or [],
        "metadata": {
            "de": o_de.metadata or {},
            "enrichment": o_enr.metadata or {},
        },
    }


def write_outputs(out_dir: Path, result: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Save the top pathways rows in a separate file for easy diffing.
    ora_rows: List[Dict[str, Any]] = list(result.get("ora_rows") or [])
    (out_dir / "enrichment.json").write_text(json.dumps(ora_rows, indent=2), encoding="utf-8")

    # DE rows are large; this benchmark uses author tables and the env already exposes the top genes.


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    out_root = base / "data" / "geo_eval"

    cases = [
        "gse227102_mcf7_case.json",
        "gse227102_t47d_case.json",
    ]

    for c in cases:
        r = run_case(c)
        slug = str(r.get("case_id") or c).replace(" ", "_")
        write_outputs(out_root / slug, r)
        print(f"[OK] {c} -> {out_root/slug}")


if __name__ == "__main__":
    main()

