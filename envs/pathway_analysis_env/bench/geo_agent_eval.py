#!/usr/bin/env python3
"""
Lightweight evaluation harness for GEO-style pathway_analysis_env cases.

Goal
  - Run the environment policy loop (inspect → DE → ORA)
  - Extract top enriched pathway terms
  - Score against a simple "published theme" rubric (keyword lists)

This intentionally avoids calling any external LLM API. You can plug in an agent
later by replacing `choose_hypothesis(...)`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.pathway_environment import PathwayEnvironment, load_case


@dataclass(frozen=True)
class StudySpec:
    case_file: str
    expected_keywords: List[str]
    top_k: int = 15


def _norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s).strip()


def extract_top_pathway_names(observation: Any, top_k: int) -> List[str]:
    # Preferred: PathwayObservation fields
    try:
        if getattr(observation, "top_pathways", None):
            return list(observation.top_pathways)[:top_k]
        rows = getattr(observation, "pathway_enrichment", None) or []
    except Exception:
        rows = []

    # Fallback: some callers may stash rows in metadata
    meta = getattr(observation, "metadata", None) or {}
    rows = rows or meta.get("pathway_enrichment") or meta.get("ora_rows") or meta.get("ora") or []

    names: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get("term") or r.get("pathway") or r.get("name") or r.get("Term")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
        if len(names) >= top_k:
            break
    return names


def choose_hypothesis(top_pathways: List[str], expected_keywords: List[str]) -> str:
    """
    Baseline "agent": pick the first enriched pathway that matches any keyword.
    If none match, fall back to the top enriched pathway (or a placeholder).
    """
    kws = [_norm(k) for k in expected_keywords]
    for p in top_pathways:
        pn = _norm(p)
        if any(k in pn for k in kws if k):
            return p
    return top_pathways[0] if top_pathways else "Unknown"


def score(top_pathways: List[str], expected_keywords: List[str]) -> Dict[str, Any]:
    joined = " | ".join(_norm(p) for p in top_pathways)
    hits = []
    for kw in expected_keywords:
        k = _norm(kw)
        if k and k in joined:
            hits.append(kw)
    return {
        "keyword_hits": hits,
        "keyword_hit_rate": (len(hits) / max(1, len(expected_keywords))),
        "any_hit": bool(hits),
    }


def run_one(spec: StudySpec, strict: bool) -> Dict[str, Any]:
    case = load_case(spec.case_file)
    ref = (case.get("default_contrast") or {}).get("reference")
    alt = (case.get("default_contrast") or {}).get("alternate")

    env = PathwayEnvironment(case_file=spec.case_file)
    env.reset(strict=strict)

    env.step(PathwayAction(action_type="inspect_dataset"))
    o_de = env.step(
        PathwayAction(
            action_type="run_differential_expression",
            condition_a=ref,
            condition_b=alt,
        )
    )
    if (o_de.metadata or {}).get("failure_code"):
        return {
            "case_id": case.get("case_id"),
            "case_file": spec.case_file,
            "strict_failure": bool(o_de.done),
            "stage_failure": "differential_expression",
            "failure_code": (o_de.metadata or {}).get("failure_code"),
            "message": o_de.message,
            "trace_path": o_de.trace_path,
        }
    if o_de.done:
        return {
            "case_id": case.get("case_id"),
            "case_file": spec.case_file,
            "strict_failure": True,
            "failure_code": (o_de.metadata or {}).get("failure_code"),
            "message": o_de.message,
        }

    o_ora = env.step(PathwayAction(action_type="run_pathway_enrichment"))
    if not getattr(o_ora, "top_pathways", None) and (getattr(o_ora, "metadata", None) or {}).get(
        "failure_code"
    ):
        return {
            "case_id": case.get("case_id"),
            "case_file": spec.case_file,
            "strict_failure": False,
            "stage_failure": "enrichment",
            "failure_code": (o_ora.metadata or {}).get("failure_code"),
            "message": o_ora.message,
            "trace_path": o_ora.trace_path,
        }
    top = extract_top_pathway_names(o_ora, top_k=spec.top_k)
    hyp = choose_hypothesis(top, spec.expected_keywords)
    scoring = score(top, spec.expected_keywords)

    # We submit the hypothesis so the trace has an explicit "answer" action.
    o_fin = env.step(PathwayAction(action_type="submit_answer", hypothesis=hyp))

    return {
        "case_id": case.get("case_id"),
        "case_file": spec.case_file,
        "strict_failure": False,
        "hypothesis": hyp,
        "top_pathways": top,
        "expected_keywords": spec.expected_keywords,
        "score": scoring,
        "trace_path": o_fin.trace_path,
    }


def load_specs(path: Path) -> List[StudySpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: List[StudySpec] = []
    for item in raw.get("studies", []):
        specs.append(
            StudySpec(
                case_file=item["case_file"],
                expected_keywords=list(item["expected_keywords"]),
                top_k=int(item.get("top_k", 15)),
            )
        )
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        type=Path,
        default=Path("data/geo_eval/geo_agent_eval_studies.json"),
        help="JSON spec with case files + expected keyword rubrics.",
    )
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--jsonl-out", type=Path, default=None)
    args = ap.parse_args()

    specs = load_specs(args.spec)
    results = [run_one(s, strict=args.strict) for s in specs]

    if args.jsonl_out:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl_out.write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    print(json.dumps({"n": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()

