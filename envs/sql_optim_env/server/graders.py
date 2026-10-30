"""
graders.py — Execution-Grounded Reward Function
=================================================
What makes this environment unique: reward is computed from REAL
DuckDB execution results, not just keyword heuristics.

Scoring breakdown (sums to 1.0):
  Real Execution Speedup    35%   — actual timing ratio from DuckDB
  Result Correctness        20%   — both queries return identical data?
  Issue Detection           25%   — keyword match vs ground truth
  Approval Correctness       8%   — correctly flags query as bad?
  Summary Quality            7%   — is the written analysis thorough?
  Severity Labels            5%   — are severity values present?

Optional ``GradeMask`` (keyword arg ``mask=``) zeroes components for ablations;
production calls omit ``mask`` (full scoring, including the 0.02 minimum when
appropriate).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    # In-repo imports (running from the OpenEnv repository)
    from ..models import SQLOptimAction as Action
    from .executor import get_executor
    from .scoring_models import Reward
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (running via uvicorn server.app:app)
    from models import SQLOptimAction as Action
    from server.executor import get_executor
    from server.scoring_models import Reward


@dataclass(frozen=True)
class GradeMask:
    """Toggle reward components (for ablations). All True = production grading."""

    execution_speedup: bool = True
    result_correctness: bool = True
    issue_detection: bool = True
    approval_correctness: bool = True
    summary_quality: bool = True
    severity_labels: bool = True


# ── Helpers ──────────────────────────────────────────────────────────────


def _kw_match(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _suggestions_text(action: Action) -> str:
    # Only the agent's analysis (summary + structured suggestions) counts toward
    # issue detection — deliberately NOT `optimized_query`. Otherwise echoing the
    # slow SQL (which contains the anti-pattern tokens) would farm keyword credit
    # without any real analysis.
    parts = [action.summary, action.estimated_improvement]
    for s in action.suggestions:
        parts += [
            str(s.get("issue_type", "")),
            str(s.get("description", "")),
            str(s.get("fix", "")),
            str(s.get("severity", "")),
        ]
    return " ".join(parts)


# ── Speedup → score mapping ───────────────────────────────────────────────


def _speedup_score(speedup: float, has_error: bool) -> float:
    """Map real speedup ratio to a score in [0.0, 0.35]."""
    if has_error:
        return 0.0
    if speedup >= 15.0:
        return 0.35
    if speedup >= 8.0:
        return 0.30
    if speedup >= 4.0:
        return 0.25
    if speedup >= 2.0:
        return 0.18
    if speedup >= 1.2:
        return 0.10
    if speedup >= 0.9:  # slightly slower — acceptable
        return 0.04
    return 0.0  # regression


# ── Main grader ───────────────────────────────────────────────────────────


def grade(
    task_data: Dict[str, Any],
    action: Action,
    *,
    mask: Optional[GradeMask] = None,
) -> Reward:
    original_query: str = task_data["sql_query"]
    optimized_query: str = (action.optimized_query or "").strip()
    ground_truth: List[Dict[str, Any]] = task_data["ground_truth_issues"]
    full_text = _suggestions_text(action)

    # ── 1. Real Execution (0.0–0.35) ─────────────────────────────────
    exec_info: Dict[str, Any] = {}
    speedup_sc = 0.0
    correctness_sc = 0.0
    exec_feedback: List[str] = []

    if optimized_query:
        try:
            ex = get_executor()
            exec_info = ex.compare(original_query, optimized_query)
            speedup = exec_info.get("speedup", 1.0)
            r_match = exec_info.get("results_match", False)
            opt_err = exec_info.get("optimized_error")

            # 1a. Speedup score — only credited when the rewrite is correct, so a
            # trivially fast but wrong query (e.g. `SELECT 1` against a heavy
            # original) earns nothing for speed.
            speedup_sc = _speedup_score(speedup, has_error=bool(opt_err) or not r_match)

            # 1b. Correctness score (0.0-0.20)
            if opt_err:
                correctness_sc = 0.0
            elif r_match:
                correctness_sc = 0.20
            elif exec_info.get("optimized_rows", 0) > 0:
                # Query ran but different results -- partial credit
                correctness_sc = 0.05

            # Feedback lines
            exec_feedback = [
                "\n[DuckDB Execution Results]",
                f"   Original  : {exec_info['original_ms']:.1f} ms "
                f"({exec_info['original_rows']} rows)",
                f"   Optimized : {exec_info['optimized_ms']:.1f} ms "
                f"({exec_info['optimized_rows']} rows)",
                f"   Speedup   : {speedup:.2f}x",
                f"   Correct?  : {'YES' if r_match else 'NO -- results differ'}",
                f"   Verdict   : {exec_info.get('verdict', '')}",
            ]
            if opt_err:
                exec_feedback.append(f"   SQL Error : {opt_err[:200]}")

        except Exception as exc:
            exec_feedback = [f"\n[WARN] Execution engine error: {exc}"]

    # ── 2. Issue Detection (0.0–0.25) ────────────────────────────────
    detected = 0
    detection_fb: List[str] = ["\n[Issue Detection]"]
    for gt in ground_truth:
        found = _kw_match(full_text, gt["keywords"])
        if found:
            detected += 1
            detection_fb.append(f"   [FOUND] {gt['type']} (line ~{gt['line']})")
        else:
            detection_fb.append(f"   [MISS ] {gt['type']} (line ~{gt['line']})")
    detection_sc = (detected / len(ground_truth)) * 0.25 if ground_truth else 0.0

    # ── 3. Approval Correctness (0.0–0.08) ───────────────────────────
    expected_approved = task_data.get("approved_expected", False)
    approval_sc = 0.08 if action.approved == expected_approved else 0.0

    # ── 4. Summary Quality (0.0–0.07) ────────────────────────────────
    summary_sc = 0.0
    slen = len(action.summary)
    if slen > 50:
        summary_sc = 0.03
    if slen > 120:
        summary_sc = 0.07

    # ── 5. Severity Labels (0.0–0.05) ────────────────────────────────
    sev_kw = ["critical", "high", "medium", "low"]
    has_sev = any(
        _kw_match(str(s.get("severity", "")), sev_kw) for s in action.suggestions
    )
    severity_sc = 0.05 if has_sev else 0.0

    breakdown = {
        "execution_speedup": round(speedup_sc, 4),
        "result_correctness": round(correctness_sc, 4),
        "issue_detection": round(detection_sc, 4),
        "approval_correctness": round(approval_sc, 4),
        "summary_quality": round(summary_sc, 4),
        "severity_labels": round(severity_sc, 4),
    }

    # ── Total (optional component mask for ablations) ─────────────────
    m = mask or GradeMask()
    contrib = {k: (v if getattr(m, k) else 0.0) for k, v in breakdown.items()}
    total = round(min(max(sum(contrib.values()), 0.0), 1.0), 4)
    if mask is None and total == 0.0 and action.suggestions:
        total = 0.02  # minimum signal for any submission (production only)

    feedback = "\n".join(
        exec_feedback
        + detection_fb
        + [
            f"\n   Suggestions submitted: {len(action.suggestions)} "
            f"(expected ~{len(ground_truth)})",
            f"   Approval: {'✅' if action.approved == expected_approved else '❌'} "
            f"(got {'approved' if action.approved else 'rejected'}, "
            f"expected {'approved' if expected_approved else 'rejected'})",
            f"\n🏆 Total score: {total:.4f}",
        ]
    )

    return Reward(
        score=total,
        breakdown=breakdown,
        feedback=feedback,
        execution=exec_info or None,
    )
