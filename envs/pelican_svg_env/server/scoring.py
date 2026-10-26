# SPDX-License-Identifier: BSD-3-Clause

"""Run every scoring layer over one submission and assemble the verdict.

Layers run cheapest first, and the judge only ever sees submissions that already
cleared the free ones. The submission is parsed and rasterised exactly once here
and the analysis is carried on the observation, so the rubric layer reads those
numbers instead of paying for the same render again.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from .gate import GateResult, run_gate
from .geometry import extract_shapes
from .structure import analyse_structure, StructureReport
from .svg_source import extract_svg, parse_svg, SvgParseError, TruncatedSvgError
from .tasks import Task
from .vision_judge import JudgeReport, VisionJudge

STRUCTURE_WEIGHT = 0.35
SEMANTIC_WEIGHT = 0.65

# What survives of the semantic score when the judge says nothing is riding the
# vehicle. See `Evaluation.semantic_score` for why it is a penalty, not a zero.
NOT_RIDING_PENALTY = 0.25


def component_weights(judge_enabled: bool) -> tuple[float, float]:
    """Return the `(structure, semantic)` weights for a run.

    Keyed on whether a judge is *configured*, deliberately not on whether one
    *answered*: renormalising after a failed judge call would score a submission
    higher precisely when nobody could look at it, making "break the judge" a
    winning move. A failed call leaves the semantic component at zero and sets
    `judged` false so a harness can drop the sample instead of believing it.

    Both [`Evaluation.reward`] and
    [`~envs.pelican_svg_env.server.rubric.build_rubric`] read their weights from
    here, so the two cannot disagree.

    Args:
        judge_enabled (`bool`):
            Whether a vision judge is configured for the run.

    Returns:
        `tuple[float, float]`: Weights summing to 1.0.
    """
    if not judge_enabled:
        return 1.0, 0.0
    return STRUCTURE_WEIGHT, SEMANTIC_WEIGHT


@dataclass(frozen=True)
class Evaluation:
    """Everything known about one submission.

    Attributes:
        task ([`Task`]):
            The request the submission was answering.
        svg (`str`):
            The extracted SVG source, empty when extraction failed.
        gate ([`GateResult`]):
            Verdict of the deterministic admission check.
        structure ([`StructureReport`] or `None`):
            Structural analysis, `None` when the gate rejected the submission.
        judge ([`JudgeReport`] or `None`):
            Semantic analysis, `None` when the gate rejected the submission or
            no judge was configured.
        judge_enabled (`bool`):
            Whether a judge was configured for this run, regardless of whether
            it managed to answer. Drives the reward weighting.
    """

    task: Task
    svg: str
    gate: GateResult
    structure: StructureReport | None = None
    judge: JudgeReport | None = None
    judge_enabled: bool = False

    @property
    def gate_passed(self) -> bool:
        """`bool`: Whether the submission cleared the deterministic checks."""
        return self.gate.passed

    @property
    def structure_score(self) -> float:
        """`float`: Fraction of structural checks passed, in [0, 1]."""
        return self.structure.score if self.structure else 0.0

    @property
    def semantic_score(self) -> float:
        """`float`: Judge score, heavily penalised when nothing is shown riding.

        A bicycle with nothing on it does not answer "draw an animal riding a
        bicycle", so failing the posture check costs most of the score. Not all
        of it: posture is a spatial relation and the least reliable item on the
        checklist, so a hard zero would give the judge's worst call the loudest
        vote.
        """
        if self.judge is None or self.judge.error is not None:
            return 0.0
        if not self.judge.checklist.get("riding_posture", False):
            return NOT_RIDING_PENALTY * self.judge.score
        return self.judge.score

    @property
    def judged(self) -> bool:
        """`bool`: Whether a judge actually produced a verdict."""
        return self.judge is not None and self.judge.error is None

    @property
    def reward(self) -> float:
        """`float`: Overall reward in [0, 1].

        Zero if the gate rejected the submission. Otherwise the weighted sum
        defined by [`component_weights`], which is also what the rubric tree
        computes.
        """
        if not self.gate_passed:
            return 0.0
        structure_weight, semantic_weight = component_weights(self.judge_enabled)
        return (
            structure_weight * self.structure_score
            + semantic_weight * self.semantic_score
        )

    @property
    def feedback(self) -> str:
        """`str`: A short human-readable explanation of the score."""
        if not self.gate_passed:
            reasons = "; ".join(v.detail for v in self.gate.violations)
            return f"Rejected before scoring: {reasons}"
        parts = [f"structure {self.structure_score:.2f}"]
        if self.structure is not None:
            missed = [c.name for c in self.structure.checks if not c.passed]
            if missed:
                parts.append("missing " + ", ".join(missed))
        if self.judged and self.judge is not None:
            parts.append(f"judge saw: {self.judge.caption}")
        elif self.judge is not None and self.judge.error:
            parts.append(f"judge unavailable ({self.judge.error})")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable breakdown of the whole evaluation."""
        return {
            "task": self.task.to_dict(),
            "reward": round(self.reward, 4),
            "gate": self.gate.to_dict(),
            "structure": self.structure.to_dict() if self.structure else None,
            "judge": self.judge.to_dict() if self.judge else None,
            "structure_score": round(self.structure_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "judged": self.judged,
            "feedback": self.feedback,
        }


def evaluate_deterministic(response: str, task: Task) -> Evaluation:
    """Score a submission using only the free layers.

    Runs extraction, the gate and the structural analysis. Useful on its own
    for tests, for offline runs, and as the first half of
    [`evaluate_submission`].

    Args:
        response (`str`):
            Raw model output, which may wrap the SVG in prose or a code fence.
        task ([`Task`]):
            The request being answered.

    Returns:
        [`Evaluation`]: With `judge` left as `None`.

    Examples:

    ```python
    evaluation = evaluate_deterministic(model_reply, make_task("pelican", "bicycle"))
    print(evaluation.gate_passed, evaluation.structure_score)
    ```
    """
    from .svg_source import SourceViolation

    try:
        svg = extract_svg(response)
    except TruncatedSvgError as exc:
        return Evaluation(
            task=task,
            svg="",
            gate=GateResult(
                passed=False,
                violations=[SourceViolation("truncated_svg", str(exc))],
            ),
        )
    except SvgParseError as exc:
        return Evaluation(
            task=task,
            svg="",
            gate=GateResult(
                passed=False,
                violations=[SourceViolation("no_svg_in_response", str(exc))],
            ),
        )

    gate = run_gate(svg, forbidden_terms=task.forbidden_terms)
    if not gate.passed:
        return Evaluation(task=task, svg=svg, gate=gate)

    structure = analyse_structure(
        extract_shapes(parse_svg(svg)), expected_wheels=task.vehicle.wheels
    )
    return Evaluation(task=task, svg=svg, gate=gate, structure=structure)


async def evaluate_submission(
    response: str, task: Task, judge: VisionJudge | None = None
) -> Evaluation:
    """Score a submission through every configured layer.

    Args:
        response (`str`):
            Raw model output.
        task ([`Task`]):
            The request being answered.
        judge ([`VisionJudge`], *optional*):
            The semantic layer. When omitted the reward comes from structure
            alone.

    Returns:
        [`Evaluation`]: The complete verdict.

    Examples:

    ```python
    evaluation = await evaluate_submission(reply, task, VisionJudge(HFVisionClient()))
    print(evaluation.reward, evaluation.feedback)
    ```
    """
    # Parsing and rasterising is CPU-bound, so it runs in a worker thread to
    # keep the server's event loop free for other sessions.
    evaluation = await asyncio.to_thread(evaluate_deterministic, response, task)
    enabled = judge is not None
    if judge is None or not evaluation.gate_passed or evaluation.gate.png is None:
        return replace(evaluation, judge_enabled=enabled)

    report: JudgeReport = await judge.evaluate(evaluation.gate.png, task)
    return replace(evaluation, judge=report, judge_enabled=enabled)
