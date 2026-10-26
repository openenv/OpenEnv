# SPDX-License-Identifier: BSD-3-Clause

"""The deterministic admission check for a submission.

Nothing here needs a model, and nothing here is a matter of taste. The gate
answers a narrow question: is this a well-formed, honestly-constructed SVG that
puts a drawing on the canvas? Submissions that fail score zero and are never
sent to the expensive scoring layers, which is what keeps a benchmark run
affordable when a model is producing garbage.

Thresholds were calibrated against the fixture corpus in ``fixtures/``. The
sparsest genuine drawing there covers 4.2% of the canvas in ink and the densest
degenerate one covers 1.3%, so the floor sits at 1.5% with margin on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .render import image_stats, ImageStats, render_png, RenderError
from .svg_source import inspect_source, SourceReport, SourceViolation, SvgParseError

# Ink coverage floor. Real model output measured between 3.9% and 23.7%
# coverage and the densest degenerate fixture reached 1.3%, so this sits below
# both with room to spare.
#
# There is no ceiling. A canvas flooded by one shape makes that shape the modal
# colour, which drives ink coverage toward zero, so `blank_canvas` already
# catches it. An earlier edge-density check is gone for a worse reason: measured
# against real submissions it failed in both directions, rejecting genuine
# filled-shape drawings that scored as low as 0.0068 while the
# write-the-answer-as-text cheat scored high, because glyphs are nothing but
# edges.
MIN_INK_FRACTION = 0.015


@dataclass(frozen=True)
class GateResult:
    """Outcome of the deterministic admission check.

    Attributes:
        passed (`bool`):
            Whether the submission cleared every check.
        violations (`list[SourceViolation]`):
            Reasons for rejection. Empty when `passed` is `True`.
        source (`SourceReport` or `None`):
            Document inventory, `None` if the source would not parse.
        stats (`ImageStats` or `None`):
            Raster measurements, `None` if the source would not render.
        png (`bytes` or `None`):
            The rendered raster, reused by the vision layer so the submission
            is rasterised exactly once per evaluation.
    """

    passed: bool
    violations: list[SourceViolation] = field(default_factory=list)
    source: SourceReport | None = None
    stats: ImageStats | None = None
    png: bytes | None = None

    @property
    def codes(self) -> list[str]:
        """`list[str]`: Violation codes, in the order they were raised."""
        return [v.code for v in self.violations]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, excluding the raster bytes."""
        return {
            "passed": self.passed,
            "violations": [
                {"code": v.code, "detail": v.detail} for v in self.violations
            ],
            "element_counts": self.source.element_counts if self.source else {},
            "drawable_count": self.source.drawable_count if self.source else 0,
            "image_stats": self.stats.to_dict() if self.stats else None,
        }


def run_gate(
    svg_source: str,
    forbidden_terms: Sequence[str] = (),
    render_size: int = 512,
) -> GateResult:
    """Run every deterministic check against a submission.

    Args:
        svg_source (`str`):
            SVG document, already extracted from any surrounding prose by
            [`~envs.pelican_svg_env.server.svg_source.extract_svg`].
        forbidden_terms (`Sequence[str]`, *optional*):
            Nouns that must not appear as rendered text in the drawing.
        render_size (`int`, *optional*, defaults to `512`):
            Raster size used for the ink measurements.

    Returns:
        [`GateResult`]: Pass or fail, with the evidence behind the decision.

    Examples:

    ```python
    result = run_gate(svg, forbidden_terms=["pelican", "bicycle"])
    if not result.passed:
        print(result.codes)
    ```
    """
    try:
        report = inspect_source(svg_source, forbidden_terms)
    except SvgParseError as exc:
        return GateResult(
            passed=False,
            violations=[SourceViolation("unparseable", str(exc))],
        )

    violations = list(report.violations)

    try:
        png = render_png(svg_source, size=render_size)
    except RenderError as exc:
        violations.append(SourceViolation("render_failed", str(exc)))
        return GateResult(passed=False, violations=violations, source=report)

    stats = image_stats(png)

    if stats.ink_fraction < MIN_INK_FRACTION:
        # Distinguish "drew nothing" from "drew something nobody can see", so
        # the model gets an actionable message rather than a blanket rejection.
        if report.drawable_count >= 3:
            violations.append(
                SourceViolation(
                    "content_off_canvas",
                    f"{report.drawable_count} drawable elements but only "
                    f"{stats.ink_fraction:.4f} ink coverage; the drawing falls "
                    "outside the viewBox",
                )
            )
        else:
            violations.append(
                SourceViolation(
                    "blank_canvas",
                    f"ink coverage {stats.ink_fraction:.4f} is below the "
                    f"{MIN_INK_FRACTION} minimum",
                )
            )
    return GateResult(
        passed=not violations,
        violations=violations,
        source=report,
        stats=stats,
        png=png,
    )
