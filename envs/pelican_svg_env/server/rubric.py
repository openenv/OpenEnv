# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition for the Pelican SVG environment.

The children are thin readers: all the work happens once in
[`~envs.pelican_svg_env.server.scoring.evaluate_submission`], which hangs its
results on the observation. Composing them therefore costs nothing and the tree
stays introspectable via `env.rubric.named_rubrics()`.

See RFC 004 for the rubric design: `rfcs/004-rubrics.md`.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum

from .scoring import component_weights


class GatePassed(Rubric):
    """One if the submission cleared the deterministic checks, zero otherwise."""

    def forward(self, action: Any, observation: Any) -> float:
        """Return the gate verdict as a score."""
        return 1.0 if observation.gate_passed else 0.0


class StructureScore(Rubric):
    """Fraction of structural checks the drawing passed."""

    def forward(self, action: Any, observation: Any) -> float:
        """Return the structural score carried on the observation."""
        return observation.structure_score


class SemanticScore(Rubric):
    """The vision judge's verdict on whether the drawing shows the subject."""

    def forward(self, action: Any, observation: Any) -> float:
        """Return the semantic score carried on the observation."""
        return observation.semantic_score


def build_rubric(judge_enabled: bool = True) -> Rubric:
    """Assemble the reward tree.

    The gate comes first and zeroes everything when it fails, so a submission
    that embedded a bitmap cannot collect structural credit for the bitmap's
    contents. Past the gate, structure and semantics are combined by weight.

    Args:
        judge_enabled (`bool`, *optional*, defaults to `True`):
            Whether a vision judge is configured. Without one the semantic
            score is unmeasured rather than bad, so structure takes the full
            weight. The split comes from
            [`~envs.pelican_svg_env.server.scoring.component_weights`], the
            same function the observation's reward uses.

    Returns:
        [`~openenv.core.rubrics.Rubric`]: The composed rubric.

    Examples:

    ```python
    rubric = build_rubric(judge_enabled=False)
    reward = rubric(action, observation)
    ```
    """
    structure_weight, semantic_weight = component_weights(judge_enabled)
    return Sequential(
        Gate(GatePassed(), threshold=1.0),
        WeightedSum(
            [StructureScore(), SemanticScore()],
            weights=[structure_weight, semantic_weight],
        ),
    )
