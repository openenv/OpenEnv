# SPDX-License-Identifier: BSD-3-Clause

"""Reward composition for the chart-to-code environment.

The children only read scores that
[`~envs.chart2code_env.server.scoring.score_figure`] already put on the
observation, so the tree stays introspectable via `env.rubric.named_rubrics()`.
See RFC 004 for the rubric design: `rfcs/004-rubrics.md`.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum

from .scoring import WEIGHTS


class GatePassed(Rubric):
    """One if the code ran, drew data and used no raster image."""

    def forward(self, action: Any, observation: Any) -> float:
        """Return the gate verdict as a score."""
        return 1.0 if observation.gate_passed else 0.0


class ComponentScore(Rubric):
    """One component of the comparison: structure, data, text or color."""

    def __init__(self, name: str):
        super().__init__()
        self.component = name

    def forward(self, action: Any, observation: Any) -> float:
        """Return this component's score from the observation."""
        return observation.scores.get(self.component, 0.0)


def build_rubric() -> Rubric:
    """The gate, then the weighted components of
    [`~envs.chart2code_env.server.scoring.WEIGHTS`]."""
    return Sequential(
        Gate(GatePassed(), threshold=1.0),
        WeightedSum(
            [ComponentScore(name) for name in WEIGHTS],
            weights=list(WEIGHTS.values()),
        ),
    )
