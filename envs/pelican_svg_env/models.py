# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the Pelican SVG environment.

An episode is a single exchange: the environment states the drawing task, the
model replies with an SVG, and the environment scores it. There is no
multi-turn state to carry, so the observation is designed to be a complete
record of why a submission scored what it did.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class PelicanSvgAction(Action):
    """A submitted drawing.

    Attributes:
        response (`str`):
            The model's raw reply. The environment extracts the SVG from it, so
            surrounding prose or a fenced code block is fine. Handing over the
            unedited reply rather than a pre-cleaned document keeps extraction
            failures visible in the score instead of hidden in the harness.
    """

    response: str


class PelicanSvgObservation(Observation):
    """The task, or the verdict on an attempt at it.

    Attributes:
        prompt (`str`):
            The drawing instruction. Populated on reset.
        task_id (`str`):
            Identifier of the sampled task, `"{subject}_{vehicle}"`.
        subject (`str`):
            The animal that was requested.
        vehicle (`str`):
            The vehicle that was requested.
        expected_wheels (`int`):
            Wheels the vehicle should show in side view.
        held_out (`bool`):
            `False` only for the canonical pelican-and-bicycle pair, which is
            the one every lab can already optimise for.
        feedback (`str`):
            Short explanation of the score, safe to show to the model.
        gate_passed (`bool`):
            Whether the submission cleared the deterministic checks.
        structure_score (`float`):
            Fraction of structural checks passed, in [0, 1].
        semantic_score (`float`):
            Judge score, in [0, 1]. Zero when no judge ran.
        judged (`bool`):
            Whether a judge verdict was actually obtained. Distinguishes "the
            drawing scored zero" from "nobody looked at it".
        violations (`list[str]`):
            Gate violation codes, empty when the gate passed.
        breakdown (`dict[str, Any]`):
            The full per-layer analysis, for debugging and for leaderboards.
        image_png_base64 (`str` or `None`):
            The rendered submission, included only when the environment is
            configured to return it.
    """

    prompt: str = ""
    task_id: str = ""
    subject: str = ""
    vehicle: str = ""
    expected_wheels: int = 2
    held_out: bool = True
    feedback: str = ""
    gate_passed: bool = False
    structure_score: float = 0.0
    semantic_score: float = 0.0
    judged: bool = False
    violations: list[str] = Field(default_factory=list)
    breakdown: dict[str, Any] = Field(default_factory=dict)
    image_png_base64: str | None = None


class PelicanSvgState(State):
    """Internal state of an episode.

    Attributes:
        task_id (`str`):
            The task sampled for this episode.
        submitted (`bool`):
            Whether the single allowed submission has been made.
    """

    task_id: str = ""
    submitted: bool = False
