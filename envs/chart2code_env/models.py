# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the chart-to-code environment.

An episode shows the agent a chart and asks for matplotlib code that draws it
again. The agent may render drafts to see what its code draws, then submits
once. Only the submission is scored.
"""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class Chart2CodeAction(Action):
    """Code to render as a draft or to submit for scoring.

    Attributes:
        code (`str`):
            Python source that draws the chart with matplotlib. It runs in a
            fresh process on the Agg backend; the last open figure is the one
            read.
        mode (`str`, *optional*, defaults to `"submit"`):
            `"render"` returns the PNG the code draws, or its error, and no
            score. `"submit"` scores the code and ends the episode.
    """

    code: str
    mode: Literal["render", "submit"] = "submit"


class Chart2CodeObservation(Observation):
    """The target chart, a rendered draft, or the verdict on a submission.

    Attributes:
        prompt (`str`):
            The instruction shown to the agent.
        task_id (`str`):
            `"{level}-{split}-{seed}"`, enough to rebuild the task.
        level (`str`):
            `"easy"`, `"medium"` or `"hard"`.
        target_png_base64 (`str`):
            The chart to reproduce.
        width_in (`float`):
            Width of the target figure, in inches.
        height_in (`float`):
            Height of the target figure, in inches.
        dpi (`int`):
            Resolution the target was saved at.
        rendered_png_base64 (`str` or `None`):
            What the agent's code drew, after a render or a submission.
        error (`str`):
            The traceback or rule the code broke, empty when it ran.
        renders_left (`int`):
            Drafts the agent can still render this episode.
        scores (`dict[str, float]`):
            Per-component scores of a submission: `structure`, `data`,
            `text` and `color`, each in [0, 1].
        gate_passed (`bool`):
            Whether the submission ran, drew a figure and used no raster image.
    """

    prompt: str = ""
    task_id: str = ""
    level: str = ""
    target_png_base64: str = ""
    width_in: float = 0.0
    height_in: float = 0.0
    dpi: int = 0
    rendered_png_base64: str | None = None
    error: str = ""
    renders_left: int = 0
    scores: dict[str, float] = Field(default_factory=dict)
    gate_passed: bool = False


class Chart2CodeState(State):
    """Internal state of an episode.

    Attributes:
        task_id (`str`):
            The task served this episode.
        renders_used (`int`):
            Drafts rendered so far.
        submitted (`bool`):
            Whether the episode's one submission has been scored.
    """

    task_id: str = ""
    renders_used: int = 0
    submitted: bool = False
