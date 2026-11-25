# SPDX-License-Identifier: BSD-3-Clause

"""Chart-to-code environment: reproduce a chart with matplotlib code.

On reset the agent gets a PNG of a chart drawn from a seeded spec. It may send
its code with `mode="render"` up to `max_renders` times to see what the code
draws, with no score, then sends it with `mode="submit"`. The submission's
figure is compared with the target's artists (data, text, colors and
structure), so the reward is deterministic and needs no judge.
"""

from __future__ import annotations

import asyncio
import functools
import html
import random
from typing import Any, Optional

from openenv.core.env_server import Environment

from ..models import Chart2CodeAction, Chart2CodeObservation, Chart2CodeState
from .executor import run_code
from .rubric import build_rubric
from .scoring import score_figure
from .specs import (
    LEVELS,
    reference_code,
    sample_spec,
    SPLITS,
    TEST_SEED_OFFSET,
    TEST_SIZE,
)

PROMPT = (
    "Write Python code that uses matplotlib to reproduce the chart in the image "
    "(figure size {width:g} x {height:g} inches). Send it with mode 'render' to "
    "see what your code draws ({renders} renders, no score), then with mode "
    "'submit' to be scored on the chart's data, text, colors and layout. Pasting "
    "the image into the figure scores zero."
)


@functools.lru_cache(maxsize=256)
def _target(level: str, split: str, index: int) -> tuple[Any, dict, str]:
    """The spec, figure summary and PNG of a task, drawn by its reference code."""
    spec = sample_spec(level, split, index)
    result = run_code(reference_code(spec), spec.dpi, png=True, timeout_s=60)
    if result["error"]:
        raise RuntimeError(f"task {spec.task_id} failed to draw: {result['error']}")
    return spec, result["figure"], result["png"]


class Chart2CodeEnvironment(
    Environment[Chart2CodeAction, Chart2CodeObservation, Chart2CodeState]
):
    """Scores matplotlib code against a target chart.

    Args:
        level (`str`, *optional*, defaults to `"easy"`):
            Default level: `"easy"` (one series, values drawn on the chart),
            `"medium"` (several series, a legend, no value labels) or `"hard"`
            (two charts side by side, one with a log axis).
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` serves any seed below 10^9. `"test"` serves 500 fixed
            tasks per level that no train seed reaches.
        max_renders (`int`, *optional*, defaults to `4`):
            Drafts the agent may render before submitting.
        timeout_s (`float`, *optional*, defaults to `30.0`):
            Seconds the agent's code may run per render or submission.

    Examples:

    ```python
    env = Chart2CodeEnvironment(level="medium")
    observation = env.reset(seed=7)
    result = env.step(Chart2CodeAction(code=my_code, mode="submit"))
    print(result.reward, result.scores)
    ```
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        level: str = "easy",
        split: str = "train",
        max_renders: int = 4,
        timeout_s: float = 30.0,
    ):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        super().__init__(rubric=build_rubric())
        self._level = level
        self._split = split
        self._max_renders = max_renders
        self._timeout_s = timeout_s
        self._state = Chart2CodeState()
        self._task: tuple[Any, dict, str] | None = None
        self._final_observation: Chart2CodeObservation | None = None

    @property
    def state(self) -> Chart2CodeState:
        """[`Chart2CodeState`]: The current episode state."""
        return self._state

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Chart2CodeObservation:
        """Serve a chart to reproduce.

        Args:
            seed (`int`, *optional*):
                Picks the task: the train task with this seed, or test task
                `seed % 500`. Without a seed or `task_index` the task is random.
            episode_id (`str`, *optional*):
                Identifier recorded on the state.
            level (`str`, *optional*):
                Level for this episode, overriding the constructor's.
            split (`str`, *optional*):
                Split for this episode, overriding the constructor's.
            task_index (`int`, *optional*):
                Exact task number in the split, taking precedence over `seed`.

        Returns:
            [`Chart2CodeObservation`]: The target chart and the instructions.
        """
        level = kwargs.get("level") or self._level
        split = kwargs.get("split") or self._split
        index = kwargs.get("task_index")
        if index is None and seed is not None:
            index = seed % TEST_SIZE if split == "test" else seed
        if index is None:
            index = random.SystemRandom().randrange(
                TEST_SIZE if split == "test" else TEST_SEED_OFFSET
            )
        self._task = _target(level, split, index)
        spec, _, png = self._task
        self._final_observation = None
        self._state = Chart2CodeState(
            episode_id=episode_id, step_count=0, task_id=spec.task_id
        )
        width, height = spec.figsize
        return self._apply_transform(
            Chart2CodeObservation(
                prompt=PROMPT.format(
                    width=width, height=height, renders=self._max_renders
                ),
                task_id=spec.task_id,
                level=level,
                target_png_base64=png,
                width_in=width,
                height_in=height,
                dpi=spec.dpi,
                renders_left=self._max_renders,
                done=False,
                reward=None,
            )
        )

    def step(
        self,
        action: Chart2CodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Chart2CodeObservation:
        """Render a draft or score a submission.

        Once the submission is scored, later steps run nothing and return the
        final observation until [`reset`].

        Raises:
            RuntimeError: If called before [`reset`].
        """
        if self._task is None:
            raise RuntimeError("reset() must be called before step()")
        if self._final_observation is not None:
            return self._final_observation.model_copy(deep=True)
        spec, target, _ = self._task
        self._state.step_count += 1
        renders_left = self._max_renders - self._state.renders_used
        observation = Chart2CodeObservation(
            task_id=spec.task_id, level=spec.level, renders_left=renders_left
        )
        if action.mode == "render":
            if renders_left <= 0:
                observation.error = "no renders left; submit the code"
                return self._apply_transform(observation)
            self._state.renders_used += 1
            result = run_code(
                action.code, spec.dpi, png=True, timeout_s=self._timeout_s
            )
            observation.renders_left = renders_left - 1
            observation.rendered_png_base64 = result["png"]
            observation.error = result["error"]
            return self._apply_transform(observation)

        result = run_code(action.code, spec.dpi, png=True, timeout_s=self._timeout_s)
        verdict = score_figure(target, result["figure"])
        observation.rendered_png_base64 = result["png"]
        observation.error = result["error"] or verdict["reason"]
        observation.gate_passed = verdict["gate_passed"]
        observation.scores = {
            name: verdict[name] for name in ("structure", "data", "text", "color")
        }
        observation.done = True
        observation.reward = self._apply_rubric(action, observation)
        self._state.submitted = True
        self._final_observation = self._apply_transform(observation)
        return self._final_observation.model_copy(deep=True)

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Chart2CodeObservation:
        """[`reset`] in a thread: drawing a new target takes about a second."""
        return await asyncio.to_thread(self.reset, seed, episode_id, **kwargs)

    async def step_async(
        self,
        action: Chart2CodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Chart2CodeObservation:
        """[`step`] in a thread, so running the code never blocks the server."""
        return await asyncio.to_thread(self.step, action, timeout_s, **kwargs)

    def render_web(self, observation: dict[str, Any]) -> Optional[str]:
        """The target next to the latest drawing, with the scores below."""
        if not observation.get("target_png_base64") and self._task is None:
            return None

        def image(data: str | None, caption: str) -> str:
            body = (
                f'<img src="data:image/png;base64,{data}" style="max-width:100%">'
                if data
                else "<em>nothing drawn yet</em>"
            )
            return f'<figure style="flex:1;margin:0">{body}<figcaption>{caption}</figcaption></figure>'

        target = observation.get("target_png_base64") or (
            self._task[2] if self._task else ""
        )
        scores = observation.get("scores") or {}
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.3f}</td></tr>"
            for name, value in scores.items()
        )
        error = observation.get("error") or ""
        return (
            '<div style="display:flex;gap:12px">'
            + image(target, "target")
            + image(observation.get("rendered_png_base64"), "your code")
            + "</div>"
            + (f"<table>{rows}</table>" if rows else "")
            + (f"<pre>{html.escape(error)}</pre>" if error else "")
        )
