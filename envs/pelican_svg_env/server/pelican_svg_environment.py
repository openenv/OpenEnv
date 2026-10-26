# SPDX-License-Identifier: BSD-3-Clause

"""The Pelican SVG environment.

One episode is one drawing. `reset()` samples a task and returns the prompt,
`step()` scores the submitted SVG and ends the episode. Keeping episodes to a
single exchange is deliberate: the thing being measured is whether a model can
hold a spatial arrangement in its head and emit coordinates for it blind, and
letting it iterate would measure something else.
"""

from __future__ import annotations

import asyncio
import base64
import html
import os
from typing import Any, Optional

from huggingface_hub import get_token
from openenv.core.env_server import Environment

from ..models import PelicanSvgAction, PelicanSvgObservation, PelicanSvgState
from .rubric import build_rubric
from .scoring import evaluate_submission, Evaluation
from .tasks import CANONICAL_PAIR, make_task, sample_task, Task, task_from_ids
from .vision_judge import DEFAULT_JUDGE_MODEL, HFVisionClient, VisionJudge


def _judge_from_env() -> VisionJudge | None:
    """Build a judge from environment variables, or `None` to run offline.

    A missing token is not an error: the environment stays fully usable with
    the deterministic layers alone, which is what tests and cost-sensitive
    training loops want. It does mean no judge, though. Configuring one whose
    every call fails would keep the 0.35/0.65 weight split while `semantic`
    never scores, silently capping the reward at 0.35.
    """
    if os.environ.get("PELICAN_SVG_DISABLE_JUDGE", "").lower() in {"1", "true", "yes"}:
        return None
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )
    if not token:
        return None
    model = os.environ.get("PELICAN_SVG_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    return VisionJudge(HFVisionClient(model=model, api_key=token))


class PelicanSvgEnvironment(
    Environment[PelicanSvgAction, PelicanSvgObservation, PelicanSvgState]
):
    """Scores SVG drawings of an animal riding a vehicle.

    Args:
        subject (`str`, *optional*):
            Fix the animal instead of sampling one. Pair with `vehicle` to pin
            the task entirely, which is what a leaderboard run wants. Pinning
            just one fills the other from the canonical pelican-bicycle pair,
            and any pin takes precedence over `sample_tasks`.
        vehicle (`str`, *optional*):
            Fix the vehicle instead of sampling one.
        sample_tasks (`bool`, *optional*, defaults to `False`):
            Draw a random task on every reset instead of always serving Simon
            Willison's pelican-and-bicycle prompt. Off by default, because a
            default that changes on every reset silently makes two runs
            incomparable.
        held_out_only (`bool`, *optional*, defaults to `False`):
            Sample, but never the pelican-and-bicycle prompt. Implies
            `sample_tasks`. This is a hard promise: a pin that resolves to the
            canonical task raises `ValueError` rather than serving it.
        judge ([`VisionJudge`], *optional*):
            Semantic scorer. When omitted and `enable_judge` is `True`, one is
            built from a Hugging Face Inference Providers token if the ambient
            environment supplies one.
        enable_judge (`bool`, *optional*, defaults to `True`):
            Set `False` to score with the deterministic layers only. Passing
            `judge=None` on its own does not disable the judge, it just asks
            for the default one.
        return_image (`bool`, *optional*, defaults to `False`):
            Include the rendered PNG in the observation as base64. Useful for
            building a leaderboard gallery, wasteful during training.

    Examples:

    ```python
    env = PelicanSvgEnvironment(subject="capybara", vehicle="unicycle")
    observation = env.reset()
    print(observation.prompt)
    ```
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        subject: str | None = None,
        vehicle: str | None = None,
        sample_tasks: bool = False,
        held_out_only: bool = False,
        judge: VisionJudge | None = None,
        enable_judge: bool = True,
        return_image: bool = False,
    ):
        if not enable_judge:
            self._judge = None
        else:
            self._judge = judge if judge is not None else _judge_from_env()
        super().__init__(rubric=build_rubric(judge_enabled=self._judge is not None))
        self._fixed_subject = subject
        self._fixed_vehicle = vehicle
        self._sample_tasks = sample_tasks
        self._held_out_only = held_out_only
        self._return_image = return_image
        self._state = PelicanSvgState()
        self._task: Task | None = None
        self._svg = ""

    @property
    def state(self) -> PelicanSvgState:
        """[`PelicanSvgState`]: The current episode state."""
        return self._state

    @property
    def task(self) -> Task | None:
        """[`Task`] or `None`: The task sampled for this episode."""
        return self._task

    def _pick_task(
        self,
        seed: Optional[int],
        subject: str | None = None,
        vehicle: str | None = None,
        task_id: str | None = None,
    ) -> Task:
        if task_id:
            task = task_from_ids([task_id])[0]
        else:
            subject = subject or self._fixed_subject
            vehicle = vehicle or self._fixed_vehicle
            if subject or vehicle:
                # A partial pin is honoured, with the unpinned half taken from
                # the canonical pair rather than silently ignoring the request.
                task = make_task(
                    subject or CANONICAL_PAIR[0], vehicle or CANONICAL_PAIR[1]
                )
            elif self._sample_tasks or self._held_out_only:
                return sample_task(seed=seed, held_out_only=self._held_out_only)
            else:
                # Default to Simon Willison's original prompt. It is the one
                # with a public body of results behind it, and a default that
                # changes on every reset makes two runs incomparable by
                # accident.
                task = make_task(*CANONICAL_PAIR)
        if self._held_out_only and not task.held_out:
            raise ValueError(
                "held_out_only promises the canonical task is never served, "
                f"but this pin resolves to {task.task_id!r}. Pin a held-out "
                "combination or drop held_out_only."
            )
        return task

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PelicanSvgObservation:
        """Sample a task and return its prompt.

        Args:
            seed (`int`, *optional*):
                Makes task selection reproducible.
            episode_id (`str`, *optional*):
                Identifier recorded on the state.
            task_id (`str`, *optional*):
                Pin the exact task, for example `"capybara_unicycle"`. Takes
                precedence over everything else. A benchmark run wants this:
                without it every reset draws a fresh task and two models are
                never asked the same question.
            subject (`str`, *optional*):
                Pin the animal for this episode.
            vehicle (`str`, *optional*):
                Pin the vehicle for this episode.

        Returns:
            [`PelicanSvgObservation`]: Carrying the prompt and task metadata,
                with `done` false and no reward yet.

        Raises:
            KeyError: If a pinned task, subject or vehicle is not in the
                catalogue.
        """
        task = self._pick_task(
            seed,
            subject=kwargs.get("subject"),
            vehicle=kwargs.get("vehicle"),
            task_id=kwargs.get("task_id"),
        )
        self._task = task
        self._svg = ""
        self._state = PelicanSvgState(
            episode_id=episode_id, step_count=0, task_id=task.task_id, submitted=False
        )
        return self._apply_transform(
            PelicanSvgObservation(
                prompt=task.prompt,
                task_id=task.task_id,
                subject=task.subject.name,
                vehicle=task.vehicle.name,
                expected_wheels=task.vehicle.wheels,
                held_out=task.held_out,
                done=False,
                reward=None,
            )
        )

    def step(
        self,
        action: PelicanSvgAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PelicanSvgObservation:
        """Score a submission. Synchronous wrapper around [`step_async`]."""
        return asyncio.run(self.step_async(action, timeout_s=timeout_s, **kwargs))

    async def step_async(
        self,
        action: PelicanSvgAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PelicanSvgObservation:
        """Score a submission and end the episode.

        Args:
            action ([`PelicanSvgAction`]):
                The model's reply, from which the SVG is extracted.
            timeout_s (`float`, *optional*):
                Unused; scoring is bounded by the judge client's own timeout.

        Returns:
            [`PelicanSvgObservation`]: With `done` true, the reward, and the
                full scoring breakdown.

        Raises:
            RuntimeError: If called before [`reset`].
        """
        if self._task is None:
            raise RuntimeError("reset() must be called before step()")

        evaluation = await evaluate_submission(action.response, self._task, self._judge)
        self._state.step_count += 1
        self._state.submitted = True
        self._svg = evaluation.svg
        observation = self._to_observation(evaluation)
        # The rubric containers hand back a coroutine whenever they are called
        # from inside a running loop, even with entirely synchronous children,
        # so the reward has to be awaited through the async helper here.
        observation.reward = await self._apply_rubric_async(action, observation)
        return self._apply_transform(observation)

    def render_web(self, observation: dict[str, Any]) -> Optional[str]:
        """Draw the task on reset, then the submitted SVG next to its scores."""
        if "prompt" not in observation:
            return None
        breakdown = observation["breakdown"]
        if not breakdown:
            return (
                '<div style="max-width:500px;padding:10px;'
                "border:1px solid var(--border-color-primary);border-radius:10px;"
                'background:var(--background-fill-secondary);color:var(--body-text-color)">'
                f"{html.escape(observation['prompt'])}</div>"
            )
        # An <img> renders the SVG without running any script it carries.
        picture = (
            f'<img alt="Pelican SVG drawing" src="data:image/svg+xml;base64,'
            f'{base64.b64encode(self._svg.encode()).decode()}" style="width:200px;height:200px;'
            'object-fit:contain;background:white;border:1px solid var(--border-color-primary);border-radius:6px">'
            if self._svg
            else '<div style="width:200px;text-align:center">No SVG found</div>'
        )
        lines = [
            f"<b>reward {breakdown['reward']:.2f}</b>",
            f"gate: {'passed' if observation['gate_passed'] else 'rejected'}",
            f"structure: {observation['structure_score']:.2f}",
            f"judge: {observation['semantic_score']:.2f}"
            if observation["judged"]
            else "judge: off",
        ]
        if breakdown["judge"]:
            lines += [
                f"{'✓' if ok else '✗'} {html.escape(item)}"
                for item, ok in breakdown["judge"]["checklist"].items()
            ]
        lines.append(
            f'<span style="opacity:.75">{html.escape(observation["feedback"])}</span>'
        )
        return (
            '<div style="display:flex;'
            "flex-wrap:wrap;gap:12px;max-width:500px;padding:10px;border:1px solid var(--border-color-primary);"
            'border-radius:10px;background:var(--background-fill-secondary);color:var(--body-text-color)">'
            + picture
            + '<div style="flex:1;min-width:180px;display:flex;flex-direction:column;gap:4px">'
            + "".join(f"<div>{line}</div>" for line in lines)
            + "</div></div>"
        )

    def _to_observation(self, evaluation: Evaluation) -> PelicanSvgObservation:
        task = evaluation.task
        image: str | None = None
        if self._return_image and evaluation.gate.png is not None:
            image = base64.b64encode(evaluation.gate.png).decode()

        observation = PelicanSvgObservation(
            prompt=task.prompt,
            task_id=task.task_id,
            subject=task.subject.name,
            vehicle=task.vehicle.name,
            expected_wheels=task.vehicle.wheels,
            held_out=task.held_out,
            feedback=evaluation.feedback,
            gate_passed=evaluation.gate_passed,
            structure_score=evaluation.structure_score,
            semantic_score=evaluation.semantic_score,
            judged=evaluation.judged,
            violations=evaluation.gate.codes,
            breakdown=evaluation.to_dict(),
            image_png_base64=image,
            done=True,
        )
        return observation
