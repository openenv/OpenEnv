# SPDX-License-Identifier: BSD-3-Clause

"""The semantic layer: does the rendered drawing show what was asked for?

Two questions, in this order. First a blind caption: the judge is shown the
picture and asked what it is, with no mention of the task, so a leading question
cannot inflate the answer. Then a checklist of individual yes-or-no features,
because "rate this pelican 0-10" drifts between runs and "does it have a throat
pouch" mostly does not.

The core [`~openenv.core.rubrics.LLMJudge`] is text-only, so the multimodal
client plumbing here is new rather than reused.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .tasks import Task

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

# Weight split between the blind caption and the feature checklist.
BLIND_WEIGHT = 0.5

_CAPTION_PROMPT = (
    "Describe this image in one short sentence. Name the things you can "
    "actually identify. If you cannot tell what it depicts, say so plainly. "
    "Do not guess to be helpful."
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class VisionClient(Protocol):
    """Minimal protocol for a multimodal chat endpoint."""

    model: str

    async def complete_with_image(
        self,
        prompt: str,
        png_bytes: bytes,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 400,
    ) -> str:
        """Send a prompt plus an image and return the reply text."""


class HFVisionClient:
    """Vision client backed by Hugging Face Inference Providers.

    Args:
        model (`str`, *optional*, defaults to `"Qwen/Qwen2.5-VL-72B-Instruct"`):
            Repository id of the judge model.
        api_key (`str`, *optional*):
            Token to authenticate with. Falls back to the ambient Hugging Face
            token when omitted.
        timeout (`float`, *optional*, defaults to `120.0`):
            Per-request timeout in seconds.

    Examples:

    ```python
    client = HFVisionClient(model="Qwen/Qwen3-VL-30B-A3B-Instruct")
    reply = await client.complete_with_image("What is this?", png)
    ```
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        from huggingface_hub import get_token

        self.model = model
        self._api_key = api_key or get_token()
        self._timeout = timeout

    async def complete_with_image(
        self,
        prompt: str,
        png_bytes: bytes,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 400,
    ) -> str:
        """Send a prompt plus an image and return the reply text."""
        from huggingface_hub import AsyncInferenceClient

        uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "checklist", "schema": schema, "strict": True},
            }
        # The client is built per call rather than held on the instance. Its
        # underlying session binds to whichever event loop created it, and the
        # synchronous `step()` path runs each call under its own short-lived
        # loop, so a cached client fails with "Event loop is closed" on every
        # request after the first.
        async with AsyncInferenceClient(
            api_key=self._api_key, timeout=self._timeout
        ) as client:
            response = await client.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                **kwargs,
            )
        return response.choices[0].message.content or ""


@dataclass(frozen=True)
class JudgeReport:
    """What the judge concluded about a rendered drawing.

    Attributes:
        caption (`str`):
            The judge's unprompted description of the image.
        blind_subject (`float`):
            `1.0` if the caption names the requested animal, `0.5` if it names
            only its family, `0.0` otherwise.
        blind_vehicle (`float`):
            `1.0` if the caption names the requested vehicle.
        checklist (`dict[str, bool]`):
            Per-feature verdicts.
        model (`str`):
            Which judge produced this verdict. Recorded because the judge
            dominates the score: on one identical drawing Qwen2.5-VL-72B scored
            the semantic component 0.639 and Qwen3-VL-30B scored it 0.875. A
            result that does not say who judged it is not reproducible.
        error (`str` or `None`):
            Set when the judge could not be reached, in which case the scores
            are zero and the caller should treat the sample as unjudged rather
            than as a failure.
    """

    caption: str = ""
    blind_subject: float = 0.0
    blind_vehicle: float = 0.0
    checklist: dict[str, bool] = field(default_factory=dict)
    model: str = ""
    error: str | None = None

    @property
    def blind_score(self) -> float:
        """`float`: Mean of the two unprompted recognition signals."""
        return (self.blind_subject + self.blind_vehicle) / 2.0

    @property
    def checklist_score(self) -> float:
        """`float`: Fraction of checklist items answered yes."""
        if not self.checklist:
            return 0.0
        return sum(1 for v in self.checklist.values() if v) / len(self.checklist)

    @property
    def score(self) -> float:
        """`float`: Weighted combination of the blind and checklist scores."""
        if self.error is not None:
            return 0.0
        return (
            BLIND_WEIGHT * self.blind_score
            + (1.0 - BLIND_WEIGHT) * self.checklist_score
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "model": self.model,
            "caption": self.caption,
            "blind_subject": self.blind_subject,
            "blind_vehicle": self.blind_vehicle,
            "blind_score": round(self.blind_score, 4),
            "checklist": self.checklist,
            "checklist_score": round(self.checklist_score, 4),
            "score": round(self.score, 4),
            "error": self.error,
        }


def _mentions(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(t.lower())}\b", lowered) for t in terms)


def build_checklist(task: Task) -> dict[str, str]:
    """Build the per-feature questions for a task.

    Args:
        task ([`Task`]):
            The drawing request being judged.

    Returns:
        `dict[str, str]`: Question keyed by stable metric name.
    """
    questions: dict[str, str] = {
        "subject_recognisable": (
            f"Is the animal in this drawing recognisably a {task.subject.name}, "
            "rather than a generic animal shape?"
        ),
        "vehicle_recognisable": (
            f"Is the vehicle in this drawing recognisably a {task.vehicle.name}?"
        ),
        "riding_posture": (
            "Is the animal positioned on top of the vehicle in a riding "
            "posture, rather than beside it, under it, or floating away from it?"
        ),
        "coherent_figure": (
            "Do the parts join into one coherent figure, rather than reading as "
            "disconnected shapes that happen to overlap?"
        ),
    }
    for index, feature in enumerate(task.subject.features):
        questions[f"subject_feature_{index}"] = f"Does the animal have {feature}?"
    for index, feature in enumerate(task.vehicle.features):
        questions[f"vehicle_feature_{index}"] = f"Does the vehicle have {feature}?"
    return questions


def _checklist_schema(questions: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: {"type": "boolean"} for key in questions},
        "required": list(questions),
        "additionalProperties": False,
    }


def _checklist_prompt(questions: dict[str, str]) -> str:
    lines = "\n".join(f"- {key}: {text}" for key, text in questions.items())
    return (
        "You are grading a drawing that a language model produced blind, by "
        "writing SVG coordinates without ever seeing the result. Judge only "
        "what is actually visible in the image.\n\n"
        "Answer each question true or false. Be strict: answer false when the "
        "feature is absent, ambiguous, or you have to squint to find it. A "
        "crude but correct drawing should score well; a drawing that is merely "
        "in the right general area should not.\n\n"
        f"{lines}\n\n"
        "Reply with a JSON object using exactly these keys."
    )


class VisionJudge:
    """Scores a rendered drawing against the task it was meant to satisfy.

    Args:
        client ([`VisionClient`]):
            The multimodal endpoint used for both calls.

    Examples:

    ```python
    judge = VisionJudge(HFVisionClient())
    report = await judge.evaluate(png_bytes, task)
    print(report.caption, report.score)
    ```
    """

    def __init__(self, client: VisionClient):
        self._client = client

    @property
    def model(self) -> str:
        """`str`: The judge model id, recorded on every result."""
        return self._client.model

    async def evaluate(self, png_bytes: bytes, task: Task) -> JudgeReport:
        """Run the blind caption and the feature checklist against a drawing.

        Args:
            png_bytes (`bytes`):
                The rasterised submission.
            task ([`Task`]):
                The request the drawing was meant to satisfy.

        Returns:
            [`JudgeReport`]: Caption, per-feature verdicts and scores. On a
                transport failure the report carries `error` and scores zero.
        """
        questions = build_checklist(task)
        try:
            caption, raw_checklist = await asyncio.gather(
                self._client.complete_with_image(
                    _CAPTION_PROMPT, png_bytes, max_tokens=120
                ),
                self._client.complete_with_image(
                    _checklist_prompt(questions),
                    png_bytes,
                    schema=_checklist_schema(questions),
                    max_tokens=400,
                ),
            )
        except Exception as exc:
            return JudgeReport(model=self.model, error=f"{type(exc).__name__}: {exc}")

        caption = (caption or "").strip()
        if _mentions(caption, task.subject.synonyms):
            blind_subject = 1.0
        elif _mentions(caption, task.subject.family):
            blind_subject = 0.5
        else:
            blind_subject = 0.0
        blind_vehicle = 1.0 if _mentions(caption, task.vehicle.synonyms) else 0.0

        checklist = _parse_checklist(raw_checklist, questions)

        return JudgeReport(
            caption=caption,
            blind_subject=blind_subject,
            blind_vehicle=blind_vehicle,
            checklist=checklist,
            model=self.model,
        )


def _parse_checklist(raw: str, questions: dict[str, str]) -> dict[str, bool]:
    """Parse the judge's JSON reply, defaulting missing keys to false.

    A missing or unparseable answer counts against the submission rather than
    for it, so a flaky judge cannot inflate a score. Only a JSON `true` counts:
    the schema asks for booleans, and truthy strings such as `"false"` or
    `"no"` from a judge that ignored it must not read as approval.
    """
    payload: dict[str, Any] = {}
    match = _JSON_BLOCK.search(raw or "")
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {}
    return {key: payload.get(key) is True for key in questions}
