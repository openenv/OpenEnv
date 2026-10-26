# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AdvocacyAction(Action):
    """The policy's one-shot advocacy argument."""

    text: str = Field(
        ..., description="The argument completion, using <claim>/<cite> tags."
    )


class AdvocacyObservation(Observation):
    """Task on reset; scored result on step.

    On reset: ``prompt`` holds the full system prompt (passage + question +
    answer-to-defend), ``done`` is False.
    On step: ``prompt`` is empty, ``done`` is True, and ``metadata`` carries all
    eight reward components.

    ``reward``/``done`` are inherited from the base ``Observation`` (reward
    defaults to ``None``). On step, the typed client mirrors the weighted
    aggregate and terminal flag from the response envelope into the observation,
    so they match ``StepResult.reward`` and ``StepResult.done``. ``reset()``
    leaves ``reward`` as ``None`` (no action scored yet), matching the framework
    convention.

    The eight reward components are also mirrored in the declared ``components``
    field. The base ``metadata`` dict is stripped by the framework's HTTP
    serialization layer, so ``components`` is what survives the wire; the typed
    client re-populates ``metadata`` from it on the way back.
    """

    prompt: str = Field("", description="Full prompt the policy must answer.")
    answer_to_defend: str = Field(
        "", description="The answer the policy advocates for."
    )
    item_id: str = Field("", description="Source QuALITY article id.")
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Eight reward components (mirror of metadata; survives HTTP).",
    )
    error: str = Field(
        "",
        description="Diagnostic message (e.g. step-before-reset); survives serialization.",
    )
