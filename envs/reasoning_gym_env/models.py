# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Reasoning Gym Environment.

The reasoning_gym environment integrates the Reasoning Gym library to provide
single-step reasoning tasks. Each episode presents one question, the agent submits
an answer, and receives a score.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ReasoningGymAction(Action):
    """Action for the Reasoning Gym environment - the agent's answer."""

    answer: str = Field(..., description="The agent's answer to the current question")


class ReasoningGymObservation(Observation):
    """Observation from the Reasoning Gym environment."""

    question: Optional[str] = Field(
        default=None,
        description="The current question to answer (None after step)",
    )
    score: Optional[float] = Field(
        default=None,
        description="Score for the answer (0.0 to 1.0 range, from dataset.score_answer())",
    )
    correct_answer: Optional[str] = Field(
        default=None,
        description="The correct answer (revealed after step)",
    )
    dataset_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata from the reasoning gym dataset entry",
    )
