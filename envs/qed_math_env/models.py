# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the QED Math Environment.

Defines action and observation types for mathematical proof submission
and grading.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


RewardValue = bool | int | float | None


class QEDMathAction(Action):
    """Base action for the QED Math environment."""


class SubmitProof(QEDMathAction):
    """Submit a proof attempt for the current problem."""

    proof: str = Field(..., description="The proof text submitted by agent")
    attempt_number: int = Field(default=1, description="Attempt counter")


class GetProblem(QEDMathAction):
    """Request the current problem statement."""


class GetGradingGuidelines(QEDMathAction):
    """Request the grading guidelines/rubric for current problem."""


class QEDMathObservation(Observation):
    """Base observation for the QED Math environment."""


class ProblemObservation(QEDMathObservation):
    """Observation containing the problem statement."""

    problem: str = Field(default="", description="The mathematical problem")
    reference_solution: str = Field(default="", description="Ground truth solution")
    grading_guidelines: str = Field(
        default="", description="Rubric for grading (0-7 scale)"
    )
    problem_id: str = Field(default="", description="Unique problem identifier")
    dataset_source: str = Field(default="", description="Source dataset name")
    problem_type: str = Field(
        default="proof",
        description="Problem type: proof, answer, or multi_step",
    )
    max_attempts: int = Field(
        default=1,
        description="Maximum number of allowed submission attempts",
    )


class ProofSubmissionObservation(QEDMathObservation):
    """Observation returned after submitting a proof."""

    proof: str = Field(default="", description="The submitted proof")
    score: int = Field(default=0, description="Grade from rubric (0-7)")
    feedback: str = Field(default="", description="Grader feedback")
    reward: RewardValue = Field(
        default=0.0,
        description="Normalized reward (score/7)",
    )
    done: bool = Field(default=True, description="Episode ends after proof submission")
    problem_type: str = Field(
        default="proof",
        description="Problem type used to evaluate this submission",
    )
    attempt_number: int = Field(
        default=1, description="1-based submission attempt index"
    )
    attempts_remaining: int = Field(
        default=0,
        description="Remaining submission attempts in the current episode",
    )
    is_correct: bool = Field(
        default=False,
        description="Whether the submission is considered fully correct",
    )
