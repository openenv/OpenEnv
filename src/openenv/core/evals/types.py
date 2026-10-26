# SPDX-License-Identifier: BSD-3-Clause

"""Pydantic models for eval configuration and results."""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class EvalConfig(BaseModel):
    """Configuration for running an evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    harness_name: str = Field(description="Name of the evaluation harness")
    harness_version: str = Field(description="Version of the evaluation harness")
    library_versions: Dict[str, str] = Field(
        description="Versions of libraries used in the evaluation"
    )
    dataset: str = Field(description="Name of the dataset to evaluate on")
    eval_parameters: Dict[str, Any] = Field(description="Parameters for the evaluation")


class EvalResult(BaseModel):
    """Result of running an evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    config: EvalConfig = Field(description="Configuration used for the evaluation")
    scores: Dict[str, Any] = Field(description="Scores from the evaluation")
