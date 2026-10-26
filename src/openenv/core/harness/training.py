# SPDX-License-Identifier: BSD-3-Clause

"""Validated token capture shared by environment producers and trainers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingTurn(BaseModel):
    """One engine call. The required loss mask covers prompt plus completion."""

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(min_length=1)
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    per_token_logps: list[float]
    loss_mask: list[int]
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capture(self) -> TrainingTurn:
        from .capture.validate import validate_training_turn

        validate_training_turn(
            self.prompt_token_ids,
            self.completion_token_ids,
            self.per_token_logps,
            self.loss_mask,
        )
        return self


class TrainingTrace(BaseModel):
    """Selected agent calls, each with an authoritative mask and unique identity.

    Zero-masked turns remain available for context and usage accounting. Task rewards
    come from the session's `verify()` method, independently of these masks.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    turns: list[TrainingTurn]

    @model_validator(mode="after")
    def unique_calls(self) -> TrainingTrace:
        ids = [turn.node_id for turn in self.turns]
        if len(set(ids)) != len(ids):
            raise ValueError("training trace contains duplicate node_ids")
        return self

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> TrainingTrace:
        """Adapt producer-selected entries without inferring tokens or masks."""
        return cls(
            turns=[
                TrainingTurn(
                    node_id=entry["metadata"]["node_id"],
                    prompt_token_ids=entry["prompt_token_ids"],
                    completion_token_ids=entry["completion_token_ids"],
                    per_token_logps=entry["per_token_logps"],
                    loss_mask=entry["loss_mask"],
                    request=entry.get("request", {}),
                    response=entry.get("response", {}),
                    metadata=entry.get("metadata", {}),
                )
                for entry in entries
            ]
        )

    def to_trace_entries(self) -> list[dict[str, Any]]:
        """Return diagnostic records for reward functions and trace viewers."""
        return [
            {
                **turn.model_dump(exclude={"node_id"}),
                "metadata": {**turn.metadata, "node_id": turn.node_id},
            }
            for turn in self.turns
        ]
