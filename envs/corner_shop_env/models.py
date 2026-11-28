# Copyright 2026 Kayam
# SPDX-License-Identifier: Apache-2.0

"""Typed action, observation and state for the Corner Shop environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CornerShopAction(Action):
    """One week of decisions.

    Send either `message` (a model's raw reply: the last JSON object in it is used,
    preferring ```json blocks) or `decision` (the JSON action itself). If both are set,
    `decision` wins. An empty action is a week with no decisions. An unparseable reply is
    reported back in the next observation's `errors`, and the week still passes.

    `end_episode=True` stops playing: the remaining weeks run unattended (no decisions)
    and the episode is scored, so stopping early can never dodge a loss.
    """

    message: str = Field(
        default="", description="Raw model reply; the last JSON object is the action"
    )
    decision: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The week's action as a JSON object (prices, orders, pay, borrow, ...)",
    )
    end_episode: bool = Field(
        default=False, description="Run the remaining weeks unattended and score"
    )


class CornerShopObservation(Observation):
    """What the operator sees after reset or a step.

    `prompt` is the text to show a model as the next user turn (`system_prompt` goes in
    the system turn). `shop` is the same state as a structured dict. `reward` is 0 until
    the final step; then it is the episode score and `metrics` holds the full breakdown.
    """

    prompt: str = Field(
        default="", description="Next user turn: the week's state, inbox and the ask"
    )
    system_prompt: str = Field(
        default="", description="System prompt for a model playing the shop"
    )
    shop: Dict[str, Any] = Field(
        default_factory=dict, description="Structured state of the firm"
    )
    firm: int = Field(default=0, description="Firm index, 1..100")
    week: int = Field(default=0, description="Current week (0-based)")
    weeks_total: int = Field(default=0, description="Episode horizon in weeks")
    errors: List[str] = Field(
        default_factory=list, description="Parts of the last action that were refused"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Episode metrics (final step only)"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Episode details (final step only)"
    )


class CornerShopState(State):
    """Server-side episode state."""

    firm: int = 0
    week: int = 0
    weeks_total: int = 0
    done: bool = False
    bankrupt: bool = False
    cash: float = 0.0
    reward_kind: str = "going_concern"
    normalize: bool = True
    gc_rollouts: int = 16
    gc_weeks: int = 13
