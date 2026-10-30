# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the Harvest Rush environment.

One episode is one contact decision: `reset()` returns the briefing and the
priced options, `step()` takes the choice and returns the programmatic reward
with `done=True`.
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class HarvestRushAction(Action):
    """
    The decision at one contact.

    Send either `choice` directly, or the raw model reply in `message` (one
    line of JSON, `{"choice": "<option>"}`), which is parsed exactly as the
    verifiers environment in the `harvest-rush-train` package parses it.
    """

    choice: Optional[str] = Field(
        default=None, description='"continue", "swerve" or "reroute"'
    )
    message: Optional[str] = Field(default=None, description="raw model reply to parse")


class HarvestRushObservation(Observation):
    """
    What the policy sees at one contact. `system` and `prompt` are the two chat
    messages to show the model; the scoring fields are filled in after `step()`.
    """

    system: str = Field(default="", description="the briefing (system message)")
    prompt: str = Field(
        default="", description="the contact and its priced options (user message)"
    )
    options: List[str] = Field(
        default_factory=list, description="choices offered at this contact"
    )
    parsed_choice: Optional[str] = Field(
        default=None, description="after step(): the choice that was scored"
    )
