# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Models for the Terminus environment."""

from __future__ import annotations

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """Outcome of one shell command run inside the E2B sandbox."""

    command: str
    output: str = ""
    error: str | None = None
    success: bool = True


class TerminusState(State):
    """Per-session state for the single-tool Terminus environment."""

    sandbox_id: str | None = None
    setup_results: list[CommandResult] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    verify_results: list[CommandResult] = Field(default_factory=list)
    commands: list[CommandResult] = Field(default_factory=list)
    submitted_answer: str | None = None
    last_reward: float | None = None
    reward_override_ignored: str | None = None
    last_error: str | None = None
