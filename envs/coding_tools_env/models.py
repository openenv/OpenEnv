# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Models for coding_tools_env."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import State
from pydantic import BaseModel, Field


class CommandResult(BaseModel):
    """Normalized result for one tool execution."""

    tool: str
    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TodoItem(BaseModel):
    """Todo item tracked by todo_write."""

    id: str
    content: str
    status: str
    priority: str


class CodingToolsState(State):
    """Per-session state for coding_tools_env."""

    sandbox_id: str | None = None
    setup_results: list[CommandResult] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    verify_results: list[CommandResult] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)
    tool_history: list[CommandResult] = Field(default_factory=list)
    submitted: bool = False
    last_reward: float | None = None
    reward_override_ignored: str | None = None
    last_error: str | None = None


class EditSpec(BaseModel):
    """One edit operation for multi_edit."""

    old_string: str
    new_string: str
    replace_all: bool = False
