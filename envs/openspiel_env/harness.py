# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Harness-oriented OpenSpiel session adapters.

Follows the pattern introduced by ``browsergym_env.harness``: exposes an
OpenSpiel client as a ``ResourceSession`` driven through MCP-style tools,
so it can be consumed by ``openenv.core.harness`` adapters (e.g. the TTT
teacher rollouts used by the collect pipeline).
"""

from __future__ import annotations

from typing import Any, Callable

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ResourceSessionFactory,
    StepEnvSessionAdapter,
    ToolResult,
)

from .client import OpenSpielEnv
from .models import OpenSpielAction

_OPENSPIEL_TOOLS: list[Tool] = [
    Tool(
        name="play_move",
        description=(
            "Play a move by its integer action_id. The action_id must come "
            "from the `legal_actions` list shown in the latest observation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "integer",
                    "description": "Action id drawn from legal_actions.",
                },
            },
            "required": ["action_id"],
        },
    ),
]


def render_tic_tac_toe_board(info_state: list[float]) -> str:
    """Render a TTT ``info_state`` (27 floats, 3 planes) as a 3x3 grid.

    OpenSpiel's TTT info_state stores an empty/X/O one-hot for each cell
    across three consecutive 9-cell planes. Empty cells are rendered with
    their action_id so the prompt doubles as an action legend.
    """
    if len(info_state) != 27:
        return ""

    rows: list[str] = []
    for r in range(3):
        cells: list[str] = []
        for c in range(3):
            idx = r * 3 + c
            x_plane = info_state[9 + idx]
            o_plane = info_state[18 + idx]
            if x_plane > 0.5:
                cells.append("X")
            elif o_plane > 0.5:
                cells.append("O")
            else:
                cells.append(str(idx))
        rows.append(" | ".join(cells))
    return "\n---------\n".join(rows)


def _format_initial_prompt(observation: Any, game_name: str) -> str:
    legal = list(getattr(observation, "legal_actions", []) or [])
    info_state = list(getattr(observation, "info_state", []) or [])
    lines = [
        f"You are playing {game_name}.",
        "Call the play_move tool with an action_id drawn from legal_actions.",
        f"Legal actions: {legal}",
    ]
    if game_name == "tic_tac_toe":
        board = render_tic_tac_toe_board(info_state)
        if board:
            lines.extend(["", "Board:", board])
    return "\n".join(lines)


def _build_tool_result(game_name: str) -> Callable[..., ToolResult]:
    def builder(
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        state: Any,
    ) -> ToolResult:
        observation = result.observation
        legal = list(getattr(observation, "legal_actions", []) or [])
        info_state = list(getattr(observation, "info_state", []) or [])
        data: dict[str, Any] = {
            "action_id": arguments.get("action_id"),
            "legal_actions": legal,
            "reward": result.reward,
            "done": result.done,
            "current_player_id": getattr(observation, "current_player_id", None),
        }
        if game_name == "tic_tac_toe":
            board = render_tic_tac_toe_board(info_state)
            if board:
                data["board"] = board
        return ToolResult(
            data=data,
            done=bool(result.done),
            metadata={
                "reward": result.reward,
                "state": state.model_dump() if hasattr(state, "model_dump") else state,
            },
        )

    return builder


class OpenSpielSessionFactory(ResourceSessionFactory):
    """Create OpenSpiel-backed resource sessions for harness rollouts."""

    def __init__(
        self,
        client_factory: Callable[[], OpenSpielEnv],
        *,
        game_name: str = "tic_tac_toe",
    ):
        self._client_factory = client_factory
        self._game_name = game_name

    def create(
        self,
        task: Any = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> StepEnvSessionAdapter:
        client = self._client_factory()
        game = self._game_name

        return StepEnvSessionAdapter(
            client=client,
            task=task,
            seed=seed,
            episode_id=episode_id,
            tool_specs=list(_OPENSPIEL_TOOLS),
            action_builder=lambda name, arguments: OpenSpielAction(
                action_id=int(arguments["action_id"]),
                game_name=game,
            ),
            initial_messages_builder=lambda result, current_task: [
                {
                    "role": "user",
                    "content": _format_initial_prompt(result.observation, game),
                }
            ],
            tool_result_builder=_build_tool_result(game),
        )


__all__ = [
    "OpenSpielSessionFactory",
    "render_tic_tac_toe_board",
]
