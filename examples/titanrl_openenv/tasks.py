# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task profiles: the per-environment half of the TitanRL <-> OpenEnv example.

The bridge in ``openenv_bridge.py`` is deliberately environment-agnostic: it
speaks the OpenEnv protocol and nothing else. Real environments still need three
task-specific decisions, and a :class:`TaskProfile` bundles exactly those:

* **instruction** — the opening user message describing the task,
* **tool** — the tool schema the model calls, and which of its arguments form
  the OpenEnv action dict (``tool_action_key``),
* **render** — how to turn that environment's structured observation into text,
* **shaping** — optional extra per-step rewards read out of the observation,
  for environments whose own reward is too sparse to train on directly.

Profiles are looked up by name (:func:`get_task_profile`) so a TitanRL config can
select one with a plain string field, which keeps the configs dataclass-only and
CLI-overridable.

Two profiles ship here:

``"chess"``
    Drives OpenEnv's built-in :mod:`envs.chess_env` — a real, multi-turn,
    CPU-only environment with per-step rewards. This is the showcase.

``"generic"``
    The environment-agnostic fallback: the ``openenv_act`` tool with a free-form
    ``action`` object, and best-effort observation rendering. Use it to point the
    example at any other OpenEnv environment without writing a profile.

This module depends only on ``openenv`` (in fact only on the standard library),
so it is importable and testable without ``torch`` / ``torchtitan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:  # package import (``examples.titanrl_openenv`` / ``titanrl_openenv``)
    from .openenv_bridge import DEFAULT_ACT_TOOL, render_observation
except ImportError:  # flat import, with the example directory on sys.path
    from openenv_bridge import DEFAULT_ACT_TOOL, render_observation

__all__ = [
    "TaskProfile",
    "CHESS_INSTRUCTION",
    "CHESS_MOVE_TOOL",
    "CHESS_EVAL_SCALE",
    "CHESS_POSITION_REWARD",
    "render_chess_observation",
    "chess_position_rewards",
    "GENERIC_TASK",
    "CHESS_TASK",
    "TASK_PROFILES",
    "get_task_profile",
]


@dataclass(frozen=True)
class TaskProfile:
    """Everything environment-specific about one OpenEnv task.

    Attributes:
        name: Registry key used by ``OpenEnvMessageEnv.Config.task``.
        instruction: Opening user message, used when the dataset sample carries
            no prompt of its own.
        tool: Tool schema offered to the model in ``tool`` action mode; ``None``
            falls back to the generic ``openenv_act`` tool.
        tool_action_key: Tool-argument key holding the action object. ``None``
            means the tool's whole argument dict *is* the OpenEnv action — which
            is what a typed tool like ``chess_move(move=...)`` wants.
        render: Observation -> text renderer. ``None`` falls back to the bridge's
            generic :func:`~examples.titanrl_openenv.openenv_bridge.render_observation`.
        shaping: Observation -> extra ``{reward_name: value}`` entries, merged
            into the turn's ``env_rewards`` alongside the environment's own
            reward. It receives the observation with the step's ``reward`` and
            ``done`` put back in (the OpenEnv wire format moves them onto the
            step result). ``None`` means the environment's reward is the only
            signal.
            Use it when that reward is too sparse for GRPO to separate siblings
            on; each key is scored by its own ``OpenEnvReward`` in the rubric.
    """

    name: str
    instruction: str = ""
    tool: Optional[dict[str, Any]] = None
    tool_action_key: Optional[str] = "action"
    render: Optional[Callable[[Any], str]] = None
    shaping: Optional[Callable[[Any], dict[str, float]]] = None


# --------------------------------------------------------------------------- #
# chess — OpenEnv's built-in ``envs/chess_env`` (python-chess + moonfish)
# --------------------------------------------------------------------------- #

CHESS_INSTRUCTION = (
    "You are playing a full game of chess against an engine. Each turn you are "
    "given the position in FEN plus the complete list of legal moves. Pick the "
    "strongest move and call the `chess_move` tool exactly once with it in UCI "
    "notation (e.g. `e2e4`, `e7e8q` to promote). The move must come from the "
    "legal-move list — illegal or malformed moves are rejected, cost reward, and "
    "waste a turn. Win the game. Keep your reasoning short — a few sentences at "
    "most — then call the tool: a turn that runs out of tokens before the tool "
    "call is thrown away."
)

CHESS_MOVE_TOOL: dict[str, Any] = {
    "name": "chess_move",
    "description": (
        "Play one move in the current chess position. The move must be one of "
        "the legal moves listed in the latest observation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": "The move in UCI notation, e.g. 'g1f3' or 'e7e8q'.",
            },
        },
        "required": ["move"],
    },
}


def render_chess_observation(observation: Any) -> str:
    """Render a ``ChessObservation`` dict into the text the model reads.

    Surfaces the position (FEN), whose turn it is, check status, the full legal
    move list, and — because the environment answers an illegal move with a
    ``-0.1`` reward rather than an error — an explicit note when the previous
    move was rejected.
    """
    if not isinstance(observation, dict) or "fen" not in observation:
        return render_observation(observation)

    lines: list[str] = []

    # A rejected move pays -0.1 and leaves the game running; a lost game pays
    # -1.0 and ends it, so only a negative reward mid-game means "rejected".
    reward = observation.get("reward")
    try:
        rejected = (
            reward is not None and float(reward) < 0.0 and not observation.get("done")
        )
    except (TypeError, ValueError):
        rejected = False
    if rejected:
        lines.append(
            "Your previous move was rejected (not a legal UCI move) and was not "
            "played. The position is unchanged — choose a move from the list below."
        )

    fen = observation.get("fen") or ""
    if fen:
        lines.append(f"Position (FEN): {fen}")
        fields = fen.split()
        if len(fields) > 1:
            lines.append(f"Side to move: {'white' if fields[1] == 'w' else 'black'}")
    if observation.get("is_check"):
        lines.append("The side to move is in check.")

    legal_moves = observation.get("legal_moves") or []
    if legal_moves:
        lines.append(
            f"Legal moves ({len(legal_moves)}): {', '.join(str(m) for m in legal_moves)}"
        )

    if observation.get("done"):
        result = observation.get("result")
        lines.append(f"Game over{f' — result {result}' if result else ''}.")

    return "\n".join(lines) if lines else render_observation(observation)


CHESS_POSITION_REWARD = "position"
"""``env_rewards`` key holding the shaped position score (see :func:`chess_position_rewards`)."""

CHESS_EVAL_SCALE = 200.0
"""Divisor applied to moonfish's evaluation before squashing it into [-1, 1].

moonfish scores a position in centipawn-like units — roughly 60 per pawn and
1011 for a queen — so 200 puts a one-pawn edge at ``tanh(0.3) ≈ 0.29`` and a
three-pawn edge at ``tanh(0.9) ≈ 0.72``, leaving room above for the ±1.0 the
environment itself pays for checkmate.
"""


def chess_position_rewards(observation: Any) -> dict[str, float]:
    """Shape the sparse chess reward with the engine's own position evaluation.

    ``envs/chess_env`` pays ``+/-1.0`` for a decided game, ``-0.1`` for an
    illegal move, and **0.0 for every legal non-terminal move**. A rollout is a
    partial game of a handful of moves, so in practice nothing but 0.0 is ever
    paid: sibling rollouts all score identically, GRPO sees no advantage to
    learn from, and TitanRL's batcher eventually aborts the run with "N
    consecutive untrainable batches".

    The environment already computes what is needed to break the tie — it puts
    moonfish's static evaluation of the resulting position in
    ``metadata["evaluation"]`` on every step. This turns that into a bounded
    reward under :data:`CHESS_POSITION_REWARD`, so "played four moves and came
    out a pawn up" scores above "played four moves and hung a knight".

    No sign correction is needed mid-game: moonfish evaluates from the side to
    move's point of view, and the agent is the side to move in every ongoing
    observation (the environment plays the opponent's reply before returning),
    for either color. A finished game is scored by its outcome instead — the
    environment's terminal reward, +1 / 0 / -1 for a win / draw / loss — both
    because the result beats any static evaluation and because after the
    agent's own game-ending move the opponent never replies, so the evaluation
    would be from the opponent's side. That needs ``reward`` and ``done`` in the
    observation; the OpenEnv wire format moves them onto the step result, so
    callers put them back — :class:`~examples.titanrl_openenv.titanrl_env.OpenEnvMessageEnv`
    passes ``{**turn.raw_observation, "reward": turn.reward, "done": turn.done}``.
    """
    if not isinstance(observation, dict):
        return {}
    if observation.get("done"):
        try:
            outcome = float(observation.get("reward"))
        except (TypeError, ValueError):
            return {}
        return {CHESS_POSITION_REWARD: max(-1.0, min(1.0, outcome))}
    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    evaluation = metadata.get("evaluation")
    try:
        value = float(evaluation)
    except (TypeError, ValueError):
        return {}
    if math.isnan(value) or math.isinf(value):
        return {}
    return {CHESS_POSITION_REWARD: math.tanh(value / CHESS_EVAL_SCALE)}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

GENERIC_TASK = TaskProfile(
    name="generic",
    instruction="Interact with the environment to maximize your reward.",
    tool=DEFAULT_ACT_TOOL,
    tool_action_key="action",
    render=None,
)

CHESS_TASK = TaskProfile(
    name="chess",
    instruction=CHESS_INSTRUCTION,
    tool=CHESS_MOVE_TOOL,
    # ``chess_move``'s arguments are already the OpenEnv action: {"move": "e2e4"}
    # maps straight onto ChessAction, so there is no wrapper key to unpack.
    tool_action_key=None,
    render=render_chess_observation,
    shaping=chess_position_rewards,
)

TASK_PROFILES: dict[str, TaskProfile] = {
    GENERIC_TASK.name: GENERIC_TASK,
    CHESS_TASK.name: CHESS_TASK,
}


def get_task_profile(name: str) -> TaskProfile:
    """Look up a registered :class:`TaskProfile` by name."""
    try:
        return TASK_PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(TASK_PROFILES))
        raise ValueError(
            f"unknown task profile {name!r}; known profiles: {known}"
        ) from None
