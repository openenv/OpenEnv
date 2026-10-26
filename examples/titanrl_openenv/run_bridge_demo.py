# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone, end-to-end demo of the TitanRL <-> OpenEnv bridge on real chess.

This script needs **only** ``openenv`` plus the chess environment's own
dependencies (``python-chess`` and ``moonfish``) — no ``torch`` / ``torchtitan``,
no GPU. It

1. serves OpenEnv's built-in :mod:`envs.chess_env` in-process over the real
   OpenEnv HTTP/WebSocket protocol,
2. drives it through :class:`~examples.titanrl_openenv.openenv_bridge.OpenEnvBridge`
   with the ``chess`` task profile, in both ``tool`` mode (the assistant "calls"
   ``chess_move``) and ``text`` mode (the assistant's message text is the move),
   and
3. plays a deliberately illegal move first, to show how the environment's
   ``-0.1`` rejection reward and the profile's renderer surface back to the model.

A scripted stand-in picks moves from the legal-move list the environment reports,
so the demo is deterministic and model-free — exactly the path
:class:`~examples.titanrl_openenv.titanrl_env.OpenEnvMessageEnv` takes during a
TitanRL rollout, with the policy replaced by a seeded RNG.

Run it from the OpenEnv repo root::

    pip install python-chess moonfish
    python examples/titanrl_openenv/run_bridge_demo.py

The environment factory and server helper are importable so the test suite can
reuse them for a live round-trip test.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import sys
import threading
import time
from typing import Any

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))

# Allow running as a plain script (``python run_bridge_demo.py``): make the
# sibling bridge modules importable without the ``examples.titanrl_openenv``
# package context, and the repo root importable so ``envs.chess_env`` resolves.
for _path in (_PKG_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The chess environment is a normal OpenEnv environment; importing it needs its
# own dependencies (python-chess, moonfish), not the training stack.
from envs.chess_env.models import ChessAction, ChessObservation  # noqa: E402
from envs.chess_env.server.chess_environment import ChessEnvironment  # noqa: E402
from openenv.core.env_server import create_app  # noqa: E402
from openenv_bridge import OpenEnvBridge  # noqa: E402
from tasks import CHESS_MOVE_TOOL, CHESS_TASK  # noqa: E402


# --------------------------------------------------------------------------- #
# Serve OpenEnv's built-in chess environment in-process.
# --------------------------------------------------------------------------- #


def _make_chess_env() -> ChessEnvironment:
    """Chess env pinned for a fast, reproducible demo.

    ``agent_color="white"`` removes ``ChessEnvironment``'s default per-episode
    colour alternation (which keys off ``hash(episode_id)``), and depth-1
    moonfish keeps each opponent reply quick on CPU.
    """
    return ChessEnvironment(
        opponent="moonfish",
        opponent_depth=1,
        agent_color="white",
        max_moves=40,
    )


def build_demo_app():
    """Build the FastAPI app serving ``chess_env`` over OpenEnv's protocol."""
    return create_app(
        _make_chess_env,
        ChessAction,
        ChessObservation,
        env_name="chess_env",
    )


def free_port() -> int:
    """Pick an unused localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_in_background(
    host: str = "127.0.0.1",
    port: int = 8765,
    startup_timeout_s: float = 15.0,
):
    """Serve the demo env in a daemon thread; return the ``uvicorn.Server``.

    Blocks until the server reports it has started (or raises on timeout). Stop
    it with ``server.should_exit = True``.
    """
    import uvicorn

    config = uvicorn.Config(build_demo_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + startup_timeout_s
    while not server.started:
        if time.time() > deadline:
            server.should_exit = True
            raise TimeoutError("OpenEnv demo server did not start in time")
        time.sleep(0.05)
    return server, thread


# --------------------------------------------------------------------------- #
# A scripted stand-in for the policy.
# --------------------------------------------------------------------------- #


def move_tool_call(move: str) -> dict[str, Any]:
    """An OpenAI-style ``chess_move`` tool call playing ``move``."""
    return {
        "function": {
            "name": CHESS_MOVE_TOOL["name"],
            "arguments": json.dumps({"move": move}),
        }
    }


def pick_move(observation: Any, rng: random.Random) -> str | None:
    """Choose a legal move from the observation, the way a policy would."""
    if not isinstance(observation, dict):
        return None
    legal_moves = observation.get("legal_moves") or []
    return rng.choice(legal_moves) if legal_moves else None


def _summarize(turn) -> str:
    """One-line view of a turn (the full rendering is multi-line)."""
    fen = (turn.raw_observation or {}).get("fen", "") if turn.raw_observation else ""
    return f"reward={turn.reward} done={turn.done} fen={fen.split(' ')[0][:32]}"


# --------------------------------------------------------------------------- #
# The two modes.
# --------------------------------------------------------------------------- #


async def _play_tool_mode(base_url: str, max_moves: int = 6) -> None:
    print("\n=== tool mode (assistant calls chess_move) ===")
    bridge = OpenEnvBridge(
        base_url=base_url,
        action_mode="tool",
        # chess_move's arguments *are* the action: {"move": "e2e4"} -> ChessAction
        tool_action_key=CHESS_TASK.tool_action_key,
        observation_renderer=CHESS_TASK.render,
    )
    rng = random.Random(0)
    turn = await bridge.start()
    print("reset ->")
    print(turn.text)
    try:
        # First, an illegal move: the env rejects it with -0.1 and the renderer
        # tells the model why, instead of failing the rollout.
        turn = await bridge.act_from_tool_calls([move_tool_call("a1a8")])
        print(f"\nillegal 'a1a8'  -> {_summarize(turn)}")
        print(turn.text.splitlines()[0])

        for i in range(max_moves):
            move = pick_move(turn.raw_observation, rng)
            if move is None:
                break
            turn = await bridge.act_from_tool_calls([move_tool_call(move)])
            print(f"move {i + 1} {move!r:8} -> {_summarize(turn)}")
            if turn.done:
                print(turn.text.splitlines()[-1])
                break
    finally:
        await bridge.stop()


async def _play_text_mode(base_url: str, max_moves: int = 4) -> None:
    print("\n=== text mode (assistant text is the move) ===")
    # In text mode the raw message text becomes the value of ``action_key``, so
    # point it at the chess action's field name.
    bridge = OpenEnvBridge(
        base_url=base_url,
        action_mode="text",
        action_key="move",
        observation_renderer=CHESS_TASK.render,
    )
    rng = random.Random(1)
    turn = await bridge.start()
    print(f"reset -> {_summarize(turn)}")
    try:
        for i in range(max_moves):
            move = pick_move(turn.raw_observation, rng)
            if move is None:
                break
            turn = await bridge.act_from_text(move)
            print(f"say  {i + 1} {move!r:8} -> {_summarize(turn)}")
            if turn.done:
                break
    finally:
        await bridge.stop()


async def main(port: int | None = None) -> None:
    port = port or free_port()
    server, _thread = serve_in_background(port=port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        await _play_tool_mode(base_url)
        await _play_text_mode(base_url)
        print(
            "\nDemo complete: TitanRL's bridge played OpenEnv's chess environment "
            "end-to-end over the real OpenEnv protocol."
        )
    finally:
        server.should_exit = True


if __name__ == "__main__":
    asyncio.run(main())
