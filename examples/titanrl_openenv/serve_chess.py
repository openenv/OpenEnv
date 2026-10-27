# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serve OpenEnv's ``envs/chess_env`` with enough capacity for RL rollouts.

``python -m envs.chess_env.server.app`` is the right way to *try* the chess
environment, but it is not enough to *train* against it: ``create_app`` defaults
to ``max_concurrent_envs=1``, so the second concurrent rollout gets

    Server error: Server at capacity: 1/1 sessions active. (code: CAPACITY_REACHED)

A TitanRL run needs far more than one step's worth of sessions. The async loop
keeps a whole *buffer* of rollout groups in flight -- ``(target_offpolicy_steps
+ 1) * num_prompts_per_train_step`` groups, each of ``num_samples_per_prompt``
rollouts. Validation runs only before the first step and after the last, so it
matters only if it is larger::

    max_sessions >= max((target_offpolicy_steps + 1)
                        * num_prompts_per_train_step
                        * num_samples_per_prompt,
                        validation.num_samples)

For the 30B recipe's defaults (3, 8, 8, 64) that is 4 * 8 * 8 = 256, not the 64
of a single step. Undersizing it does not fail cleanly: the server
accepts the WebSocket and then closes it, so most rollouts die on the next
send with ``ConnectionClosedOK`` rather than a readable ``CAPACITY_REACHED``,
and they are scored as ``ERROR``. The default below is generous on purpose --
the cap is only a ceiling, and sessions are created lazily. With ``--workers``
above 1 it applies per worker process (each has its own session table); size it
against the full demand anyway, since connections need not spread evenly.

Raising the cap requires the environment to opt in: ``HTTPEnvServer`` refuses
``max_concurrent_envs > 1`` unless the environment class sets
``SUPPORTS_CONCURRENT_SESSIONS = True``. ``ChessEnvironment`` does not set it
today, even though it is safe to run concurrently — the server builds one
instance per session, each owning its own ``chess.Board``, and moonfish's
``search_move`` takes the board as an argument rather than keeping engine state
(its only global is a pure memoization dict). So this launcher opts in with a
subclass rather than patching the environment.

Capacity alone is not enough: the server also has to be *fast* enough. moonfish
is pure Python, so every opponent reply burns CPU under the GIL, and one
uvicorn process answers them one at a time however many sessions are open.
Measured on one box, a single worker holds ~13.5 env steps/s no matter the
load, so latency grows linearly with concurrency:

===========  ==============  ===============
 sessions     step p50        step p95
===========  ==============  ===============
 1            0.05 s          0.05 s
 8            0.29 s          1.02 s
 32           2.09 s          2.96 s
 128          8.90 s          15.04 s
===========  ==============  ===============

At the ~256 sessions the 30B recipe asks for, a single env step costs ~18 s and
a four-turn rollout spends over a minute waiting on the opponent. That starves
the generator, and the trainer's ranks sit in a collective until NCCL's
600 s watchdog aborts them -- a training crash whose stack trace says nothing
about chess. Hence ``--workers``: separate processes, separate GILs.

It also pins the agent's color, which the environment otherwise alternates per
episode — see :func:`build_app`.

Usage::

    python examples/titanrl_openenv/serve_chess.py --max-sessions 512 --workers 16

Run it from the OpenEnv repo root so ``envs`` is importable, and install the
chess environment's own dependencies first (``pip install python-chess
moonfish``).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

# Allow running this file directly from a checkout: ``python
# examples/titanrl_openenv/serve_chess.py``.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from envs.chess_env.models import ChessAction, ChessObservation  # noqa: E402
from envs.chess_env.server.chess_environment import ChessEnvironment  # noqa: E402
from openenv.core.env_server import create_app  # noqa: E402

# Defaults, also used when this module is imported by ``uvicorn --workers``.
DEFAULT_MAX_SESSIONS = int(os.environ.get("CHESS_MAX_SESSIONS", "512"))
DEFAULT_OPPONENT_DEPTH = int(os.environ.get("CHESS_OPPONENT_DEPTH", "1"))
DEFAULT_MAX_MOVES = int(os.environ.get("CHESS_MAX_MOVES", "200"))
DEFAULT_AGENT_COLOR = os.environ.get("CHESS_AGENT_COLOR", "white")
DEFAULT_WORKERS = int(os.environ.get("CHESS_WORKERS", "16"))


class ConcurrentChessEnvironment(ChessEnvironment):
    """``ChessEnvironment`` that declares itself safe for concurrent sessions.

    Nothing else changes: the server already gives every WebSocket session its
    own environment instance and its own executor thread.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True


def build_app(
    *,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    opponent_depth: int = DEFAULT_OPPONENT_DEPTH,
    max_moves: int = DEFAULT_MAX_MOVES,
    agent_color: str | None = DEFAULT_AGENT_COLOR,
):
    """Build the FastAPI app that serves the chess environment for training."""
    # ``"alternate"`` is the CLI/env spelling of the environment's own default;
    # normalize it here so worker processes, which read the setting back out of
    # the environment as a string, agree with a direct call.
    if agent_color == "alternate":
        agent_color = None
    # ``functools.partial`` keeps the class visible to the server's concurrency
    # check, which unwraps partials before looking for the opt-in flag.
    factory = functools.partial(
        ConcurrentChessEnvironment,
        opponent="moonfish",
        # moonfish is pure Python, so every opponent reply costs CPU on the
        # server. Depth 1 keeps a 64-rollout step from becoming CPU-bound; raise
        # it if you want a stronger opponent and can spare the cores.
        opponent_depth=opponent_depth,
        max_moves=max_moves,
        # ``ChessEnvironment`` defaults to alternating the agent's color per
        # episode, which is wrong for GRPO: the rollouts in a group are ranked
        # against each other, so if half of them play white and half play black
        # they are not the same task and the reward spread is dominated by which
        # color the coin landed on rather than by how well the model played.
        # Pin it. Train on the other side by running a second server with
        # ``--agent-color black``.
        agent_color=agent_color,
    )
    return create_app(
        factory,
        ChessAction,
        ChessObservation,
        env_name="chess_env",
        max_concurrent_envs=max_sessions,
    )


# Module-level app so this can also be served as
# ``uvicorn titanrl_openenv.serve_chess:app --workers N``.
app = build_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS,
        help="Concurrent WebSocket sessions to allow, per worker process. This "
        "has to cover the async loop's whole in-flight buffer, not one step: "
        "(target_offpolicy_steps + 1) * num_prompts_per_train_step * "
        "num_samples_per_prompt (or validation.num_samples, if larger).",
    )
    parser.add_argument("--opponent-depth", type=int, default=DEFAULT_OPPONENT_DEPTH)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument(
        "--agent-color",
        choices=("white", "black", "alternate"),
        default=DEFAULT_AGENT_COLOR,
        help="Color the agent plays. 'alternate' restores the environment's "
        "per-episode coin flip, which makes the rollouts in a GRPO group "
        "incomparable; prefer pinning it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Server processes. moonfish is pure Python, so a single worker "
        "serializes every opponent reply on one core; see the module "
        "docstring. 1 disables multiprocessing.",
    )
    args = parser.parse_args()

    import uvicorn

    if args.workers > 1:
        # uvicorn can only fork workers from an import string, so the settings
        # travel to the children through the environment that the module-level
        # ``app`` reads.
        os.environ["CHESS_MAX_SESSIONS"] = str(args.max_sessions)
        os.environ["CHESS_OPPONENT_DEPTH"] = str(args.opponent_depth)
        os.environ["CHESS_MAX_MOVES"] = str(args.max_moves)
        os.environ["CHESS_AGENT_COLOR"] = args.agent_color
        # ``titanrl_openenv`` also has to be importable in the children. They
        # are spawned, not forked, and multiprocessing seeds a spawned child's
        # ``sys.path`` from this process's -- so put ``examples`` on both that
        # and PYTHONPATH, since only the former survives when the parent was
        # started as ``python examples/titanrl_openenv/serve_chess.py``.
        examples_dir = os.path.join(_REPO_ROOT, "examples")
        if examples_dir not in sys.path:
            sys.path.insert(0, examples_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            p for p in (examples_dir, os.environ.get("PYTHONPATH")) if p
        )
        uvicorn.run(
            "titanrl_openenv.serve_chess:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
        )
        return

    uvicorn.run(
        build_app(
            max_sessions=args.max_sessions,
            opponent_depth=args.opponent_depth,
            max_moves=args.max_moves,
            agent_color=None if args.agent_color == "alternate" else args.agent_color,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
