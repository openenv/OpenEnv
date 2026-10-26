# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A tiny, dependency-free OpenEnv environment used by the bridge tests.

The showcase demo (``run_bridge_demo.py``) drives OpenEnv's real
:mod:`envs.chess_env`, which pulls in ``python-chess`` and ``moonfish``. This
module exists so the bridge's live round-trip tests can still run with nothing
but ``openenv`` + ``uvicorn`` installed, and so the ``generic`` task profile is
exercised against a second, differently-shaped environment.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Optional

from openenv.core.env_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

WORDS = ["apple", "banana", "cherry", "date", "elderberry"]


class GuessAction(Action):
    """Action schema: a single word guess."""

    guess: str = ""


class GuessObservation(Observation):
    """Observation schema: a human-readable line of feedback."""

    text: str = ""


class WordGuessEnv(Environment):
    """Guess-the-fruit game: reward 1.0 for the right word, else 0.0.

    Deterministic given ``reset(seed=...)`` so runs are reproducible. The episode
    ends on a correct guess or after ``max_steps`` wrong ones.
    """

    def __init__(self, max_steps: int = 6) -> None:
        super().__init__()
        self._max_steps = max_steps
        self._secret = WORDS[0]
        self._steps = 0
        self._state = State()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GuessObservation:
        self._secret = random.Random(seed).choice(WORDS)
        self._steps = 0
        self._state = State(episode_id=episode_id, step_count=0)
        return GuessObservation(
            text=(f"I'm thinking of a fruit ({len(WORDS)} options). Guess the word."),
            done=False,
            reward=None,
        )

    def step(
        self,
        action: GuessAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GuessObservation:
        self._steps += 1
        self._state.step_count = self._steps
        guess = (action.guess or "").strip().lower()
        if guess == self._secret:
            return GuessObservation(
                text=f"Correct! The word was '{self._secret}'.",
                done=True,
                reward=1.0,
            )
        done = self._steps >= self._max_steps
        suffix = f" The word was '{self._secret}'." if done else " Try again."
        return GuessObservation(
            text=f"'{guess}' is not it.{suffix}",
            done=done,
            reward=0.0,
        )

    @property
    def state(self) -> State:
        return self._state


def build_app():
    """Build the FastAPI app serving :class:`WordGuessEnv`."""
    return create_app(
        WordGuessEnv,
        GuessAction,
        GuessObservation,
        env_name="word_guess",
    )


def serve_in_background(
    host: str = "127.0.0.1",
    port: int = 8765,
    startup_timeout_s: float = 15.0,
):
    """Serve the tiny env in a daemon thread; return ``(server, thread)``."""
    import uvicorn

    config = uvicorn.Config(build_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + startup_timeout_s
    while not server.started:
        if time.time() > deadline:
            server.should_exit = True
            raise TimeoutError("OpenEnv test server did not start in time")
        time.sleep(0.05)
    return server, thread
