# SPDX-License-Identifier: BSD-3-Clause

"""Harvest Rush as an OpenEnv environment.

A thin wrapper over the `harvest-rush-train` package: the same generated
examples and the same programmatic reward as its verifiers environment, served
one contact decision per episode. Configured with environment variables:

    HRT_MODE       control_consistent (default) | harm_averse
    HRT_SPLIT      train (default) | eval
    HRT_POOL_SIZE  examples generated at start-up and cycled (default 500)
    HRT_SEED       master seed for generation (default 0)
"""

import logging
import os
import threading
from typing import Any, Optional
from uuid import uuid4

from harvest_rush_train.generate import generate_examples
from harvest_rush_train.reward import MODES, parse_choice, score_choice
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HarvestRushAction, HarvestRushObservation
except ImportError:
    from harvest_rush_env.models import HarvestRushAction, HarvestRushObservation

log = logging.getLogger("harvest_rush_env")

_POOL: list = []
_POOL_LOCK = threading.Lock()
_CURSOR = 0


def _pool() -> list:
    """Generate the example pool once per process; sessions share it read-only."""
    global _POOL
    with _POOL_LOCK:
        if not _POOL:
            mode = os.environ.get("HRT_MODE", "control_consistent")
            if mode not in MODES:
                raise ValueError(f"HRT_MODE must be one of {MODES}, got {mode!r}")
            n = int(os.environ.get("HRT_POOL_SIZE", "500"))
            split = os.environ.get("HRT_SPLIT", "train")
            seed = int(os.environ.get("HRT_SEED", "0"))
            log.info(
                "generating pool: n=%d split=%s seed=%d mode=%s", n, split, seed, mode
            )
            _POOL = generate_examples(n, split, seed, mode)
            log.info("pool ready: %d examples", len(_POOL))
        return _POOL


def _next_index(seed: Optional[int]) -> int:
    global _CURSOR
    pool = _pool()
    if seed is not None:
        return int(seed) % len(pool)
    with _POOL_LOCK:
        i = _CURSOR % len(pool)
        _CURSOR += 1
    return i


class HarvestRushEnvironment(
    Environment[HarvestRushAction, HarvestRushObservation, State]
):
    """
    Single-step environment: one priced contact decision per episode.

    `reset()` draws the next example from a pool generated once per process,
    `step()` scores the choice with `harvest_rush_train.reward.score_choice`
    and ends the episode. Driving over an animal never scores above 0.
    """

    # Instances only hold a reference into the shared read-only pool.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._mode = os.environ.get("HRT_MODE", "control_consistent")
        self._row: Optional[dict] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HarvestRushObservation:
        """
        Start an episode by drawing one contact decision.

        Args:
            seed (`int`, *optional*):
                Selects a fixed example, `seed` modulo the pool size. Without
                it, examples are served in order.
            episode_id (`str`, *optional*):
                Identifier to use for the episode. A random one is generated
                when omitted.

        Returns:
            [`HarvestRushObservation`] with the briefing, the contact prompt and the offered options.
        """
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._row = _pool()[_next_index(seed)]
        msgs = self._row["prompt"]
        return HarvestRushObservation(
            system=msgs[0]["content"],
            prompt=msgs[-1]["content"],
            options=list(self._row["info"]["options"]),
            done=False,
            reward=0.0,
        )

    def step(  # type: ignore[override]
        self,
        action: HarvestRushAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HarvestRushObservation:
        """
        Score the decision and end the episode. Call `reset()` first.

        Args:
            action ([`HarvestRushAction`]):
                The choice, or the raw model reply to parse.
            timeout_s (`float`, *optional*):
                Unused; scoring is immediate.

        Returns:
            [`HarvestRushObservation`] with `done=True`, the reward, the parsed
            choice and the scoring metadata (`kind`, `condition`, `answer`,
            `format_ok`, `mode`).
        """
        if self._row is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.step_count:
            raise RuntimeError("The episode is over; call reset() before step().")
        self._state.step_count += 1
        info = self._row["info"]
        choice = action.choice if action.choice else parse_choice(action.message or "")
        if choice not in info["options"]:
            choice = None
        reward = score_choice(choice, info, self._mode)
        log.info(
            "episode=%s kind=%s choice=%s reward=%.2f",
            self._state.episode_id,
            info["kind"],
            choice,
            reward,
        )
        msgs = self._row["prompt"]
        return HarvestRushObservation(
            system=msgs[0]["content"],
            prompt=msgs[-1]["content"],
            options=list(info["options"]),
            parsed_choice=choice,
            done=True,
            reward=reward,
            metadata={
                "kind": info["kind"],
                "condition": info.get("condition"),
                "answer": self._row.get("answer"),
                "format_ok": choice is not None,
                "mode": self._mode,
            },
        )

    @property
    def state(self) -> State:
        """The current episode id and step count."""
        return self._state
