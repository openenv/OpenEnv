# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-rollout inputs for the TorchTitan-RL <-> OpenEnv example.

An :class:`OpenEnvSample` carries everything one rollout needs: the opening
instruction shown to the model, the ``reset`` kwargs forwarded to the OpenEnv
server (e.g. a per-episode ``seed``), and an optional ``target`` the rubric can
score against. :class:`OpenEnvDataset` is a small, seeded, endless stream of
these samples — enough to drive training without depending on a specific
dataset. Swap it for your own :class:`~torchtitan.config.Configurable` dataset
to train on real tasks.

This module imports ``torchtitan``; the OpenEnv-only bridge lives in
``openenv_bridge.py`` and has no such dependency.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class OpenEnvSample:
    """One rollout's input to the OpenEnv-backed environment."""

    prompt: str = ""
    """Opening instruction rendered as the first user message."""

    reset_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the OpenEnv server's ``reset`` (e.g. seed)."""

    target: Any = None
    """Optional ground-truth passed through to the rubric; ``None`` if unused."""


class OpenEnvDataset(Configurable):
    """Endless, seeded stream of :class:`OpenEnvSample` from an inline prompt list.

    Row order is shuffled with ``seed`` and reshuffled on each wrap, so a run sees
    a fresh permutation every epoch. Each sample's ``reset_kwargs`` gets a
    per-rollout ``seed`` derived from the base seed so OpenEnv episodes are
    reproducible.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        prompts: list[str] = field(default_factory=list)
        """Pool of opening instructions to draw from.

        Left empty (the default), every sample carries an empty prompt and
        :class:`~examples.titanrl_openenv.titanrl_env.OpenEnvMessageEnv` falls
        back to the task profile's instruction. Set it to train on a real prompt
        set (e.g. one opening description per position).
        """

        seed: int = 42
        """Seed for the row-order shuffle and per-episode reset seeds."""

        shuffle: bool = True
        """Shuffle order (reshuffling on wrap); set False for deterministic eval."""

        pass_episode_seed: bool = True
        """When True, put a derived ``seed`` into each sample's ``reset_kwargs``."""

    def __init__(self, config: Config) -> None:
        # An empty pool means "no per-sample prompt" -> the task profile's
        # instruction is used instead.
        self._prompts = list(config.prompts) or [""]
        self._seed = config.seed
        self._shuffle = config.shuffle
        self._pass_episode_seed = config.pass_episode_seed
        self._rng = random.Random(config.seed)
        self._order = list(range(len(self._prompts)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0
        self._episode = 0

    def __iter__(self) -> Iterator[OpenEnvSample]:
        return self

    def __next__(self) -> OpenEnvSample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        reset_kwargs: dict[str, Any] = {}
        if self._pass_episode_seed:
            reset_kwargs["seed"] = self._seed + self._episode
        self._episode += 1
        return OpenEnvSample(
            prompt=self._prompts[idx],
            reset_kwargs=reset_kwargs,
        )

    def state_dict(self) -> dict:
        """Snapshot RNG + position.

        TitanRL does not call this on checkpoint/resume yet, so a resumed run
        restarts the data stream; it is here for when it does.
        """
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
            "episode": self._episode,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]
        self._episode = state_dict["episode"]
