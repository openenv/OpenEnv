# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward functions that score a rollout from the OpenEnv rewards it reported.

OpenEnv environments already emit a reward per step; :class:`OpenEnvMessageEnv`
forwards those into each turn's ``env_rewards``. :class:`OpenEnvReward` simply
aggregates them into the single scalar TitanRL's :class:`Rubric` needs, so the
environment's own reward drives training.

:class:`OpenEnvShapingReward` is the same aggregation over a second
``env_rewards`` key, for environments whose reward is too sparse to rank a group
of rollouts on its own — see
:func:`~examples.titanrl_openenv.tasks.chess_position_rewards`.

This module imports ``torchtitan``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rubric import RewardFn


class OpenEnvReward(RewardFn):
    """Aggregate the per-step OpenEnv rewards attached to a rollout's turns."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        reward_name: str = "openenv"
        """The ``env_rewards`` key to read (must match the env's ``reward_name``)."""

        aggregate: str = "sum"
        """How to combine per-turn rewards: ``"sum"``, ``"last"``, or ``"mean"``."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # sets self.weight from RewardFn.Config
        if config.aggregate not in ("sum", "last", "mean"):
            raise ValueError(
                f"aggregate must be 'sum', 'last', or 'mean', got {config.aggregate!r}"
            )
        self._reward_name = config.reward_name
        self._aggregate = config.aggregate

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        rewards = [
            turn.env_rewards[self._reward_name]
            for turn in rollout.turns
            if self._reward_name in turn.env_rewards
        ]
        if not rewards:
            return 0.0
        if self._aggregate == "last":
            return float(rewards[-1])
        if self._aggregate == "mean":
            return float(sum(rewards) / len(rewards))
        return float(sum(rewards))


class OpenEnvShapingReward(OpenEnvReward):
    """:class:`OpenEnvReward` over a shaping key, as a separately named reward fn.

    Behaviorally identical to its base class — only the name differs, and
    :class:`Rubric` uses the class name both as the uniqueness key for its
    reward-fn list (two ``OpenEnvReward`` entries are rejected) and as the label
    in ``reward_breakdown``. So a rubric that reads two ``env_rewards`` keys —
    the environment's own reward plus a shaping term contributed by a
    :class:`~examples.titanrl_openenv.tasks.TaskProfile` — needs two classes,
    and gets the two terms logged separately for free.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OpenEnvReward.Config):
        """Same fields as the base config.

        Redeclared because ``Configurable.__init_subclass__`` only wires
        ``Config._owner`` for a class that declares its own nested ``Config`` —
        inheriting it would make ``build()`` construct an ``OpenEnvReward``, and
        the rubric would reject the pair as duplicate names.
        """
