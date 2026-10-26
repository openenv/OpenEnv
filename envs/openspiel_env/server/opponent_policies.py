# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Opponent policies for multi-player OpenSpiel games.

These policies are used to control non-agent players in multi-player games,
allowing single-agent RL training against fixed or adaptive opponents.
"""

import random
from typing import Any, Protocol


class OpponentPolicy(Protocol):
    """Protocol for opponent policies."""

    def select_action(
        self, legal_actions: list[int], observations: dict[str, Any]
    ) -> int:
        """
        Select an action for the opponent.

        Args:
            legal_actions: List of legal action IDs.
            observations: Current observations from the environment.

        Returns:
            Selected action ID.
        """
        ...


class RandomOpponent:
    """Random opponent that selects uniformly from legal actions."""

    def select_action(
        self, legal_actions: list[int], observations: dict[str, Any]
    ) -> int:
        """Select a random legal action."""
        if not legal_actions:
            raise ValueError("No legal actions available")
        return random.choice(legal_actions)


class FixedActionOpponent:
    """Opponent that always selects the same action (e.g., first legal action)."""

    def __init__(self, action_selector: str = "first"):
        """
        Initialize fixed action opponent.

        Args:
            action_selector: Which action to select ("first", "last", "middle").
        """
        self.action_selector = action_selector

    def select_action(
        self, legal_actions: list[int], observations: dict[str, Any]
    ) -> int:
        """Select a fixed legal action based on selector."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        if self.action_selector == "first":
            return legal_actions[0]
        elif self.action_selector == "last":
            return legal_actions[-1]
        elif self.action_selector == "middle":
            return legal_actions[len(legal_actions) // 2]
        else:
            return legal_actions[0]


def get_opponent_policy(policy_name: str) -> OpponentPolicy:
    """
    Get an opponent policy by name.

    Args:
        policy_name: Name of the policy ("random", "first", "last", "middle").

    Returns:
        OpponentPolicy instance.

    Raises:
        ValueError: If policy_name is not recognized.
    """
    if policy_name == "random":
        return RandomOpponent()
    elif policy_name in ("first", "last", "middle"):
        return FixedActionOpponent(action_selector=policy_name)
    else:
        raise ValueError(f"Unknown opponent policy: {policy_name}")
