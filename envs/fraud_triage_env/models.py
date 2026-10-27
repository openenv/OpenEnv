# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Fraud Triage Environment.

This module defines the Action, Observation, and State types for a
sequential transaction-risk-decisioning task exposed through the OpenEnv
interface.

An agent is streamed one transaction at a time and must decide how to
handle it. The environment rewards catching fraud while penalizing false
positives (blocking or escalating legitimate transactions) and false
negatives (approving fraudulent ones) at different weights, mirroring the
asymmetric cost structure real payment/fraud teams operate under.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class TriageDecision(IntEnum):
    """The four actions available to the agent for each transaction."""

    APPROVE = 0
    FLAG = 1  # allow, but tag for downstream/offline review
    ESCALATE = 2  # hold for a human analyst before settling
    BLOCK = 3  # reject the transaction outright


class FraudTriageAction(Action):
    """
    Action for the Fraud Triage environment.

    Attributes:
        decision: One of TriageDecision (0=APPROVE, 1=FLAG, 2=ESCALATE, 3=BLOCK).
    """

    decision: TriageDecision


class FraudTriageObservation(Observation):
    """
    Observation for the Fraud Triage environment.

    Represents a single transaction with engineered risk features, similar
    in spirit to the feature set a production anomaly-detection pipeline
    would compute in real time.

    Attributes:
        transaction_id: Unique identifier for this transaction.
        amount: Transaction amount, normalized to the account's typical range.
        merchant_category: Coarse merchant category code (0-9, synthetic).
        velocity_1h: Number of transactions on this account in the last hour.
        velocity_24h: Number of transactions on this account in the last 24h.
        amount_zscore: How many standard deviations this amount is from the
            account's historical mean transaction size.
        is_new_merchant: Whether this merchant has never been used by this
            account before.
        is_new_device: Whether this transaction originates from a device
            not previously associated with this account.
        cross_border: Whether the transaction crosses a national border.
        hour_of_day: Hour (0-23) the transaction occurred, local to the account.
        account_age_days: Age of the account in days.
        legal_actions: The list of valid TriageDecision values (always all four
            here, but included for interface consistency with other envs).
        done: Whether the episode (transaction stream) has ended.
        reward: Reward for the last action.
    """

    transaction_id: str = ""
    amount: float = 0.0
    merchant_category: int = 0
    velocity_1h: int = 0
    velocity_24h: int = 0
    amount_zscore: float = 0.0
    is_new_merchant: bool = False
    is_new_device: bool = False
    cross_border: bool = False
    hour_of_day: int = 0
    account_age_days: int = 0
    legal_actions: List[int] = Field(
        default_factory=lambda: [d.value for d in TriageDecision]
    )


class FraudTriageState(State):
    """
    State for the Fraud Triage environment.

    Attributes:
        episode_id: Unique ID for the current transaction stream.
        step_count: Number of transactions processed so far this episode.
        episode_length: Total number of transactions in this episode.
        true_positives: Fraudulent transactions correctly flagged/escalated/blocked.
        false_positives: Legitimate transactions incorrectly flagged/escalated/blocked.
        false_negatives: Fraudulent transactions incorrectly approved.
        true_negatives: Legitimate transactions correctly approved.
        cumulative_reward: Running sum of reward this episode.
    """

    episode_id: str = ""
    step_count: int = 0
    episode_length: int = 200
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    cumulative_reward: float = 0.0
