# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synthetic transaction stream generator for the Fraud Triage Environment.

Generates a stream of transactions with a configurable fraud rate. Legitimate
and fraudulent transactions are drawn from different feature distributions so
the task is learnable but not trivial -- fraud correlates with, but is not
perfectly determined by, any single feature (mirroring real payment data,
where no single rule catches everything).

This is intentionally dependency-light (pure Python + random) so the
environment has no heavyweight ML dependencies at the server layer -- an
agent is expected to learn the decision policy, not have one handed to it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Transaction:
    transaction_id: str
    amount: float
    merchant_category: int
    velocity_1h: int
    velocity_24h: int
    amount_zscore: float
    is_new_merchant: bool
    is_new_device: bool
    cross_border: bool
    hour_of_day: int
    account_age_days: int
    is_fraud: bool  # ground truth, withheld from the agent


class TransactionGenerator:
    """Generates a stream of synthetic transactions for one episode."""

    def __init__(self, fraud_rate: float = 0.12, seed: int | None = None):
        self.fraud_rate = fraud_rate
        self._rng = random.Random(seed)

    def sample(self) -> Transaction:
        is_fraud = self._rng.random() < self.fraud_rate

        if is_fraud:
            # Fraudulent transactions skew toward higher amounts, unusual
            # merchants/devices, higher velocity, and off-hours activity --
            # but with enough overlap with legitimate traffic that a naive
            # single-threshold rule will not solve the task perfectly.
            amount = max(1.0, self._rng.lognormvariate(5.2, 1.1))
            amount_zscore = self._rng.uniform(1.5, 6.0)
            velocity_1h = self._rng.randint(1, 8)
            velocity_24h = velocity_1h + self._rng.randint(0, 15)
            is_new_merchant = self._rng.random() < 0.7
            is_new_device = self._rng.random() < 0.55
            cross_border = self._rng.random() < 0.4
            hour_of_day = self._rng.choice(
                list(range(0, 6)) + list(range(0, 24))
            )  # biased toward late night/early morning
        else:
            amount = max(1.0, self._rng.lognormvariate(3.6, 0.9))
            amount_zscore = self._rng.uniform(-1.0, 1.8)
            velocity_1h = self._rng.randint(0, 2)
            velocity_24h = velocity_1h + self._rng.randint(0, 5)
            is_new_merchant = self._rng.random() < 0.15
            is_new_device = self._rng.random() < 0.05
            cross_border = self._rng.random() < 0.08
            hour_of_day = self._rng.randint(0, 23)

        return Transaction(
            transaction_id=f"{self._rng.getrandbits(128):032x}",
            amount=round(amount, 2),
            merchant_category=self._rng.randint(0, 9),
            velocity_1h=velocity_1h,
            velocity_24h=velocity_24h,
            amount_zscore=round(amount_zscore, 3),
            is_new_merchant=is_new_merchant,
            is_new_device=is_new_device,
            cross_border=cross_border,
            hour_of_day=hour_of_day,
            account_age_days=self._rng.randint(1, 2500),
            is_fraud=is_fraud,
        )
