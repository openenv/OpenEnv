# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fraud Triage Environment - a sequential transaction risk-decisioning task.

An agent is streamed synthetic financial transactions one at a time and must
choose to APPROVE, FLAG, ESCALATE, or BLOCK each one. Reward is asymmetric:
missing real fraud is penalized far more heavily than over-flagging a
legitimate transaction, and escalation carries a small fixed cost to
discourage defaulting to it -- mirroring the precision/recall trade-offs and
cost structure real payment-risk and fraud-detection teams operate under.

Example:
    >>> from fraud_triage_env import FraudTriageEnv, FraudTriageAction, TriageDecision
    >>>
    >>> with FraudTriageEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset()
    ...     print(result.observation.amount, result.observation.amount_zscore)
    ...
    ...     result = env.step(FraudTriageAction(decision=TriageDecision.FLAG))
    ...     print(result.reward, result.done)
"""

from .client import FraudTriageEnv
from .models import (
    FraudTriageAction,
    FraudTriageObservation,
    FraudTriageState,
    TriageDecision,
)

__all__ = [
    "FraudTriageEnv",
    "FraudTriageAction",
    "FraudTriageObservation",
    "FraudTriageState",
    "TriageDecision",
]
