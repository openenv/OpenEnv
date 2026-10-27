# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fraud Triage Environment Client.

This module provides the client for connecting to a Fraud Triage
Environment server via WebSocket for persistent, low-latency multi-step
episodes.

Example:
    >>> from fraud_triage_env import FraudTriageEnv, FraudTriageAction, TriageDecision
    >>>
    >>> with FraudTriageEnv(base_url="http://localhost:8000").sync() as client:
    ...     result = client.reset()
    ...     print(result.observation.amount, result.observation.amount_zscore)
    ...
    ...     result = client.step(FraudTriageAction(decision=TriageDecision.FLAG))
    ...     print(result.reward, result.done)
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import FraudTriageAction, FraudTriageObservation, FraudTriageState


class FraudTriageEnv(
    EnvClient[FraudTriageAction, FraudTriageObservation, FraudTriageState]
):
    """
    Client for the Fraud Triage Environment.

    Maintains a persistent WebSocket connection to the environment server,
    streaming one synthetic transaction at a time and scoring the agent's
    approve/flag/escalate/block decisions against hidden ground-truth labels.

    Example:
        >>> with FraudTriageEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     print(result.observation.amount)
        ...
        ...     result = client.step(FraudTriageAction(decision=0))  # APPROVE
        ...     print(result.reward, result.done)
    """

    def _step_payload(self, action: FraudTriageAction) -> Dict[str, Any]:
        """Convert FraudTriageAction to JSON payload for the step request."""
        return {"decision": int(action.decision)}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[FraudTriageObservation]:
        """Parse server response into StepResult[FraudTriageObservation]."""
        obs_data = payload.get("observation", {})

        observation = FraudTriageObservation(
            transaction_id=obs_data.get("transaction_id", ""),
            amount=obs_data.get("amount", 0.0),
            merchant_category=obs_data.get("merchant_category", 0),
            velocity_1h=obs_data.get("velocity_1h", 0),
            velocity_24h=obs_data.get("velocity_24h", 0),
            amount_zscore=obs_data.get("amount_zscore", 0.0),
            is_new_merchant=obs_data.get("is_new_merchant", False),
            is_new_device=obs_data.get("is_new_device", False),
            cross_border=obs_data.get("cross_border", False),
            hour_of_day=obs_data.get("hour_of_day", 0),
            account_age_days=obs_data.get("account_age_days", 0),
            legal_actions=obs_data.get("legal_actions", [0, 1, 2, 3]),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FraudTriageState:
        """Parse server response into a FraudTriageState object."""
        return FraudTriageState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            episode_length=payload.get("episode_length", 200),
            true_positives=payload.get("true_positives", 0),
            false_positives=payload.get("false_positives", 0),
            false_negatives=payload.get("false_negatives", 0),
            true_negatives=payload.get("true_negatives", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
