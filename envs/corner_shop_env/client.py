# Copyright 2026 Kayam
# SPDX-License-Identifier: Apache-2.0

"""Client for the Corner Shop environment (one WebSocket session per client)."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import CornerShopAction, CornerShopObservation, CornerShopState


class CornerShopEnv(
    EnvClient[CornerShopAction, CornerShopObservation, CornerShopState]
):
    """
    Client for the Corner Shop environment.

    Example:
        >>> with CornerShopEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(firm=7)
        ...     while not result.done:
        ...         reply = my_model(result.observation.system_prompt, result.observation.prompt)
        ...         result = env.step(CornerShopAction(message=reply))
        ...     print(result.reward, result.observation.metrics["cash_usd"])
    """

    def _step_payload(self, action: CornerShopAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CornerShopObservation]:
        fields = set(CornerShopObservation.model_fields) - {
            "done",
            "reward",
            "metadata",
        }
        obs = {k: v for k, v in payload.get("observation", {}).items() if k in fields}
        observation = CornerShopObservation(
            **obs, done=payload.get("done", False), reward=payload.get("reward")
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CornerShopState:
        return CornerShopState(**payload)
