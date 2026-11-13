"""Client for the Spatial-Gym OpenEnv server (one WebSocket session per client)."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import SpatialGymAction, SpatialGymObservation, SpatialGymState
except ImportError:
    from models import SpatialGymAction, SpatialGymObservation, SpatialGymState


class SpatialGymEnv(
    EnvClient[SpatialGymAction, SpatialGymObservation, SpatialGymState]
):
    """Connect to a running server, a Docker image or a Hugging Face Space.

    >>> with SpatialGymEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(difficulty=1, seed=0)
    ...     result = env.step(SpatialGymAction(action=0))
    """

    def _step_payload(self, action: SpatialGymAction) -> Dict[str, Any]:
        return {"action": action.action}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SpatialGymObservation]:
        fields = dict(payload.get("observation", {}))
        fields["done"] = payload.get("done", False)
        fields["reward"] = payload.get("reward")
        fields["metadata"] = fields.get("metadata") or payload.get("metadata") or {}
        observation = SpatialGymObservation(
            **{
                k: v
                for k, v in fields.items()
                if k in SpatialGymObservation.model_fields
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SpatialGymState:
        return SpatialGymState(**payload)
