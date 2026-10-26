"""OpenEnv client serialization for browser actions and observations."""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import BrowserAction, BrowserObservation, BrowserState


class BrowserClient(EnvClient[BrowserAction, BrowserObservation, BrowserState]):
    def _step_payload(self, action):
        return action.model_dump()

    def _parse_result(self, payload):
        observation = BrowserObservation.model_validate(payload["observation"])
        done = bool(payload.get("done", False) or observation.done)
        return StepResult(
            observation=observation.model_copy(update={"done": done}),
            reward=payload.get("reward"),
            done=done,
        )

    def _parse_state(self, payload):
        return BrowserState.model_validate(payload)
