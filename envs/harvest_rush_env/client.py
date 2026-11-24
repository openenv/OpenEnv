# SPDX-License-Identifier: BSD-3-Clause

"""Harvest Rush environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HarvestRushAction, HarvestRushObservation


class HarvestRushEnv(EnvClient[HarvestRushAction, HarvestRushObservation, State]):
    """
    Client for the Harvest Rush environment.

    Keeps a persistent WebSocket connection to the environment server. Each
    client instance has its own session on the server. An episode is a single
    step: `reset()` issues one contact decision and `step()` scores it.

    Examples:

    ```python
    with HarvestRushEnv(base_url="http://localhost:8000").sync() as client:
        obs = client.reset().observation
        result = client.step(HarvestRushAction(choice="swerve"))
        print(result.reward, result.observation.metadata["kind"])
    ```
    """

    def _step_payload(self, action: HarvestRushAction) -> Dict:
        """
        Convert an action to the JSON payload of a step message.

        Args:
            action ([`HarvestRushAction`]):
                The decision to send.

        Returns:
            `dict`: The JSON-serialisable payload.
        """
        return {"choice": action.choice, "message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[HarvestRushObservation]:
        """
        Parse a server response into a step result.

        Args:
            payload (`dict`):
                The JSON response from the server.

        Returns:
            [`~openenv.core.client_types.StepResult`] holding a [`HarvestRushObservation`].
        """
        obs_data = payload.get("observation", {})
        observation = HarvestRushObservation(
            system=obs_data.get("system", ""),
            prompt=obs_data.get("prompt", ""),
            options=obs_data.get("options", []),
            parsed_choice=obs_data.get("parsed_choice"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse a server response into a state object.

        Args:
            payload (`dict`):
                The JSON response to a state request.

        Returns:
            [`~openenv.core.env_server.types.State`] with the episode id and step count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
