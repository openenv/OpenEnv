"""
Cloud SRE OpenEnv Client.

Implements EnvClient for communicating with the Cloud SRE environment
via WebSocket/HTTP when deployed as a Docker container or HF Space.
"""

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import SREAction, SREObservation, SREState


class CloudSREEnv(EnvClient[SREAction, SREObservation, SREState]):
    """
    Client for the Cloud SRE & FinOps environment.

    Usage (async):
        async with CloudSREEnv(base_url="https://your-space.hf.space") as client:
            result = await client.reset()
            result = await client.step(SREAction(command="inspect", resource_id="ec2-web-001"))

    Usage (sync):
        with CloudSREEnv(base_url="...").sync() as client:
            result = client.reset()
            result = client.step(SREAction(command="terminate", resource_id="ebs-orphan-001"))
    """

    def _step_payload(self, action: SREAction) -> dict:
        """Serialize an SREAction into the JSON payload for the server."""
        payload = {"command": action.command}
        if action.resource_id:
            payload["resource_id"] = action.resource_id
        if action.params:
            payload["params"] = action.params
        return payload

    def _parse_result(self, payload: dict) -> StepResult[SREObservation]:
        """Deserialize the server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        obs = SREObservation(
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            resources=obs_data.get("resources", []),
            alerts=obs_data.get("alerts", []),
            total_hourly_cost=obs_data.get("total_hourly_cost", 0.0),
            system_uptime=obs_data.get("system_uptime", 100.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 15),
            budget_limit=obs_data.get("budget_limit"),
            task_description=obs_data.get("task_description", ""),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SREState:
        """Deserialize the server state response into a typed SREState."""
        return SREState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            current_step=payload.get("current_step", 0),
            done=payload.get("done", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            action_count=payload.get("action_count", 0),
        )
