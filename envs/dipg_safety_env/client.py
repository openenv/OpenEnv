# envs/dipg_safety_env/client.py
"""
Client implementation for the custom DIPGSafetyEnv.

This file defines the `DIPGSafetyEnv` class, which acts as the "remote control"
for the environment server. It maintains a persistent WebSocket connection
for efficient multi-step interactions:
  1.  It takes Python objects (like an Action) from the agent's code.
  2.  It converts them into JSON to send to the server.
  3.  It receives JSON responses from the server.
  4.  It parses that JSON back into useful Python objects (like Observations and Rewards).
"""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import DIPGAction, DIPGObservation, DIPGState


class DIPGSafetyEnv(EnvClient[DIPGAction, DIPGObservation, DIPGState]):
    """
    Client for interacting with the `DIPGSafetyEnv` server.

    This class inherits from the base `EnvClient` and is specialized to handle
    the specific data types of our environment: `DIPGAction` and `DIPGObservation`.
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initializes the client.

        Args:
            base_url: The URL of the running environment server.
            timeout: The number of seconds to wait for a server response.
        """
        # This correctly calls the parent initializer with the expected
        # 'message_timeout_s' keyword argument.
        super().__init__(base_url=base_url, message_timeout_s=timeout)

    # ----------------------------------------

    def _step_payload(self, action: DIPGAction) -> dict:
        """
        Formats the `DIPGAction` object into a JSON-serializable dictionary.

        This dictionary becomes the body of the HTTP POST request sent to the
        server's `/step` endpoint.

        Args:
            action: The `DIPGAction` object containing the model's response.

        Returns:
            A dictionary to be sent as the JSON request body.
        """
        return {"llm_response": action.llm_response}

    def _parse_result(self, payload: dict) -> StepResult[DIPGObservation]:
        """
        Parses the JSON payload from the server into a `StepResult`,
        robustly handling inconsistencies and potential missing data.

        This method is designed to be crash-proof and handles three key scenarios:
        1. The single-nested 'observation' dictionary from the `/reset` endpoint.
        2. The double-nested 'observation' dictionary from the `/step` endpoint.
        3. A payload where the 'observation' key might be missing entirely.

        Args:
            payload: The raw dictionary parsed from the server's JSON response.

        Returns:
            A structured `StepResult` object.
        """
        # Safely get the top-level 'observation' object. It could be a dict or None.
        obs_data = payload.get("observation")

        # Check if the object is a dictionary and contains the nested 'observation' key.
        # This identifies the double-nested structure from the /step endpoint.
        if isinstance(obs_data, dict) and "observation" in obs_data:
            # If so, go one level deeper to get the actual data payload.
            actual_obs_data = obs_data.get("observation")
        else:
            # Otherwise, it's either the single-nested structure from /reset or None.
            actual_obs_data = obs_data if isinstance(obs_data, dict) else {}

        # To prevent crashes, ensure `actual_obs_data` is a dictionary before
        # we try to access keys from it. If it was None, it becomes an empty dict.
        if not isinstance(actual_obs_data, dict):
            actual_obs_data = {}

        # Construct the DIPGObservation object safely.
        # Using .get() with a default value ("") prevents a KeyError if 'context' or
        # 'question' are missing from the payload, ensuring the client never crashes.
        obs = DIPGObservation(
            context=actual_obs_data.get("context", ""),
            question=actual_obs_data.get("question", ""),
        )

        # Assemble and return the final, structured StepResult.
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DIPGState:
        """
        Parses the JSON payload from the server's `/state` endpoint into a `DIPGState` object.

        Args:
            payload: The raw dictionary parsed from the server's JSON response.

        Returns:
            A structured `DIPGState` object.
        """
        return DIPGState(**payload)
