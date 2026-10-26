# SPDX-License-Identifier: BSD-3-Clause

"""Client for the Pelican SVG environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import PelicanSvgAction, PelicanSvgObservation, PelicanSvgState


class PelicanSvgEnv(
    EnvClient[PelicanSvgAction, PelicanSvgObservation, PelicanSvgState]
):
    """Connects to a running Pelican SVG environment server.

    Examples:

    ```python
    with PelicanSvgEnv(base_url="http://localhost:8000") as env:
        observation = env.reset().observation
        reply = my_model(observation.prompt)
        result = env.step(PelicanSvgAction(response=reply))
        print(result.reward, result.observation.feedback)
    ```
    """

    def _step_payload(self, action: PelicanSvgAction) -> Dict[str, Any]:
        """Convert an action into the JSON body of a step request."""
        return {"response": action.response}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[PelicanSvgObservation]:
        """Parse a server response into a typed step result.

        The server hoists `reward` and `done` onto the response envelope and
        drops them from the serialised observation, so they are read from the
        envelope first and only then from the observation body.
        """
        data = payload.get("observation", {})
        reward = payload.get("reward", data.get("reward"))
        done = payload.get("done", data.get("done", False))
        observation = PelicanSvgObservation(
            prompt=data.get("prompt", ""),
            task_id=data.get("task_id", ""),
            subject=data.get("subject", ""),
            vehicle=data.get("vehicle", ""),
            expected_wheels=data.get("expected_wheels", 2),
            held_out=data.get("held_out", True),
            feedback=data.get("feedback", ""),
            gate_passed=data.get("gate_passed", False),
            structure_score=data.get("structure_score", 0.0),
            semantic_score=data.get("semantic_score", 0.0),
            judged=data.get("judged", False),
            violations=data.get("violations", []),
            breakdown=data.get("breakdown", {}),
            image_png_base64=data.get("image_png_base64"),
            done=bool(done),
            reward=reward,
            metadata=payload.get("metadata", data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PelicanSvgState:
        """Parse a response from the state endpoint into a typed state."""
        return PelicanSvgState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            submitted=payload.get("submitted", False),
        )
