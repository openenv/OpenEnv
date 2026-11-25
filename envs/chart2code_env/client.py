# SPDX-License-Identifier: BSD-3-Clause

"""Client for the chart-to-code environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import Chart2CodeAction, Chart2CodeObservation, Chart2CodeState


class Chart2CodeEnv(
    EnvClient[Chart2CodeAction, Chart2CodeObservation, Chart2CodeState]
):
    """Connects to a running chart-to-code environment server.

    Examples:

    ```python
    with Chart2CodeEnv(base_url="http://localhost:8000") as env:
        observation = env.reset(seed=3).observation
        draft = env.step(Chart2CodeAction(code=code, mode="render")).observation
        result = env.step(Chart2CodeAction(code=code, mode="submit"))
        print(result.reward, result.observation.scores)
    ```
    """

    def _step_payload(self, action: Chart2CodeAction) -> Dict[str, Any]:
        """Convert an action into the JSON body of a step request."""
        return {"code": action.code, "mode": action.mode}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[Chart2CodeObservation]:
        """Parse a server response into a typed step result.

        The server hoists `reward` and `done` onto the response envelope, so they
        are read from there first.
        """
        data = payload.get("observation", {})
        observation = Chart2CodeObservation(
            **{
                key: value
                for key, value in data.items()
                if key in Chart2CodeObservation.model_fields
                and key not in ("reward", "done")
            },
            reward=payload.get("reward", data.get("reward")),
            done=bool(payload.get("done", data.get("done", False))),
        )
        return StepResult(
            observation=observation, reward=observation.reward, done=observation.done
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Chart2CodeState:
        """Parse a response from the state endpoint into a typed state."""
        return Chart2CodeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            renders_used=payload.get("renders_used", 0),
            submitted=payload.get("submitted", False),
        )
