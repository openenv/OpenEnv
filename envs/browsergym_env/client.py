"""Client for the BrowserGym environment."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import BrowserGymAction, BrowserGymObservation, BrowserGymState


class BrowserGymEnv(
    EnvClient[BrowserGymAction, BrowserGymObservation, BrowserGymState]
):
    """Client for interacting with the BrowserGym environment.

    BrowserGym provides unified access to multiple web navigation benchmarks:
    - MiniWoB++: 100+ training tasks (no external infrastructure needed!)
    - WebArena: 812 evaluation tasks (requires backend setup)
    - VisualWebArena: Visual navigation tasks
    - WorkArena: Enterprise automation tasks

    Example usage for TRAINING (MiniWoB - works out of the box):
        ```python
        from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

        # Create environment for MiniWoB training task
        env = BrowserGymEnv.from_docker_image(
            "browsergym-env:latest",
            environment={
                "BROWSERGYM_BENCHMARK": "miniwob",
                "BROWSERGYM_TASK_NAME": "click-test",
            }
        )

        # Reset and get initial observation
        result = env.reset()
        print(f"Task: {result.observation.goal}")
        print(f"Page: {result.observation.text[:200]}")

        # Take actions
        action = BrowserGymAction(action_str="click('Submit button')")
        result = env.step(action)
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")

        env.close()
        ```

    Example usage for EVALUATION (WebArena - requires backend):
        ```python
        from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

        # Create environment for WebArena evaluation
        env = BrowserGymEnv.from_docker_image(
            "browsergym-env:latest",
            environment={
                "BROWSERGYM_BENCHMARK": "webarena",
                "BROWSERGYM_TASK_NAME": "0",  # Task 0
                # WebArena backend URLs
                "SHOPPING": "http://your-server:7770",
                "GITLAB": "http://your-server:8023",
                # ... other URLs
            }
        )

        result = env.reset()
        # ... interact with environment
        env.close()
        ```

    Available benchmarks:
        - miniwob: MiniWoB++ tasks (training, no setup required)
        - webarena: WebArena tasks (evaluation, requires backend)
        - visualwebarena: Visual WebArena tasks (evaluation, requires backend)
        - workarena: WorkArena tasks (evaluation, requires backend)
    """

    def _step_payload(self, action: BrowserGymAction) -> Dict[str, Any]:
        """Convert a BrowserGymAction to the JSON payload for the server."""
        return {
            "action_str": action.action_str,
            "metadata": action.metadata,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[BrowserGymObservation]:
        """Parse the server response into a StepResult."""
        obs_data = payload.get("observation", {})

        observation = BrowserGymObservation(
            text=obs_data.get("text", ""),
            url=obs_data.get("url", ""),
            screenshot=obs_data.get("screenshot"),
            goal=obs_data.get("goal", ""),
            axtree_txt=obs_data.get("axtree_txt", ""),
            pruned_html=obs_data.get("pruned_html", ""),
            error=obs_data.get("error", ""),
            last_action_error=obs_data.get("last_action_error", False),
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

    def _parse_state(self, payload: Dict[str, Any]) -> BrowserGymState:
        """Parse the server state response into a BrowserGymState object."""
        return BrowserGymState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            benchmark=payload.get("benchmark", ""),
            task_name=payload.get("task_name", ""),
            task_id=payload.get("task_id"),
            goal=payload.get("goal", ""),
            current_url=payload.get("current_url", ""),
            max_steps=payload.get("max_steps"),
            cum_reward=payload.get("cum_reward", 0.0),
        )
