# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TB2 Environment Client."""

from __future__ import annotations

from typing import Any


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import Tbench2Action, Tbench2Observation, Tbench2State
except ImportError:
    from models import Tbench2Action, Tbench2Observation, Tbench2State

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class Tbench2Env(EnvClient[Tbench2Action, Tbench2Observation, Tbench2State]):
    """HTTP client for the TB2 environment."""

    def _step_payload(self, action: Tbench2Action) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "command": action.command,
            "session_id": action.session_id,
            "block": action.block,
            "wait_seconds": action.wait_seconds,
            "file_path": action.file_path,
            "content": action.content,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Tbench2Observation]:
        obs_data = payload.get("observation", {})
        observation = Tbench2Observation(
            instruction=obs_data.get("instruction", ""),
            output=obs_data.get("output", ""),
            success=obs_data.get("success", True),
            error=obs_data.get("error", ""),
            task_id=obs_data.get("task_id", ""),
            task_path=obs_data.get("task_path", ""),
            session_id=obs_data.get("session_id"),
            action_type=obs_data.get("action_type", ""),
            info=obs_data.get("info", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> Tbench2State:
        return Tbench2State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_path=payload.get("task_path", ""),
            terminal_ready=payload.get("terminal_ready", False),
            last_action_type=payload.get("last_action_type", ""),
            last_command=payload.get("last_command", ""),
            last_output=payload.get("last_output", ""),
        )
