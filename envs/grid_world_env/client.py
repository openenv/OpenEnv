# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State
except ImportError:
    from core.client_types import StepResult
    from core.env_client import EnvClient
    from core.env_server.types import State


from .models import GridWorldAction, GridWorldObservation, MoveAction


class GridWorldEnv(EnvClient[GridWorldAction, GridWorldObservation, State]):
    """
    A WebSocket-based client for interacting with the GridWorld environment.

    This client inherits from EnvClient and is configured with the
    GridWorld Pydantic models for automatic (de)serialization.
    """

    def step_move(self, move: MoveAction) -> StepResult[GridWorldObservation]:
        """
        Helper method to send a simple move action.

        Args:
            move: The MoveAction enum (e.g., MoveAction.UP)
        """
        action_payload = GridWorldAction(action=move)
        # 'super().step' comes from the base HTTPEnvClient
        return super().step(action_payload)

    # --- REQUIRED ABSTRACT METHODS (The Missing Pieces) ---

    def _step_payload(self, action: GridWorldAction) -> dict:
        """Convert the Pydantic action model to a dictionary."""
        # Uses Pydantic v2 'model_dump'. If this fails, try 'action.dict()'
        return action.model_dump()

    def _parse_result(self, data: dict) -> StepResult[GridWorldObservation]:
        """Convert the raw dictionary response into a typed StepResult."""
        return StepResult(
            observation=GridWorldObservation(
                **data["observation"], reward=data["reward"], done=data["done"]
            ),
            reward=data["reward"],
            done=data["done"],
            metadata=data.get("metadata"),
        )

    def _parse_state(self, data: dict) -> State:
        """Convert the raw state dictionary into a State object."""
        return State(**data)
