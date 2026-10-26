# envs/finqa_env/server/app.py
"""
FastAPI server for the FinQA environment.

Environment Variables:
    FINQA_DATA_PATH: Path to data directory (default: /app/env/data)
    FINQA_MAX_STEPS: Maximum tool calls per episode (default: 50)
    FINQA_TASK: Task name (default: finqa)
"""

import inspect
import json
import os
from typing import Any, Dict

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation
from pydantic import field_validator

from ..models import FinQAState
from .finqa_environment import FinQAEnvironment

DATA_PATH = os.environ.get("FINQA_DATA_PATH", "/app/env/data")
MAX_STEPS = int(os.environ.get("FINQA_MAX_STEPS", "50"))
TASK = os.environ.get("FINQA_TASK", "finqa")


def _env_factory():
    """Create a new FinQAEnvironment instance for each session."""
    return FinQAEnvironment(
        data_path=DATA_PATH,
        max_steps=MAX_STEPS,
        task=TASK,
    )


class FinQACallToolAction(CallToolAction):
    """CallToolAction that accepts JSON strings for arguments (web UI sends strings)."""

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            return json.loads(v)
        return v


# Older openenv-core releases lack these model parameters.
_create_app_kwargs = (
    {"reset_observation_cls": Observation}
    if "reset_observation_cls" in inspect.signature(create_app).parameters
    else {}
)
if "state_cls" in inspect.signature(create_app).parameters:
    _create_app_kwargs["state_cls"] = FinQAState

app = create_app(
    _env_factory,
    FinQACallToolAction,
    CallToolObservation,
    **_create_app_kwargs,
    env_name="finqa_env",
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
