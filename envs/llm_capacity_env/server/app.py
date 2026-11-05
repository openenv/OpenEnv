"""Standalone OpenEnv HTTP/WebSocket and MCP server."""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from .capacity_environment import CapacityEnvironment

# Default server runs emit JSONL trajectories. Direct library evaluators can opt in separately.
os.environ.setdefault("CAPACITY_TRACE_DIR", "outputs/trajectories")

app = create_app(
    CapacityEnvironment,
    CallToolAction,
    CallToolObservation,
    reset_observation_cls=Observation,
    env_name="llm_capacity_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
