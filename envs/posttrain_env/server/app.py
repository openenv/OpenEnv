# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the post-training environment."""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from .posttrain_environment import PostTrainEnvironment

# The class is passed rather than an instance so each WebSocket session gets
# its own environment, broker and budget.
app = create_app(
    PostTrainEnvironment,
    CallToolAction,
    CallToolObservation,
    reset_observation_cls=Observation,
    env_name="posttrain_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
