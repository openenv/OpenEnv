# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI server for CARLA environment.

Exposes OpenEnv-compatible HTTP/WebSocket endpoints.
"""

import os

from openenv.core.env_server import create_app

from ..models import CarlaAction, CarlaObservation, CarlaState
from .carla_environment import CarlaEnvironment

# Configuration from environment variables
SCENARIO_NAME = os.getenv("CARLA_SCENARIO", "trolley_saves")
MODE = os.getenv("CARLA_MODE", "mock")  # "mock" or "real"
HOST = os.getenv("CARLA_HOST", "localhost")
PORT = int(os.getenv("CARLA_PORT", "2000"))


# Environment factory function
def create_environment():
    """Factory function to create CarlaEnvironment instances."""
    return CarlaEnvironment(
        scenario_name=SCENARIO_NAME,
        host=HOST,
        port=PORT,
        mode=MODE,
    )


# Create FastAPI app with environment factory
# Uses create_app which enables web interface when ENABLE_WEB_INTERFACE=true
app = create_app(
    create_environment,
    CarlaAction,
    CarlaObservation,
    env_name="carla_env",
    state_cls=CarlaState,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m carla_env.server.app
        openenv serve carla_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
