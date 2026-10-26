# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Application entry point for the Grid World Environment.

This module exposes the GridWorldEnvironment via the OpenEnv WebSocket API.
"""

import os

# Import the correct app creation function from the core library
try:
    from openenv.core.env_server import create_app
except ImportError:
    from core.env_server import create_app

# Import our models and environment classes using relative paths
from ..models import GridWorldAction, GridWorldObservation
from .grid_world_environment import GridWorldEnvironment

# Create single environment instance
# This is reused for all HTTP requests.
# env = GridWorldEnvironment()

# Create the FastAPI app
app = create_app(
    GridWorldEnvironment,
    GridWorldAction,
    GridWorldObservation,
    env_name="grid_world_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


# --- 3. Add Entry Point ---
def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
