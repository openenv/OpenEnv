# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Maze Env Environment.

This module creates an HTTP server that exposes the MazeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os

from openenv.core.env_server.http_server import create_app

# Support both in-repo and standalone imports
try:
    from ..models import MazeAction, MazeObservation, MazeState
    from .maze_env_environment import MazeEnvironment
except ImportError:
    from models import MazeAction, MazeObservation, MazeState
    from server.maze_env_environment import MazeEnvironment


# Create the app with web interface and README integration
app = create_app(
    MazeEnvironment,
    MazeAction,
    MazeObservation,
    env_name="maze_env",
    state_cls=MazeState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m maze_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn maze_env.server.app:app --workers 4
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
