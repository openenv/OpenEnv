# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Tbench2 Env Environment.

This module creates an HTTP server that exposes the Tbench2Environment
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

Environment Variables:
    TB2_MODE: Execution mode - "local" (default) or "docker"
    MAX_CONCURRENT_ENVS: Maximum concurrent WebSocket sessions (default: 8)
"""

import os


try:
    from openenv.core.env_server.http_server import create_app

    # In-repo imports
    from tbench2_env.models import Tbench2Action, Tbench2Observation, Tbench2State

    from .tbench2_env_environment import Tbench2DockerEnvironment, Tbench2Environment
except Exception as e:  # pragma: no cover
    from models import Tbench2Action, Tbench2Observation, Tbench2State

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from server.tbench2_env_environment import (
        Tbench2DockerEnvironment,
        Tbench2Environment,
    )

    _IMPORT_ERROR = e


# Determine which environment class to use based on TB2_MODE
_TB2_MODE = os.getenv("TB2_MODE", "local").lower()

if _TB2_MODE == "docker":
    _DEFAULT_ENVIRONMENT = Tbench2DockerEnvironment
    _ENV_SUFFIX = " (Docker mode)"
elif _TB2_MODE == "local":
    _DEFAULT_ENVIRONMENT = Tbench2Environment
    _ENV_SUFFIX = " (local mode)"
else:
    # No silent aliases: the old "auto" value claimed to auto-detect Docker
    # but always ran local mode, and a typo'd mode would quietly do the same.
    # Which class serves decides the isolation and scoring contract, so
    # refuse to guess.
    raise ValueError(f"Unknown TB2_MODE {_TB2_MODE!r}: expected 'local' or 'docker'.")


# Create the app with web interface and README integration
max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    _DEFAULT_ENVIRONMENT,
    Tbench2Action,
    Tbench2Observation,
    env_name="tbench2_env" + _ENV_SUFFIX,
    max_concurrent_envs=max_concurrent,
    state_cls=Tbench2State,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m tbench2_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn tbench2_env.server.app:app --workers 4
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
