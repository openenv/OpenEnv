# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Reasoning Gym Environment.

This module creates an HTTP server that exposes the ReasoningGymEnvironment
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

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Import from local models.py (PYTHONPATH includes /app/env in Docker)
try:
    from models import ReasoningGymAction, ReasoningGymObservation
except ImportError:
    from ..models import ReasoningGymAction, ReasoningGymObservation
from .reasoning_gym_environment import ReasoningGymEnvironment


_max_concurrent_raw = os.getenv("MAX_CONCURRENT_ENVS", "8")
try:
    max_concurrent = int(_max_concurrent_raw)
except ValueError:
    raise ValueError(
        f"MAX_CONCURRENT_ENVS must be an integer, got: {_max_concurrent_raw!r}"
    ) from None


def create_reasoning_gym_environment() -> ReasoningGymEnvironment:
    """Factory: fresh environment per WebSocket session.

    Required for `max_concurrent_envs > 1` so each session gets its own dataset
    iterator. `ReasoningGymEnvironment` already declares
    `SUPPORTS_CONCURRENT_SESSIONS = True`.
    """
    return ReasoningGymEnvironment()


app = create_app(
    create_reasoning_gym_environment,
    ReasoningGymAction,
    ReasoningGymObservation,
    env_name="reasoning_gym_env",
    max_concurrent_envs=max_concurrent,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m reasoning_gym.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn reasoning_gym.server.app:app --workers 4
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
