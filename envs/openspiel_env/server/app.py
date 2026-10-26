# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the OpenSpiel Environment.

This module creates an HTTP server that exposes OpenSpiel games
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server

Environment variables:
    OPENSPIEL_GAME: Game name to serve (default: "catch")
    OPENSPIEL_AGENT_PLAYER: Agent player ID (default: 0)
    OPENSPIEL_OPPONENT_POLICY: Opponent policy (default: "random")
    MAX_CONCURRENT_ENVS: Maximum concurrent WebSocket sessions (default: 8)
"""

import os

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app

    from ..models import OpenSpielAction, OpenSpielObservation, OpenSpielState
    from .openspiel_environment import OpenSpielEnvironment
except ImportError:
    from models import OpenSpielAction, OpenSpielObservation, OpenSpielState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from server.openspiel_environment import OpenSpielEnvironment

# Get game configuration from environment variables
game_name = os.getenv("OPENSPIEL_GAME", "catch")
agent_player = int(os.getenv("OPENSPIEL_AGENT_PLAYER", "0"))
opponent_policy = os.getenv("OPENSPIEL_OPPONENT_POLICY", "random")
max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))


# Factory function to create OpenSpielEnvironment instances
def create_openspiel_environment():
    """Factory function that creates OpenSpielEnvironment with config."""
    return OpenSpielEnvironment(
        game_name=game_name,
        agent_player=agent_player,
        opponent_policy=opponent_policy,
    )


# Create the FastAPI app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_openspiel_environment,
    OpenSpielAction,
    OpenSpielObservation,
    env_name="openspiel_env",
    max_concurrent_envs=max_concurrent,
    state_cls=OpenSpielState,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m openspiel_env.server.app
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
