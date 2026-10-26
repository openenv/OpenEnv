# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Atari Environment.

This module creates an HTTP server that exposes Atari games
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.atari_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.atari_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.atari_env.server.app

Environment variables:
    ATARI_GAME: Game name to serve (default: "pong")
    ATARI_OBS_TYPE: Observation type (default: "rgb")
    ATARI_FULL_ACTION_SPACE: Use full action space (default: "false")
    ATARI_MODE: Game mode (optional)
    ATARI_DIFFICULTY: Game difficulty (optional)
    ATARI_REPEAT_ACTION_PROB: Sticky action probability (default: "0.0")
    ATARI_FRAMESKIP: Frameskip (default: "4")
    MAX_CONCURRENT_ENVS: Maximum concurrent WebSocket sessions (default: "8")
"""

import os

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import AtariAction, AtariObservation, AtariState
    from .atari_environment import AtariEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import AtariAction, AtariObservation, AtariState
    from server.atari_environment import AtariEnvironment

# Get configuration from environment variables
game_name = os.getenv("ATARI_GAME", "pong")
obs_type = os.getenv("ATARI_OBS_TYPE", "rgb")
full_action_space = os.getenv("ATARI_FULL_ACTION_SPACE", "false").lower() == "true"
repeat_action_prob = float(os.getenv("ATARI_REPEAT_ACTION_PROB", "0.0"))
frameskip = int(os.getenv("ATARI_FRAMESKIP", "4"))

# Optional parameters
mode = os.getenv("ATARI_MODE")
difficulty = os.getenv("ATARI_DIFFICULTY")

# Convert to int if specified
mode = int(mode) if mode is not None else None
difficulty = int(difficulty) if difficulty is not None else None


# Factory function to create AtariEnvironment instances
def create_atari_environment():
    """Factory function that creates AtariEnvironment with config."""
    return AtariEnvironment(
        game_name=game_name,
        obs_type=obs_type,
        full_action_space=full_action_space,
        mode=mode,
        difficulty=difficulty,
        repeat_action_probability=repeat_action_prob,
        frameskip=frameskip,
    )


# Create the FastAPI app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_atari_environment,
    AtariAction,
    AtariObservation,
    env_name="atari_env",
    state_cls=AtariState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
