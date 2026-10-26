# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Unity ML-Agents Environment.

This module creates an HTTP server that exposes Unity ML-Agents environments
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Or run directly:
    uv run --project . server

Note: Unity environments are not thread-safe, so use workers=1.
"""

# Support multiple import scenarios
try:
    # In-repo imports (when running from OpenEnv repository root)
    from openenv.core.env_server.http_server import create_app

    from ..models import UnityAction, UnityObservation, UnityState
    from .unity_environment import UnityMLAgentsEnvironment
except ImportError:
    # openenv from pip
    from openenv.core.env_server.http_server import create_app

    try:
        # Direct execution from envs/unity_env/ directory
        import sys
        from pathlib import Path

        # Add parent directory to path for direct execution
        _parent = str(Path(__file__).parent.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from models import UnityAction, UnityObservation, UnityState
        from server.unity_environment import UnityMLAgentsEnvironment
    except ImportError:
        try:
            # Package installed as unity_env
            from unity_env.models import UnityAction, UnityObservation, UnityState
            from unity_env.server.unity_environment import UnityMLAgentsEnvironment
        except ImportError:
            # Running from OpenEnv root with envs prefix
            from envs.unity_env.models import UnityAction, UnityObservation, UnityState
            from envs.unity_env.server.unity_environment import UnityMLAgentsEnvironment

# Create the app with web interface
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(
    UnityMLAgentsEnvironment,
    UnityAction,
    UnityObservation,
    env_name="unity_env",
    state_cls=UnityState,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.unity_env.server.app
        openenv serve unity_env
    """
    import uvicorn

    # Note: workers=1 because Unity environments are not thread-safe
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
