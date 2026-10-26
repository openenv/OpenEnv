# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the dm_control Environment.

This module creates an HTTP server that exposes dm_control.suite environments
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

import os

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import DMControlAction, DMControlObservation, DMControlState
    from .dm_control_environment import DMControlEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app

    try:
        import sys
        from pathlib import Path

        _parent = str(Path(__file__).parent.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from models import DMControlAction, DMControlObservation, DMControlState
        from server.dm_control_environment import DMControlEnvironment
    except ImportError:
        try:
            from dm_control_env.models import (
                DMControlAction,
                DMControlObservation,
                DMControlState,
            )
            from dm_control_env.server.dm_control_environment import (
                DMControlEnvironment,
            )
        except ImportError:
            from envs.dm_control_env.models import (
                DMControlAction,
                DMControlObservation,
                DMControlState,
            )
            from envs.dm_control_env.server.dm_control_environment import (
                DMControlEnvironment,
            )

# Create the app with web interface
# Pass the class (factory) for concurrent session support
app = create_app(
    DMControlEnvironment,
    DMControlAction,
    DMControlObservation,
    env_name="dm_control_env",
    state_cls=DMControlState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.dm_control_env.server.app
        openenv serve dm_control_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
