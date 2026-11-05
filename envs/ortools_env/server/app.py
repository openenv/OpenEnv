# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the OR-Tools Environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

try:
    from ..models import ORToolsState
    from .ortools_environment import ORToolsEnvironment
except ImportError:
    from models import ORToolsState
    from server.ortools_environment import ORToolsEnvironment

app = create_app(
    ORToolsEnvironment,
    CallToolAction,
    CallToolObservation,
    reset_observation_cls=Observation,
    state_cls=ORToolsState,
    env_name="ortools_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "16")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
