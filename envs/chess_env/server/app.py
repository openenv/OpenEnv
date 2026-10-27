# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Chess Environment."""

import os

from openenv.core.env_server import create_app

from ..models import ChessAction, ChessObservation, ChessState
from .chess_environment import ChessEnvironment

# Create the FastAPI app
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(
    ChessEnvironment,
    ChessAction,
    ChessObservation,
    env_name="chess_env",
    state_cls=ChessState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
