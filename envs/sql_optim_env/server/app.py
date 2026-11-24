# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the SQL Query Optimization environment."""

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (running from the OpenEnv repository)
    from ..models import SQLOptimAction, SQLOptimObservation, SQLOptimState
    from .sql_optim_environment import SQLOptimEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (running via uvicorn server.app:app)
    from models import SQLOptimAction, SQLOptimObservation, SQLOptimState
    from server.sql_optim_environment import SQLOptimEnvironment

# Pass the class (factory) rather than an instance for WebSocket session support.
app = create_app(
    SQLOptimEnvironment,
    SQLOptimAction,
    SQLOptimObservation,
    env_name="sql_optim_env",
    state_cls=SQLOptimState,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
