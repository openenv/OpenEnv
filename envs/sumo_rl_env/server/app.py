# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for SUMO-RL environment server.

This module creates an HTTP server that exposes traffic signal control
via the OpenEnv API using SUMO (Simulation of Urban MObility).
"""

import os

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import SumoAction, SumoObservation, SumoState
    from .sumo_environment import SumoEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import SumoAction, SumoObservation, SumoState
    from server.sumo_environment import SumoEnvironment

# Get configuration from environment variables
net_file = os.getenv("SUMO_NET_FILE", "/app/nets/single-intersection.net.xml")
route_file = os.getenv("SUMO_ROUTE_FILE", "/app/nets/single-intersection.rou.xml")
num_seconds = int(os.getenv("SUMO_NUM_SECONDS", "20000"))
delta_time = int(os.getenv("SUMO_DELTA_TIME", "5"))
yellow_time = int(os.getenv("SUMO_YELLOW_TIME", "2"))
min_green = int(os.getenv("SUMO_MIN_GREEN", "5"))
max_green = int(os.getenv("SUMO_MAX_GREEN", "50"))
reward_fn = os.getenv("SUMO_REWARD_FN", "diff-waiting-time")
sumo_seed = int(os.getenv("SUMO_SEED", "42"))


# Factory function to create SumoEnvironment instances
def create_sumo_environment():
    """Factory function that creates SumoEnvironment with config."""
    return SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        num_seconds=num_seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        max_green=max_green,
        reward_fn=reward_fn,
        sumo_seed=sumo_seed,
    )


# Create FastAPI app
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_sumo_environment,
    SumoAction,
    SumoObservation,
    env_name="sumo_rl_env",
    state_cls=SumoState,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
