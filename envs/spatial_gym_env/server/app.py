"""FastAPI app that serves Spatial-Gym over HTTP and WebSocket (OpenEnv).

Endpoints: POST /reset, POST /step, GET /state, GET /schema, WS /ws (one environment per
session), and the web UI at /web when ENABLE_WEB_INTERFACE=true.

Run locally:
    uv run --project . server
    uv run --project . uvicorn server.app:app --port 8001
"""

import os
from pathlib import Path

from openenv.core.env_server.http_server import create_app

try:
    from ..models import SpatialGymAction, SpatialGymObservation, SpatialGymState
    from .spatial_gym_environment import SpatialGymEnvironment
except ImportError:
    from models import SpatialGymAction, SpatialGymObservation, SpatialGymState
    from server.spatial_gym_environment import SpatialGymEnvironment


# The web UI shows this README; by default it only looks at /app/README.md.
os.environ.setdefault(
    "ENV_README_PATH", str(Path(__file__).resolve().parent.parent / "README.md")
)

app = create_app(
    SpatialGymEnvironment,
    SpatialGymAction,
    SpatialGymObservation,
    # Also the import name the web UI's Quick Start shows (`from spatial_gym_env import ...`).
    env_name="spatial_gym_env",
    state_cls=SpatialGymState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    """Entry point for `uv run --project . server` and `python -m spatial_gym_env.server.app`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
