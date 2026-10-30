# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the Harvest Rush Environment."""

try:
    # Installed-package context (e.g. import harvest_rush_env.server.app)
    from openenv.core.env_server.http_server import create_app

    from ..models import HarvestRushAction, HarvestRushObservation
    from .harvest_rush_environment import HarvestRushEnvironment
except ImportError:
    # Container runtime context (uvicorn server.app:app, PYTHONPATH=/app/env).
    from openenv.core.env_server.http_server import create_app

    from harvest_rush_env.models import HarvestRushAction, HarvestRushObservation
    from harvest_rush_env.server.harvest_rush_environment import HarvestRushEnvironment

app = create_app(
    HarvestRushEnvironment,
    HarvestRushAction,
    HarvestRushObservation,
    env_name="harvest_rush_env",
    max_concurrent_envs=8,  # sessions share one read-only example pool
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
