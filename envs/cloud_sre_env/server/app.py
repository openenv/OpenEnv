"""
FastAPI application for the Cloud SRE Environment.
Uses the OpenEnv create_fastapi_app helper.
"""

from openenv.core.env_server import create_fastapi_app

from ..models import SREAction, SREObservation, SREState
from .cloud_sre_environment import CloudSREEnvironment

# The OpenEnv server creates one environment instance per client session.
app = create_fastapi_app(
    CloudSREEnvironment, SREAction, SREObservation, state_cls=SREState
)


def main():
    """Run the Cloud SRE server from the command line."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Cloud SRE server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
