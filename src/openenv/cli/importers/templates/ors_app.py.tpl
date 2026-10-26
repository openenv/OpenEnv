from __future__ import annotations

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .${env_name}_environment import ${class_name_prefix}Environment


app = create_app(
    ${class_name_prefix}Environment,
    CallToolAction,
    CallToolObservation,
    env_name="${env_name}",
    max_concurrent_envs=1,
)


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
