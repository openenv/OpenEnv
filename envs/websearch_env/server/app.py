# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Websearch Env Environment.

This module creates an HTTP server that exposes the WebSearchEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import inspect

from models import WebSearchAction, WebSearchObservation
from openenv.core.env_server.http_server import create_app

from .web_search_environment import WebSearchEnvironment


def _create_websearch_app():
    """Build app across create_app variants that may expect a factory or an instance."""
    try:
        first_param = next(iter(inspect.signature(create_app).parameters.values()))
        annotation_text = str(first_param.annotation)
    except (StopIteration, TypeError, ValueError):
        annotation_text = "typing.Callable"

    expects_instance = (
        "Environment" in annotation_text and "Callable" not in annotation_text
    )
    env_arg = WebSearchEnvironment() if expects_instance else WebSearchEnvironment
    return create_app(
        env_arg,
        WebSearchAction,
        WebSearchObservation,
        env_name="websearch_env",
    )


# Create the app with web interface and README integration.
app = _create_websearch_app()


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m websearch_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn websearch_env.server.app:app --workers 4
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
