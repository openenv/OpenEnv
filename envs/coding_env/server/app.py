# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Coding Environment.

This module creates an HTTP server that exposes the PythonCodeActEnv
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.coding_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.coding_env.server.app
"""

import os
from contextlib import suppress

from coding_env.models import CodeAction, CodeObservation, CodeState
from coding_env.server.python_codeact_env import PythonCodeActEnv
from coding_env.server.python_executor import set_non_dumpable
from openenv.core.env_server import create_app

# Model code runs in worker processes that may share this server's uid, so keep
# them from reading this process's memory and environment through /proc.
set_non_dumpable()

# Create the app with web interface and README integration
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(
    PythonCodeActEnv,
    CodeAction,
    CodeObservation,
    env_name="coding_env",
    state_cls=CodeState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    """Main entry point for running the server."""
    import uvicorn

    port = int(os.environ.get("SBX_SERVICE_PORT", "8000"))
    if proxy_dir := os.environ.get("SBX_PROXY_DIR"):
        socket_path = os.path.join(proxy_dir, f"{port}.sock")
        os.makedirs(proxy_dir, exist_ok=True)
        with suppress(FileNotFoundError):
            os.unlink(socket_path)
        uvicorn.run(app, uds=socket_path)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
