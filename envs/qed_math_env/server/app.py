# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the QED Math Environment.

Exposes QEDMathEnvironment over HTTP and WebSocket endpoints,
compatible with MCPToolClient.

Examples:

```bash
# Development (with auto-reload):
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Or via uv:
uv run --project . server
```
"""

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .qed_math_environment import QEDMathEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.qed_math_environment import QEDMathEnvironment

# Pass the class (not an instance) to enable per-session environment instances.
# Use MCP types since this is a pure MCP environment.
app = create_app(
    QEDMathEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="qed_math_env",
)


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Lightweight service health endpoint for basic orchestration checks."""
    return {"status": "ok"}


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
