# Copyright 2026 Kayam
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for the Corner Shop environment.

The HTTP /reset and /step endpoints are stateless (a fresh simulator per request), so a
full episode needs the WebSocket session that CornerShopEnv (client.py) opens.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os

from openenv.core.env_server.http_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import CornerShopAction, CornerShopObservation, CornerShopState
    from .corner_shop_environment import CornerShopEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import CornerShopAction, CornerShopObservation, CornerShopState
    from server.corner_shop_environment import CornerShopEnvironment

# Pass the class (factory), not an instance: each WebSocket session owns its simulator.
app = create_app(
    CornerShopEnvironment,
    CornerShopAction,
    CornerShopObservation,
    env_name="corner_shop_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "16")),
    state_cls=CornerShopState,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
