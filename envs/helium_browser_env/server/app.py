"""FastAPI application for the Helium browser environment."""

import os
from contextlib import suppress

from openenv.core.env_server import create_app

from ..models import BrowserAction, BrowserObservation, BrowserState
from .helium_browser_environment import BrowserEnvironment

app = create_app(
    BrowserEnvironment,
    BrowserAction,
    BrowserObservation,
    state_cls=BrowserState,
    env_name="helium_browser_env",
    max_concurrent_envs=1,
)


def main():
    """Run the server over TCP or the Unix socket requested by HF Sandbox."""
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
