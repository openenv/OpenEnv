# server/app.py
import os

from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.web_interface import load_environment_metadata

from ..models import WildfireAction, WildfireObservation, WildfireState
from .wildfire_environment import WildfireEnvironment
from .wildfire_web_interface import get_wildfire_web_interface_html

W = int(os.getenv("WILDFIRE_WIDTH", "16"))
H = int(os.getenv("WILDFIRE_HEIGHT", "16"))


# Factory function to create WildfireEnvironment instances
def create_wildfire_environment():
    """Factory function that creates WildfireEnvironment with config."""
    return WildfireEnvironment(width=W, height=H)


# Check if web interface should be enabled for custom routes
enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")

# Create the app with web interface support
# create_app handles ENABLE_WEB_INTERFACE automatically
app = create_app(
    create_wildfire_environment,
    WildfireAction,
    WildfireObservation,
    env_name="wildfire_env",
    state_cls=WildfireState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)

# Override the default /web route with our custom wildfire interface
# This must be done AFTER create_app to ensure it overrides the default route
if enable_web:
    # Load metadata for custom wildfire interface
    env_instance = create_wildfire_environment()
    metadata = load_environment_metadata(env_instance, "wildfire_env")

    # Remove any existing /web GET route and add our custom one
    # FastAPI uses the first matching route, so we need to remove the default one first
    routes_to_remove = []
    for route in app.routes:
        # Check if this is a GET route for /web
        if hasattr(route, "path") and route.path == "/web":
            if hasattr(route, "methods") and "GET" in route.methods:
                routes_to_remove.append(route)
            elif (
                hasattr(route, "methods") and not route.methods
            ):  # Some route types don't have methods
                # Check if it's a GET route by inspecting the endpoint
                routes_to_remove.append(route)

    for route in routes_to_remove:
        app.routes.remove(route)

    # Add our custom wildfire interface route (overrides default /web)
    @app.get("/web", response_class=HTMLResponse)
    async def wildfire_web_interface():
        """Custom wildfire-specific web interface."""
        return get_wildfire_web_interface_html(metadata)


def main():
    """Main entry point for running the server."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
