#!/usr/bin/env python3

"""
FastAPI application for Git Environment.

This module creates an HTTP server for the Git environment that connects
to a shared external Gitea service for fast, isolated task resets.

Environment variables:
    GITEA_URL: URL of shared Gitea service (default: http://localhost:3000)
    GITEA_USERNAME: Gitea username (default: openenv)
    GITEA_PASSWORD: Gitea password (default: openenv)
    WORKSPACE_DIR: Workspace directory (optional, default: /workspace)

Usage:
    # Development (with auto-reload):
    uvicorn envs.git_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.git_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # With custom Gitea:
    GITEA_URL=http://my-gitea:3000 uvicorn envs.git_env.server.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import GitAction, GitObservation, GitState
    from .git_task_environment import GitTaskEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import GitAction, GitObservation, GitState
    from server.git_task_environment import GitTaskEnvironment

logger = logging.getLogger(__name__)

# Read configuration from environment variables
gitea_url = os.getenv("GITEA_URL", "http://localhost:3000")
gitea_username = os.getenv("GITEA_USERNAME", "openenv")
gitea_password = os.getenv("GITEA_PASSWORD", "openenv")
workspace_dir = os.getenv("WORKSPACE_DIR", "/workspace")

if "GITEA_USERNAME" not in os.environ or "GITEA_PASSWORD" not in os.environ:
    logger.warning(
        "Using default Gitea credentials. Set GITEA_USERNAME and GITEA_PASSWORD "
        "for non-local deployments."
    )


# Factory function to create GitTaskEnvironment instances
def create_git_environment():
    """Factory function that creates GitTaskEnvironment with config."""
    return GitTaskEnvironment(
        gitea_url=gitea_url,
        username=gitea_username,
        password=gitea_password,
        workspace_dir=workspace_dir,
    )


# Create the app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_git_environment,
    GitAction,
    GitObservation,
    env_name="git_env",
    state_cls=GitState,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
