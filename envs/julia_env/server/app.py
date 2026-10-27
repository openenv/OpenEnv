# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Julia Environment with concurrent execution support.

This module creates an HTTP server that exposes the JuliaCodeActEnv
over HTTP and WebSocket endpoints with optimized async execution for handling
multiple concurrent requests efficiently.

Features:
- WebSocket support for persistent sessions (required by OpenEnv clients)
- Julia Process Pool for 50-100x speedup on repeated executions
- Automatic error recovery and retry logic
- Comprehensive logging to file and console

Environment Variables:
- JULIA_MAX_WORKERS: Number of concurrent Julia executions (default: 8)
- JULIA_EXECUTION_TIMEOUT: Timeout in seconds (default: 120)
- JULIA_LOG_FILE: Log file path (default: /tmp/julia_env.log)
- JULIA_LOG_LEVEL: Log level (default: INFO)
- ENABLE_WEB_INTERFACE: Enable web interface (default: false)

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    python -m server.app
"""

import atexit
import logging
import os
import sys
import tempfile
import time as time_module
from logging.handlers import RotatingFileHandler

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app

    from ..models import JuliaAction, JuliaObservation, JuliaState
    from .julia_codeact_env import JuliaCodeActEnv
    from .julia_executor import JuliaExecutor
except ImportError:
    from models import JuliaAction, JuliaObservation, JuliaState

    # Standalone imports (when environment is standalone)
    from openenv.core.env_server.http_server import create_app
    from server.julia_codeact_env import JuliaCodeActEnv
    from server.julia_executor import JuliaExecutor

# Configuration from environment variables
MAX_WORKERS = int(os.getenv("JULIA_MAX_WORKERS", "8"))
EXECUTION_TIMEOUT = int(os.getenv("JULIA_EXECUTION_TIMEOUT", "120"))
LOG_FILE = os.getenv(
    "JULIA_LOG_FILE", os.path.join(tempfile.gettempdir(), "julia_env.log")
)
LOG_LEVEL = os.getenv("JULIA_LOG_LEVEL", "INFO")


def setup_logging():
    """Configure logging to both file and console with rotation."""
    # Configure both julia_env and openenv hierarchies to share handlers
    julia_logger = logging.getLogger("julia_env")
    openenv_logger = logging.getLogger("openenv")

    julia_logger.setLevel(getattr(logging, LOG_LEVEL))
    openenv_logger.setLevel(getattr(logging, LOG_LEVEL))

    # Prevent duplicate handlers
    if julia_logger.handlers:
        return julia_logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(process)d:%(thread)d] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation (10MB max, keep 5 backup files)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        julia_logger.addHandler(file_handler)
        openenv_logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log file {LOG_FILE}: {e}")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    julia_logger.addHandler(console_handler)
    openenv_logger.addHandler(console_handler)

    return julia_logger


# Setup logging
logger = setup_logging()


def initialize_julia_pool():
    """Initialize the Julia process pool for better performance."""
    port = int(os.getenv("PORT", "8000"))

    logger.info("=" * 80)
    logger.info("Starting Julia Environment Server")
    logger.info(f"Container Port: {port}")
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info(f"Execution Timeout: {EXECUTION_TIMEOUT}s")
    logger.info(f"Log File: {LOG_FILE}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info("=" * 80)

    # Enable Julia process pool for better performance
    pool_enabled = JuliaExecutor.enable_process_pool(
        size=MAX_WORKERS, timeout=EXECUTION_TIMEOUT
    )
    if pool_enabled:
        logger.info(f"Julia process pool enabled with {MAX_WORKERS} workers")
    else:
        logger.warning("Julia process pool not available, using subprocess mode")

    logger.info("Julia Environment Server initialized successfully")
    logger.info(
        f"Julia Environment Server started on port {port} with {MAX_WORKERS} concurrent workers"
    )


def shutdown_julia_pool():
    """Shutdown the Julia process pool."""
    logger.info("Shutting down Julia Environment Server...")
    JuliaExecutor.shutdown_pool()
    logger.info("Julia process pool shutdown complete")
    logger.info("Julia Environment Server shutdown complete")


# Initialize the pool at module load time
initialize_julia_pool()

# Register cleanup on exit
atexit.register(shutdown_julia_pool)


# Create the app using OpenEnv's create_app for WebSocket support
# Pass the class (factory) instead of an instance for session support
app = create_app(
    JuliaCodeActEnv,
    JuliaAction,
    JuliaObservation,
    env_name="julia_env",
    max_concurrent_envs=MAX_WORKERS,
    state_cls=JuliaState,
)


# Add request logging middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming HTTP requests for debugging."""

    async def dispatch(self, request: Request, call_next):
        start_time = time_module.time()
        path = request.url.path

        # Skip health check logging to reduce noise
        if path in ["/health", "/pool_status"]:
            return await call_next(request)

        logger.debug(f"HTTP Request: {request.method} {path}")

        response = await call_next(request)

        elapsed = time_module.time() - start_time
        logger.debug(
            f"HTTP Response: {request.method} {path} -> {response.status_code} ({elapsed:.2f}s)"
        )

        return response


app.add_middleware(RequestLoggingMiddleware)


# Add custom health endpoint with pool metrics
@app.get("/pool_status")
async def pool_status():
    """Get Julia process pool status."""
    return {
        "max_workers": MAX_WORKERS,
        "timeout": EXECUTION_TIMEOUT,
        "pool_enabled": JuliaExecutor.is_pool_enabled(),
        "pool_metrics": JuliaExecutor.get_pool_metrics(),
    }


def main():
    """Main entry point for running the server."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
