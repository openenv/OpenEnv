# SPDX-License-Identifier: BSD-3-Clause

#
# OpenEnv Base Image
#
# This is the standard base image for all OpenEnv environment servers.
# It includes the minimal dependencies needed to run HTTP environment servers
# and uv for fast dependency management.
#
# Build from repo root: docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
# Tag:   docker tag openenv-base:latest openenv-base:0.2.0
#

FROM ghcr.io/astral-sh/uv:0.5.27-python3.11-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Copy core pyproject.toml and lockfile for dependency installation
COPY pyproject.toml uv.lock* ./

# Install core dependencies using uv with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Final runtime stage
FROM python:3.11-slim

# Set metadata
LABEL maintainer="OpenEnv Team"
LABEL description="Base image for OpenEnv based environment servers with uv"
LABEL version="0.2.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from builder
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy console scripts installed by pip (uvicorn, fastapi, etc.)
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/fastapi /usr/local/bin/

# Set working directory
WORKDIR /app

# Default environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Default expose port (can be overridden)
EXPOSE 8000

# Note: CMD should be specified in child Dockerfiles
