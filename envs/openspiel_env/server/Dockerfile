# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# OpenSpiel Environment Dockerfile
# =============================================================================
#
# Uses a pre-built OpenSpiel base image to avoid long build times (~30-60 min).
# The base image contains compiled OpenSpiel (C++ and Python bindings).
#
# DEFAULT (recommended for HuggingFace Spaces):
#   Uses pre-built image from GHCR - no C++ compilation needed
#
# BUILD YOUR OWN BASE IMAGE (if you need custom OpenSpiel configuration):
#   1. Build the base image first (takes ~30-60 min):
#      docker build -t openspiel-base:latest -f server/Dockerfile.openspiel-base .
#   2. Then build with your local base image:
#      docker build --build-arg OPENSPIEL_BASE_IMAGE=openspiel-base:latest -t openspiel-env .
#
# =============================================================================

# Default: use pre-built image from GHCR (skips C++ compilation)
ARG OPENSPIEL_BASE_IMAGE=ghcr.io/huggingface/openenv-openspiel-base:latest
FROM ${OPENSPIEL_BASE_IMAGE}

WORKDIR /app

# Install git (needed for pip install from git repos in pyproject.toml)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment code (context is the environment directory)
COPY . /app/env

# Install Python dependencies from pyproject.toml
WORKDIR /app/env
RUN pip3 install --no-cache-dir .

WORKDIR /app

# Copy README for web interface documentation
COPY README.md /app/README.md

# Python path configuration
# - /repo and /repo/build/python: OpenSpiel paths from base image
# - /app/env: Environment code
ENV PYTHONPATH=/repo:/repo/build/python:/app/env

# OpenSpiel-specific environment variables (can be overridden at runtime)
ENV OPENSPIEL_GAME=catch
ENV OPENSPIEL_AGENT_PLAYER=0
ENV OPENSPIEL_OPPONENT_POLICY=random

# Uvicorn configuration
ENV WORKERS=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run the FastAPI server
CMD uvicorn server.app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120 --workers ${WORKERS}
