# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# OpenClaw Environment Dockerfile
# Provides an isolated execution environment for training agents on
# OpenClaw's agentic tool capabilities.

ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install system dependencies for shell execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    jq \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY envs/openclaw_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy environment code
COPY src/openenv/core/ /app/src/openenv/core/
COPY envs/openclaw_env/ /app/envs/openclaw_env/

# Create workspace directory
RUN mkdir -p /app/workspace && chmod 755 /app/workspace

# Set workspace environment variable
ENV OPENCLAW_WORKSPACE=/app/workspace

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "envs.openclaw_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
