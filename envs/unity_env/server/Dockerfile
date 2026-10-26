# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build for Unity ML-Agents environment
# Uses pip for package installation (no virtual environment)
# Note: Using Python 3.10.12 specifically because ml-agents requires >=3.10.1,<=3.10.12
# Note: Unity binaries are x86_64 only, so we force linux/amd64 platform

FROM --platform=linux/amd64 python:3.10.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Install dependencies using pip
# Note: mlagents packages are installed from git source via pyproject.toml
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

# Final runtime stage
FROM --platform=linux/amd64 python:3.10.12-slim

WORKDIR /app

# Install runtime dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the environment code
COPY . /app/env

# Create cache directory for Unity binaries
RUN mkdir -p /root/.mlagents-cache

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV UNITY_NO_GRAPHICS=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Note: Longer start period (60s) because Unity environment download may take time on first run

# Run the FastAPI server
# Note: workers=1 because Unity environments are not thread-safe
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
