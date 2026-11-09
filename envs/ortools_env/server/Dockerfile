# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local src/core)
# - Standalone environments (with openenv from pip)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already in the pyproject.toml dependencies
# For standalone builds, openenv will be installed from pip via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install git for building from git repos (build-time only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using uv sync
# First pass: install dependencies without the project (for better caching)
# Second pass: install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

# Model-written code runs as the unprivileged `sandbox` user with its own interpreters (see `sandbox.py`). The user
# cannot read /app, which holds the environment source and the server's packages, so it can call neither the
# reference solver nor the verifier; the interpreter with `ortools` (for `allow_ortools=True`) is only readable by
# the `ortools` group.
RUN useradd --system --no-create-home sandbox \
    && groupadd --system ortools \
    && python -m venv /opt/sandbox \
    && /opt/sandbox/bin/pip install --no-cache-dir "numpy>=1.26" "scipy>=1.11" \
    && python -m venv /opt/sandbox-ortools \
    && /opt/sandbox-ortools/bin/pip install --no-cache-dir "numpy>=1.26" "scipy>=1.11" "ortools>=9.12" \
    && chown -R root:ortools /opt/sandbox-ortools \
    && chmod -R o-rwx /opt/sandbox-ortools /app
ENV SANDBOX_USER=sandbox
ENV SANDBOX_PYTHON=/opt/sandbox/bin/python
ENV SANDBOX_ORTOOLS_PYTHON=/opt/sandbox-ortools/bin/python

# Health check using Python (more portable than curl/wget)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
# The module path is constructed to work with the /app/env structure
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
