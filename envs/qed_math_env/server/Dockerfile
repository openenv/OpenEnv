# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.
#
# Build from the env directory:
#   docker build -t qed-math-env:latest -f server/Dockerfile .
#
# Or from the repo root:
#   docker build -t qed-math-env:latest -f envs/qed_math_env/server/Dockerfile envs/qed_math_env
#
# Run:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY qed-math-env:latest

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv-core is already in the pyproject.toml dependencies
# For standalone builds, openenv-core will be installed from pip via pyproject.toml
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

# Unbuffered output for real-time logging
ENV PYTHONUNBUFFERED=1

# Health check using Python (more portable than curl/wget)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
# Use the installed package path (pyproject.toml maps server/ → qed_math_env.server)
# Set generous ping timeout so long-running LLM judge calls (submit_proof) don't
# trigger a keepalive ping timeout disconnect mid-grading (default is only 20s).
CMD ["uvicorn", "qed_math_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "120", "--ws-ping-timeout", "600"]
