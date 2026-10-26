# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile works for both in-repo and standalone builds.

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo

# Install system dependencies for Julia
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install juliaup and Julia 1.10
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.10

# Add Julia to PATH
ENV PATH="/root/.juliaup/bin:${PATH}"

# Configure juliaup to prevent runtime network calls
# Set the default channel explicitly and mark juliaup as initialized
ENV JULIAUP_CHANNEL="1.10"
RUN juliaup default 1.10

# Verify Julia installation
RUN julia --version

# Precompile commonly used Julia packages (Test is built-in)
RUN julia -e 'using Test; println("Julia Test module ready")'

# Copy environment code (always at root of build context)
COPY . /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

WORKDIR /app/env

# Install dependencies using uv sync
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

# Install Julia runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Julia installation from builder
# juliaup stores the manager in ~/.juliaup and actual Julia versions in ~/.julia/juliaup
COPY --from=builder /root/.juliaup /root/.juliaup
COPY --from=builder /root/.julia /root/.julia

# Create a direct symlink to Julia binary, bypassing the juliaup wrapper
# This prevents juliaup from trying to check for updates at runtime
# The juliaup wrapper (at ~/.juliaup/bin/julia) triggers network calls;
# the actual Julia binary is in ~/.julia/juliaup/julia-*/bin/julia
RUN echo "Looking for Julia binary..." && \
    ls -la /root/.julia/juliaup/ && \
    JULIA_BIN=$(find /root/.julia/juliaup -type f -executable -name "julia" 2>/dev/null | head -1) && \
    if [ -z "$JULIA_BIN" ]; then \
        echo "ERROR: Could not find Julia binary. Contents of .julia/juliaup:" && \
        find /root/.julia -name "julia*" -ls && \
        exit 1; \
    fi && \
    ln -sf "$JULIA_BIN" /usr/local/bin/julia && \
    echo "Linked Julia from: $JULIA_BIN"

# Verify Julia works without network
RUN julia --version

ENV PATH="/usr/local/bin:${PATH}"

# Disable juliaup's version update checks (as fallback if juliaup wrapper is used)
ENV JULIAUP_CHANNEL="1.10"

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Julia process pool configuration (can be overridden at runtime)
ENV JULIA_MAX_WORKERS=8
ENV JULIA_EXECUTION_TIMEOUT=120
ENV JULIA_LOG_FILE=/tmp/julia_env.log
ENV JULIA_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
