# SPDX-License-Identifier: BSD-3-Clause

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=__ENV_NAME__

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Install dependencies using uv sync
# If uv.lock exists, use it; otherwise resolve on the fly
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

# Move the venv out of /app/env so it isn't nested inside the directory we
# copy wholesale below. Copying /app/env/.venv separately AND /app/env would
# otherwise ship the venv twice (once at /app/.venv, once at
# /app/env/.venv), roughly doubling the image size.
RUN mv /app/env/.venv /app/.venv

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy the environment code (no longer contains .venv, see builder stage)
COPY --from=builder /app/env /app/env

# Recreate a .venv symlink inside the env directory for tools (e.g. `uv run`,
# editors) that expect the virtual environment alongside the project, without
# storing a second physical copy of it.
RUN ln -s /app/.venv /app/env/.venv

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# The module path is constructed to work with the /app/env structure
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
