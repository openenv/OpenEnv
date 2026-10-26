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

# Environment variables with defaults
ENV FINQA_DATA_PATH="/app/env/data"
ENV FINQA_MAX_STEPS="50"
ENV FINQA_TASK="finqa"

# Download data from HuggingFace at build time using Python API
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('snorkelai/finqa-data', repo_type='dataset', local_dir='/app/env/data')"

# Health check using Python (more portable than curl/wget)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server using the installed package
CMD ["uvicorn", "finqa_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
