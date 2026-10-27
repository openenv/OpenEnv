# τ²-bench requires Python 3.12+, and openenv-base ships 3.11, so this image
# starts from uv's Python 3.12 image instead.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git is needed to install τ²-bench, which is pinned to a GitHub commit, and curl by the
# release deployment (scripts/prepare_hf_deployment.sh), which installs its own uv.
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# τ²-bench's tasks, databases and policies live in its repository, not in the package.
# Keep the commit in sync with the `tau2` pin in pyproject.toml.
RUN git clone --filter=blob:none --no-checkout https://github.com/sierra-research/tau2-bench /opt/tau2-bench \
    && cd /opt/tau2-bench \
    && git sparse-checkout set --no-cone data/tau2/domains data/tau2/user_simulator \
    && git checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3 \
    && rm -rf .git
ENV TAU2_DATA_DIR="/opt/tau2-bench/data"

COPY . /app/env
WORKDIR /app/env

# The uv cache is shared across builds, and this project's version rarely changes, so
# rebuild it every time; otherwise a cached wheel of older code gets installed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --reinstall-package openenv-tau2-env

ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# The simulated user runs on Hugging Face Inference Providers by default. On a Space,
# visitors sign in and use their own token, and API clients pass theirs to reset(). HF_TOKEN,
# or TAU2_USER_PROVIDER=openai|anthropic with that provider's key, is the fallback.
ENV ENABLE_WEB_INTERFACE=true
ENV TAU2_DOMAIN="airline"
ENV TAU2_SPLIT="test"
ENV TAU2_USER_PROVIDER="hf"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "tau2_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
