# τ-emotion-bench requires Python 3.12+, and openenv-base ships 3.11, so this image
# starts from uv's Python 3.12 image instead.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git is needed to install τ-emotion-bench, which is pinned to a GitHub commit, and curl by
# the release deployment (scripts/prepare_hf_deployment.sh), which installs its own uv.
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# τ-emotion-bench's tasks, databases and policies live in its repository, not in the
# package. Keep the commit in sync with the `tau-emotion` pin in pyproject.toml.
RUN git clone --filter=blob:none --no-checkout https://github.com/sahsaeedi/tau-emotion-bench /opt/tau-emotion-bench \
    && cd /opt/tau-emotion-bench \
    && git sparse-checkout set --no-cone data/tau_emotion/domains \
    && git checkout 287ebfce6181b4e141f778661b3455c2c3f03859 \
    && rm -rf .git
ENV TAU_EMOTION_DATA_DIR="/opt/tau-emotion-bench/data"

COPY . /app/env
WORKDIR /app/env

# The uv cache is shared across builds, and this project's version rarely changes, so
# rebuild it every time; otherwise a cached wheel of older code gets installed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --reinstall-package openenv-tau-emotion-env

ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# The simulated user runs on Hugging Face Inference Providers by default, with the
# benchmark's reference user model. API clients pass their token to reset(); HF_TOKEN,
# or TAU_EMOTION_USER_PROVIDER=openai|anthropic with that provider's key, is the fallback.
ENV ENABLE_WEB_INTERFACE=true
ENV TAU_EMOTION_DOMAIN="airline"
ENV TAU_EMOTION_SPLIT="test"
ENV TAU_EMOTION_USER_PROVIDER="hf"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "tau_emotion_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
