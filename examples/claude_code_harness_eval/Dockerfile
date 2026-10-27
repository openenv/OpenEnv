# Claude Code as a τ²-bench agent, served in production mode on `WS /harness`.
# Claude Code runs inside the container, next to the environment it plays in.
# Build from the repository root:
#   docker build -t claude-code-tau2 -f examples/claude_code_harness_eval/Dockerfile .

# Claude Code needs Node 22, newer than Debian's.
FROM node:22-bookworm-slim AS node

# τ²-bench requires Python 3.12+.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g @anthropic-ai/claude-code@2.1.291

# git is needed to install τ²-bench, which is pinned to a GitHub commit.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY envs/tau2_env /app/tau2_env
WORKDIR /app/tau2_env
# tau2_env's version rarely changes, so reinstall it rather than reuse a cached wheel of older code.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --reinstall-package openenv-tau2-env

# τ²-bench's tasks, databases and policies, at the commit tau2_env pins.
RUN git clone --filter=blob:none --no-checkout https://github.com/sierra-research/tau2-bench /opt/tau2-bench \
    && cd /opt/tau2-bench \
    && git sparse-checkout set --no-cone data/tau2/domains data/tau2/user_simulator \
    && git checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3 \
    && rm -rf .git
ENV TAU2_DATA_DIR="/opt/tau2-bench/data"

COPY examples/claude_code_harness_eval /app/recipe
ENV PATH="/app/tau2_env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/recipe"

# Pass HF_TOKEN (the simulated customer) and ANTHROPIC_API_KEY or
# CLAUDE_CODE_OAUTH_TOKEN (Claude Code) with `docker run -e`.
RUN useradd --create-home agent
USER agent
WORKDIR /app/recipe
EXPOSE 8000
# Arguments after the image name go to serve.py, e.g. `--domain retail --task-id 5`.
ENTRYPOINT ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
