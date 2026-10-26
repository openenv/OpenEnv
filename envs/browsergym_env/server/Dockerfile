# Use the matching Playwright Python image for browsergym-core 0.14.3.
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set working directory
WORKDIR /app/env

# Install only the extra packages this env still needs on top of the Playwright image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-unifont \
    fonts-noto-color-emoji \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files first (for better caching)
# Build context should be envs/browsergym_env/ (not server/ or repo root)
COPY . .

# Make start script executable
RUN chmod +x /app/env/server/start.sh

RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache .

# Ensure the installed Playwright package has a matching Chromium runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright
RUN python -m playwright install chromium && chmod -R a+rx /usr/local/share/ms-playwright

# Install MiniWoB++ from a pinned snapshot to avoid flaky Hub-time git clones.
ARG MINIWOB_COMMIT=7fd85d71a4b60325c6585396ec4f48377d049838
RUN mkdir -p /usr/local/share/miniwob-plusplus && \
    curl -L "https://github.com/Farama-Foundation/miniwob-plusplus/archive/${MINIWOB_COMMIT}.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local/share/miniwob-plusplus && \
    chmod -R a+rX /usr/local/share/miniwob-plusplus

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BROWSERGYM_BENCHMARK=miniwob
ENV BROWSERGYM_TASK_NAME="click-test"
ENV BROWSERGYM_HEADLESS=true
ENV BROWSERGYM_VIEWPORT_WIDTH=1280
ENV BROWSERGYM_VIEWPORT_HEIGHT=720
ENV BROWSERGYM_TIMEOUT=10000
ENV BROWSERGYM_PORT=8000
ENV MINIWOB_HTML_DIR=/usr/local/share/miniwob-plusplus/miniwob/html
ENV MINIWOB_HTTP_PORT=8888
ENV MINIWOB_URL=http://127.0.0.1:8888/miniwob/
ENV ENABLE_WEB_INTERFACE=true

# For WebArena tasks, these should be set by the user when running the container:
# ENV SHOPPING=
# ENV SHOPPING_ADMIN=
# ENV REDDIT=
# ENV GITLAB=
# ENV MAP=
# ENV WIKIPEDIA=
# ENV HOMEPAGE=

# Expose ports
EXPOSE 8000
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server using the start script
CMD ["/app/env/server/start.sh"]
