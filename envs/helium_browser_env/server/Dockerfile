FROM python:3.12-slim-bookworm

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    chromium \
    chromium-driver \
    fonts-liberation \
    git \
    xdotool \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/env

RUN pip install --no-cache-dir uv \
    && if [ -f uv.lock ]; then \
        uv sync --frozen --no-dev --no-editable; \
    else \
        uv sync --no-dev --no-editable; \
    fi

RUN useradd --create-home --uid 1000 browser \
    && chown -R browser:browser /app/env

ENV PATH="/app/env/.venv/bin:$PATH" \
    BROWSER_SANDBOX=hf \
    BROWSER_MAX_STEPS=20 \
    BROWSER_ACTION_WAIT_SECONDS=1.5 \
    BROWSER_EXPLICIT_WAIT_SECONDS=3 \
    PYTHONUNBUFFERED=1

USER browser

EXPOSE 8000

CMD ["server"]
