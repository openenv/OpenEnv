# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the repository root.

FROM python:3.13-slim@sha256:aaf08c5490d52cd92bc27142a5215b1e47355a5eaf9adfad2a0fbc022d0b5490 AS builder
WORKDIR /app/env
RUN pip install --no-cache-dir uv==0.9.3
COPY pyproject.toml uv.lock README.md THIRD_PARTY_NOTICES.md ./
COPY server/data/LICENSE-BurstGPT.txt server/data/LICENSE-BurstGPT.txt
RUN uv sync --frozen --no-dev --no-install-project --no-editable
COPY . .
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim@sha256:aaf08c5490d52cd92bc27142a5215b1e47355a5eaf9adfad2a0fbc022d0b5490
WORKDIR /app/env
COPY --from=builder /app/env /app/env
ENV PATH="/app/env/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENABLE_WEB_INTERFACE=true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
