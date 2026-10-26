ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# Harbor requires Python >= 3.12 while openenv-base ships 3.11, so `uv sync` downloads its own
# interpreter and the venv's bin/python becomes a symlink into uv's install dir. Pinning that dir
# (and creating it up front, so the COPY below cannot fail when uv reuses a system interpreter)
# is what lets the runtime stage carry the interpreter the venv actually points at. Without it the
# venv arrives with a dangling bin/python and the container dies with "not found".
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN mkdir -p /opt/uv-python

WORKDIR /app/env
COPY . /app/env
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-editable; else uv sync --no-editable; fi

FROM ${BASE_IMAGE}
COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Fail at build time rather than at startup if the interpreter did not survive the stage boundary.
RUN /app/.venv/bin/python -c "import sys; print('venv python', sys.version)"

ENV PATH="/app/.venv/bin:$PATH"
# `harbor push` bundles the working tree's openenv/ into /app/env when pushing from a source
# checkout; PYTHONPATH puts it ahead of the released wheel in site-packages, which has no
# `openenv.harbor` until this lands upstream.
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["sh", "-c", "cd /app/env && exec /app/.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"]
