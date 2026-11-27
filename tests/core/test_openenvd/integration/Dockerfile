# Image for the openenvd integration test: a unit that can nest containers.
# Run privileged on Docker Desktop or a Linux host; see run.sh.
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends crun \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/openenv
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . pytest pytest-asyncio
COPY envs/echo_env /opt/envs/echo_env
ENV PYTHONPATH=/opt/envs
COPY tests/core/test_openenvd/integration/unit /opt/unit
COPY tests/core/test_openenvd/integration/test_live_unit.py /opt/tests/test_live_unit.py
COPY tests/core/test_openenvd/test_cgroups.py /opt/tests/test_cgroups.py
RUN mkdir -p /workspace /assets /inputs /services /run/openenvd /var/lib/sim /out \
    && chmod 0755 /opt/unit/observers/*.py && chown -R 1000:1000 /opt/unit/assets && chmod -R go-rwx /opt/unit/assets
