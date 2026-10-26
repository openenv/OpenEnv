FROM python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
COPY wheelhouse /wheelhouse
COPY requirements.txt /requirements.txt
# Dependencies and the source wheel were acquired and hashed by reproduce.py.
RUN --network=none python -m pip install --no-index --find-links=/wheelhouse --require-hashes -r /requirements.txt \
    && rm -rf /wheelhouse /root/.cache
WORKDIR /app
COPY served_probe /app/served_probe
USER 65532:65532
EXPOSE 8000
CMD ["python", "-m", "served_probe.app"]
