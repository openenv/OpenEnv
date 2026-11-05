# Hugging Face Spaces uses the root Dockerfile. Keep identical to server/Dockerfile.
FROM python:3.13-slim@sha256:aaf08c5490d52cd92bc27142a5215b1e47355a5eaf9adfad2a0fbc022d0b5490
WORKDIR /app/env
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock
COPY . /app/env
ENV PYTHONUNBUFFERED=1 PORT=8000 ENABLE_WEB_INTERFACE=true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
