# Dockerfile for Coding Environment
# Build from repo root:
#   docker build -t coding-env:latest -f envs/coding_env/server/Dockerfile .

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy coding_env package
COPY envs/coding_env/ ./envs/coding_env/

# Install openenv first from PyPI, then coding_env
RUN pip install --no-cache-dir "openenv>=0.2.2" && \
    pip install --no-cache-dir ./envs/coding_env/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "coding_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
