# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/env

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH="/app/env/src/core:/app/env/src:/app/env:$PYTHONPATH"

# Run the server
CMD ["python", "-m", "uvicorn", "snake_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
