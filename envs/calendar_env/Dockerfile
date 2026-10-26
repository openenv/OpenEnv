# Use Python 3.11 slim image as base
FROM python:3.11-slim

ENV API_PORT=8004

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Place seed database where the server expects it
COPY mcp_databases /app/mcp_databases

# Expose port 8004
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8004/health')" || exit 1

# Run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8004"]
