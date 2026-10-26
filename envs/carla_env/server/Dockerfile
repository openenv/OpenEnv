# CARLA Environment - Real Mode with Full CARLA Server (Standalone)
# Complete self-contained deployment with CARLA 0.10.0 server
#
# Requirements:
#   - GPU: NVIDIA A10G or larger. CARLA 0.10 may crash on start on a T4.
#   - RAM: 16GB+
#   - Disk: ~15GB
#   - HF Space Hardware: GPU A10G or better
#
# Cost on HF Spaces:
#   - A10G GPU: ~$1.10/hour (~$792/month if running 24/7)
#
# Build time: 30-60 minutes (downloads ~10GB of CARLA)

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # Python build dependencies (for cffi, cryptography, etc.)
    build-essential \
    libffi-dev \
    libssl-dev \
    # CARLA dependencies
    wget \
    curl \
    ca-certificates \
    git \
    libomp5 \
    libpng16-16 \
    libvulkan1 \
    # For offscreen rendering (OpenGL)
    xvfb \
    libsdl2-2.0-0 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    # XDG utilities (required by CARLA for user directories)
    xdg-user-dirs \
    # pkill, to restart CARLA if it fails to start
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /opt

# Download and extract CARLA 0.10.0 (UE5.5, released Dec 2024)
# Official release: https://github.com/carla-simulator/carla/releases/tag/0.10.0
# Extracts to: Carla-0.10.0-Linux-Shipping/
RUN wget -q https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/Carla-0.10.0-Linux-Shipping.tar.gz -O /tmp/CARLA_0.10.0.tar.gz \
    && mkdir -p /opt/carla_temp \
    && tar -xzf /tmp/CARLA_0.10.0.tar.gz -C /opt/carla_temp \
    && rm /tmp/CARLA_0.10.0.tar.gz \
    # Move the extracted directory to /opt/carla
    && mv /opt/carla_temp/Carla-0.10.0-Linux-Shipping /opt/carla \
    && rm -rf /opt/carla_temp \
    # Find and make executable any .sh files
    && find /opt/carla -name "*.sh" -type f -exec chmod +x {} \;

# Create symbolic link for CARLA
# In CARLA 0.10.0 (UE5), the executable is CarlaUnreal.sh
RUN ln -s /opt/carla/CarlaUnreal.sh /usr/local/bin/CarlaUE4 \
    && chmod +x /opt/carla/CarlaUnreal.sh

# Set CARLA environment variables
ENV CARLA_ROOT=/opt/carla
ENV PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"

WORKDIR /app

# Upgrade pip and install cffi explicitly (fixes _cffi_backend error)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir cffi cryptography

# Install OpenEnv core from GitHub
RUN pip install --no-cache-dir git+https://github.com/huggingface/OpenEnv.git

# Copy and install environment dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install CARLA Python client (UE5 API, compatible with CARLA 0.9.x and 0.10.x)
RUN pip install --no-cache-dir carla-ue5-api==0.10.0

# Copy CARLA environment code
COPY . /app/carla_env/

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Create non-root user for CARLA (CARLA 0.10.0 refuses to run as root for security)
RUN useradd -m -u 1000 -s /bin/bash carla \
    && chown -R carla:carla /opt/carla /app \
    # Create XDG_RUNTIME_DIR for user carla (required by CARLA)\
    && mkdir -p /run/user/1000 \
    && chown carla:carla /run/user/1000 \
    && chmod 0700 /run/user/1000

# Environment variables for REAL mode
ENV CARLA_MODE=real
ENV CARLA_SCENARIO=trolley_saves
ENV CARLA_HOST=localhost
ENV CARLA_PORT=2000

# CARLA server settings for offscreen rendering
ENV SDL_VIDEODRIVER=offscreen
ENV DISPLAY=

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
EXPOSE 2000

# Startup script (runs CARLA as non-root user)
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚗 Starting CARLA Server 0.10.0 (UE5.5)..."\n\
\n\
# Create log directory\n\
mkdir -p /tmp/carla_logs\n\
chown carla:carla /tmp/carla_logs\n\
\n\
# Launch CARLA and wait until it answers (up to 5 min). Some starts crash.\n\
start_carla() {\n\
    su - carla -c "export XDG_RUNTIME_DIR=/run/user/1000 && cd /opt/carla && ./CarlaUnreal.sh -RenderOffScreen -opengl -quality-level=Low -carla-rpc-port=2000 -fps=20 > /tmp/carla_logs/carla.log 2>&1" &\n\
    CARLA_PID=$!\n\
    for i in {1..60}; do\n\
        sleep 5\n\
        kill -0 $CARLA_PID 2>/dev/null || return 1\n\
        python3 -c "import carla; c = carla.Client('\''localhost'\'', 2000); c.set_timeout(5.0); print('\''✅ CARLA ready:'\'', c.get_world().get_map().name)" 2>/dev/null && return 0\n\
    done\n\
    return 1\n\
}\n\
\n\
echo "⏳ Waiting for CARLA to initialize..."\n\
if ! start_carla; then\n\
    echo "⚠️ CARLA failed to start, retrying once"\n\
    tail -50 /tmp/carla_logs/carla.log\n\
    pkill -u carla || true\n\
    sleep 5\n\
    start_carla || {\n\
        echo "❌ CARLA server failed to start"\n\
        tail -50 /tmp/carla_logs/carla.log\n\
        exit 1\n\
    }\n\
fi\n\
\n\
# Start OpenEnv FastAPI server as non-root\n\
echo "🚀 Starting OpenEnv server..."\n\
cd /app\n\
exec su - carla -c "cd /app && CARLA_MODE=\"${CARLA_MODE}\" CARLA_SCENARIO=\"${CARLA_SCENARIO}\" CARLA_HOST=\"${CARLA_HOST}\" CARLA_PORT=\"${CARLA_PORT}\" ENABLE_WEB_INTERFACE=\"${ENABLE_WEB_INTERFACE}\" PYTHONPATH=/app:/opt/carla/PythonAPI/carla uvicorn carla_env.server.app:app --host 0.0.0.0 --port 8000"\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
