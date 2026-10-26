# 2. Deploying an OpenEnv environment

This section covers deploying OpenEnv environments locally, on clusters, and on Hugging Face Spaces.

**Contents:**
- [Local Development with Uvicorn](#local-development-with-uvicorn)
- [Docker Deployment](#docker-deployment)
- [Hugging Face Spaces](#hugging-face-spaces)
- [Best Practices](#best-practices)

## HF Spaces are the infrastructure for OpenEnv environments

Every HF Space provides three things that OpenEnv environments need:

| Component | What it provides | How to access | Used as |
|-----------|------------------|---------------|-----------|
| **Server** | Running environment endpoint | `https://<username>-<space-name>.hf.space` | Agent and Public API |
| **Repository** | Installable Python package | `pip install git+https://huggingface.co/spaces/<username>-<space-name>` | Code and client |
| **Registry** | Docker container image | `docker pull registry.hf.space/<username>-<space-name>:latest` | Deployment |

This means a single Space deployment gives you all the components you need to use an environment in training.

### 1. Server: A running environment endpoint

When you deploy to HF Spaces, your environment runs as a server. The client connects via **WebSocket** (`/ws`) for a persistent session:

```python
from echo_env import EchoEnv, EchoAction

# Connect directly to the running Space (WebSocket under the hood)
# Async (recommended):
async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()
    result = await client.step(EchoAction(message="Hello"))

# Sync:
with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello"))
```

**Endpoints available:**

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/ws` | **WebSocket** | Persistent session (used by client) |
| `/health` | HTTP GET | Health check |
| `/reset` | HTTP POST | Reset environment (stateless) |
| `/step` | HTTP POST | Execute action (stateless) |
| `/state` | HTTP GET | Get current state |
| `/docs` | HTTP GET | OpenAPI documentation |
| `/web` | HTTP GET | Interactive web UI |

> **Note:** The Python client uses the `/ws` WebSocket endpoint by default. HTTP endpoints are available for debugging or stateless use cases.

**Example: Check if a Space is running**

```bash
curl https://openenv-echo-env.hf.space/health
# {"status": "healthy"}
```

### 2. Repository: Installable Python package

Every Space is a Git repository. OpenEnv environments include a `pyproject.toml`, making them pip-installable directly from the Space URL.

```bash
# Install client package from Space
pip install git+https://huggingface.co/spaces/openenv/echo-env
```

This installs:
- **Client class** (`EchoEnv`) — Handles HTTP/WebSocket communication
- **Models** (`EchoAction`, `EchoObservation`) — Typed action and observation classes
- **Utilities** — Any helper functions the environment provides

**After installation:**

```python
from envs.echo_env import EchoEnv, EchoAction, EchoObservation

# Now you have typed classes for the environment
action = EchoAction(message="Hello")
```

### 3. Registry: Docker container image

Every Docker-based Space has a container registry. You can pull and run the environment locally.

```bash
# Pull the image
docker pull registry.hf.space/openenv-echo-env:latest

# Run locally on port 8001
docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
```

**Find the registry URL for any Space:**

1. Go to the Space page (e.g., [openenv/echo-env](https://huggingface.co/spaces/openenv/echo-env))
2. Click **⋮** (three dots) → **"Run locally"**
3. Copy the `docker run` command

### Choosing an access method

| Method | Use when | Pros | Cons |
|--------|----------|------|------|
| **Server** | Quick testing, low volume | Zero setup | Network latency, rate limits |
| **Repository** | Need typed classes | Type safety, IDE support | Still need a server |
| **Docker** | Local dev, high throughput | Full control, no network | Requires Docker |

**Typical workflow:**

```python
import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    # Development: connect to remote Space
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        result = await client.reset()

    # Production: run locally for speed
    # docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
    async with EchoEnv(base_url="http://localhost:8001") as client:
        result = await client.reset()

    # Or let the client manage Docker for you
    client = await EchoEnv.from_env("openenv/echo-env")  # Auto-pulls and runs
    async with client:
        result = await client.reset()

asyncio.run(main())

# Sync usage:
with EchoEnv(base_url="http://localhost:8001") as client:
    result = client.reset()
```

> **Reference:** [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces) | [Environment Hub Collection](https://huggingface.co/collections/openenv/environment-hub)


## Local Development with Uvicorn

The fastest way to iterate on environment logic is running directly with Uvicorn.

## Clone and run the environment locally

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Install in editable mode
uv sync

# Start server
uv run server

# Run isolated from remote Space
uv run --isolated --project https://huggingface.co/spaces/burtenshaw/openenv-benchmark server
```

## Uvicorn directly in python

```bash
# Full control over uvicorn options
uvicorn benchmark.server.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"

# With reload for development
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --reload

# Multi-Worker Mode For better concurrency:
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

| Flag | Purpose |
|------|---------|
| `--reload` | Auto-restart on code changes |
| `--workers N` | Run N worker processes |
| `--log-level debug` | Verbose logging |

## Docker Deployment

Docker provides isolation and reproducibility for production use.

### Run the environment locally from the space

```bash
# Run the environment locally from the space
docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

### Build Image

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Using OpenEnv CLI (recommended)
openenv build -t openenv-benchmark:latest

# Or with Docker directly
docker build -t openenv-benchmark:latest -f server/Dockerfile .
```

### Run Container

```bash
# Basic run
docker run -d -p 8000:8000 my-env:latest

# With environment variables
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-env:latest

# Named container for easy management
docker run -d --name my-env -p 8000:8000 my-env:latest
```

### Connect from Python

```python
import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    # Async usage (recommended)
    async with EchoEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(EchoAction(message="Hello"))
        print(result.observation)

    # From Docker image
    client = await EchoEnv.from_docker_image("<local_docker_image>")
    async with client:
        result = await client.reset()
        print(result.observation)

asyncio.run(main())

# Sync usage
with EchoEnv(base_url="http://localhost:8000") as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello"))
    print(result.observation)
```

### Container Lifecycle

| Method | Container | WebSocket | On `close()` |
|--------|-----------|-----------|--------------|
| `from_hub(repo_id)` | Starts | Connects | Stops container |
| `from_hub(repo_id, use_docker=False)` | None (UV) | Connects | Stops UV server |
| `from_docker_image(image)` | Starts | Connects | Stops container |
| `MyEnv(base_url=...)` | None | Connects | Disconnects only |

Find Docker Commands for Any Space

1. Open the Space on HuggingFace Hub
2. Click **⋮ (three dots)** menu
3. Select **"Run locally"**
4. Copy the provided `docker run` command

## Deploy with CLI

```bash
cd my_env

# Deploy to your namespace
openenv push

# Deploy to specific repo
openenv push --repo-id username/my-env

# Deploy as private
openenv push --repo-id username/my-env --private
```

### Space Configuration

The `openenv.yaml` manifest controls Space settings:

```yaml
# openenv.yaml
name: my_env
version: "1.0.0"
description: My custom environment
```

Hardware Options:

| Tier | vCPU | RAM | Cost |
|------|------|-----|------|
| CPU Basic (Free) | 2 | 16GB | Free |
| CPU Upgrade | 8 | 32GB | $0.03/hr |

OpenEnv environments support configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions |
| `ENABLE_WEB_INTERFACE` | Auto | Enable web UI |

### Environment-Specific Variables

Some environments have custom variables:

**TextArena:**
```bash
TEXTARENA_ENV_ID=Wordle-v0
TEXTARENA_NUM_PLAYERS=1
TEXTARENA_MAX_TURNS=6
```

**Coding Environment:**
```bash
SANDBOX_TIMEOUT=30
MAX_OUTPUT_LENGTH=10000
```

# DEMO: Deploying to Hugging Face Spaces

This demo walks through the full workflow: create an environment, test locally, deploy to HF Spaces, and use it.

## Step 1: Initialize a new environment

```bash
openenv init my_env
cd my_env
```

This creates the standard OpenEnv structure:

```
my_env/
├── server/
│   ├── app.py           # FastAPI server
│   ├── environment.py   # Your environment logic
│   └── Dockerfile
├── models.py            # Action/Observation types
├── client.py            # Client (EnvClient subclass)
├── openenv.yaml         # Manifest
└── pyproject.toml
```

## Step 2: Run locally

```bash
# Start the server
uv run server

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Test the health endpoint:

```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

## Step 3: Deploy to HF Spaces

```bash
openenv push --repo-id username/my-env
```

Your environment is now live at:
- Web UI: https://username-my-env.hf.space/web
- API Docs: https://username-my-env.hf.space/docs
- Health: https://username-my-env.hf.space/health

```bash
curl https://openenv-echo-env.hf.space/health
# {"status": "healthy"}
```

## Step 4: install the environment

```bash
uv pip install git+https://huggingface.co/spaces/openenv/echo_env
```

## Step 5: Run locally via Docker (optional)

Pull and run the container from the HF registry, or open the [browser](https://huggingface.co/spaces/openenv/echo_env?docker=true):

```bash
# Pull from HF Spaces registry
docker pull registry.hf.space/openenv-echo-env:latest

# Run locally
docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/openenv-echo-env:latest
```

Now connect to your local instance:

```python
import asyncio
from echo_env import EchoEnv, EchoAction

# Async (recommended)
async def main():
    async with EchoEnv(base_url="http://localhost:7860") as env:
        result = await env.reset()
        print(result.observation)
        result = await env.step(EchoAction(message="Hello"))
        print(result.observation)

asyncio.run(main())

# Sync
with EchoEnv(base_url="http://localhost:7860") as env:
    result = env.reset()
    print(result.observation)
    result = env.step(EchoAction(message="Hello"))
    print(result.observation)
```
