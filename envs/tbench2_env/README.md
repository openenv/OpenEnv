---
title: TB2 Environment Server
emoji: "🧪"
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - terminal-bench-2
  - spaces
---

# TB2 Environment (Terminal-Bench 2)

OpenEnv wrapper for [Terminal-Bench 2](https://github.com/laude-institute/terminal-bench-2) tasks. Supports two execution modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Local** | Runs commands in the server process (no Docker) | Hugging Face Spaces, environments without Docker access |
| **Docker** | Runs each task in its own container | Full TB2.0 fidelity with custom task images |

## Quick Start

```python
from tbench2_env import Tbench2Env, Tbench2Action

env = Tbench2Env(base_url="http://localhost:8000")
result = env.reset(task_id="headless-terminal")
print(result.observation.instruction)

result = env.step(Tbench2Action(action_type="exec", command="ls -la"))
print(result.observation.output)

result = env.step(Tbench2Action(action_type="evaluate"))
print(result.reward, result.done)

env.close()
```

## Building the Docker Image

Before using the environment, build the Docker image:

```bash
# From project root
docker build -t tbench2-env:latest -f envs/tbench2_env/server/Dockerfile .
```

## Environment Details

### Action
**Tbench2Action**: Controls interaction with the TB2 task session

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_type` | str | `"exec"` | Action to perform (`exec`, `write`, `view`, `wait`, `kill`, `write_file`, `evaluate`, `close`) |
| `command` | str | `""` | Shell command or input to send |
| `session_id` | str \| None | `None` | Session ID for streaming processes |
| `block` | bool | `True` | Whether to block until command completes |
| `wait_seconds` | float \| None | `None` | Time to wait (for `wait` action) |
| `file_path` | str | `""` | File path (for `write_file` action) |
| `content` | str | `""` | Content to write (for `write_file` action) |

### Observation
**Tbench2Observation**: Contains the environment response

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | str | Task instruction/prompt from the TB2 task |
| `output` | str | Command output (stdout/stderr) |
| `success` | bool | Whether the action succeeded |
| `error` | str | Error message if action failed |
| `task_id` | str | Current task identifier |
| `task_path` | str | Path to the task directory |
| `session_id` | str \| None | Session ID for streaming processes |
| `action_type` | str | The action type that produced this observation |
| `info` | dict | Additional metadata |

**Execution budgets in the `reset` observation.** `reset` reports the time budgets that bound a single env op in `info`, so clients can derive per-message deadlines (budget plus margin) instead of guessing:

| Key | Modes | Description |
|-----|-------|-------------|
| `verifier_timeout_sec` | local, Docker | Budget for one `evaluate` call: the task's `task.toml` `[verifier].timeout_sec`, else 900. |
| `command_timeout_s` | local only | Per-command budget for `exec` (`TB2_COMMAND_TIMEOUT_S`). Docker mode has no server-side per-command timeout, so it does not report one. |

### State
**Tbench2State**: Server-side state for the task session

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task identifier |
| `task_path` | str | Path to the task directory |
| `session_id` | str | Active session ID |
| `terminal_ready` | bool | Whether the terminal is ready for commands |
| `last_action_type` | str | Last action type executed |
| `last_command` | str | Last command executed |
| `last_output` | str | Output from last command |

## Execution Modes

### Local Mode (Default)

Commands execute directly in the server process. Ideal for HF Spaces where Docker-in-Docker is unavailable.

```bash
# Default - local mode
python -m tbench2_env.server.app

# Or explicitly set mode
TB2_MODE=local python -m tbench2_env.server.app
```

**Note:** Local mode ignores Docker images specified in task.toml. Tasks requiring specific runtime environments may fail.

### Docker Mode

Each task runs in its own Docker container, using the image specified in the task's `task.toml` (tasks that declare no `docker_image` are rejected at reset — there is no host-execution fallback). Agent commands run in the image's own `WORKDIR`, and `evaluate` scores with the task's canonical `tests/test.sh`, staged for exactly the verify window — the same scoring contract as local mode:

```bash
# Enable Docker mode
TB2_MODE=docker python -m tbench2_env.server.app
```

**Requirements:**
- Docker socket mounted at `/var/run/docker.sock`
- Sufficient disk space for container images
- Network access to pull images if not cached

**Environment Variables for Docker Mode:**
- `TB2_MODE=docker` - Enable Docker-backed execution
- Docker socket must be accessible (mounted volume)

## Action Types

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `exec` | Run a shell command | `command`, optionally `block`, `session_id` |
| `write` | Send input to a running session | `session_id`, `command` |
| `view` | Read pending output | `session_id` |
| `wait` | Wait for output | `session_id`, optionally `wait_seconds` |
| `kill` | Terminate a running session | `session_id` |
| `write_file` | Write content to a file | `file_path`, `content` |
| `evaluate` | Run pytest tests, return reward | (none) |
| `close` | Stop and cleanup | (none) |

## Session IDs (Streaming Processes)

`session_id` is **only** required when you start a non-blocking process and want to interact with it (`write`, `view`, `wait`, `kill`). For plain `exec` commands, you can omit it.

Example (Python):
```python
# Start a long-running process
env.step(Tbench2Action(action_type="exec", command="python -i", block=False, session_id="sess1"))

# Send input to it
env.step(Tbench2Action(action_type="write", session_id="sess1", command="print(2+2)\n"))

# Read its output
env.step(Tbench2Action(action_type="view", session_id="sess1"))
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TB2_MODE` | `local` | Execution mode: `local` or `docker` |
| `TB2_TASKS_DIR` | (auto-download) | Path to local Terminal-Bench-2 repo checkout |
| `TB2_OUTPUT_DIR` | `/tmp/tbench2_env_runs` | Directory for session logs and cache |
| `TB2_CACHE_DIR` | `$TB2_OUTPUT_DIR/repo_cache` | Where to extract TB2 repo |
| `TB2_REPO_URL` | (GitHub main.zip) | Repo zip URL for auto-download |
| `TB2_WITHHOLD_TESTS` | `0` | `1`: `reset()` deletes the task's `tests/` and `solution/` from disk (tests kept in server memory, staged at `/tests` only while scoring). Set for RL training so the agent cannot read the expected outputs or the answer; leave off when `TB2_TASKS_DIR` is a working checkout you don't want modified |

## Reward

Binary reward on `evaluate` action:
- `1.0` - All pytest tests pass (exit code 0)
- `0.0` - Tests fail (non-zero exit code)

Intermediate steps return `reward=None`.

## Running the Server

```bash
# Install dependencies
uv sync --all-extras

# Local mode (default, for Spaces)
python -m tbench2_env.server.app --port 8000

# Docker mode (full TB2.0 compatibility)
TB2_MODE=docker python -m tbench2_env.server.app --port 8000

# With local TB2 repo
TB2_TASKS_DIR=/path/to/terminal-bench-2 python -m tbench2_env.server.app
```

## Project Structure

```
tbench2_env/
├── __init__.py              # Module exports (Tbench2Env, Tbench2Action, etc.)
├── README.md                # This file
├── client.py                # Tbench2Env client implementation
├── models.py                # Tbench2Action, Tbench2Observation, Tbench2State
├── openenv.yaml             # OpenEnv configuration
├── pyproject.toml           # Package dependencies
└── server/
    ├── __init__.py          # Server exports
    ├── app.py               # FastAPI application
    ├── tbench2_env_environment.py  # Core environment logic
    └── Dockerfile           # Container image definition
```
