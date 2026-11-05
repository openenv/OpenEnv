# OpenClaw Environment

An OpenEnv environment that exposes [OpenClaw](https://github.com/openclaw/openclaw)'s agentic tool capabilities for reinforcement learning training.

## Overview

OpenClaw is a personal AI assistant framework that provides agents with access to:
- **File system operations**: Read, write, and edit files
- **Shell execution**: Run commands inside the deployment container
- **Web research**: Search and fetch web content
- **Memory management**: Search and retrieve context from memory files

This environment wraps these capabilities as MCP (Model Context Protocol) tools, enabling RL agents to learn agentic workflows like coding, research, and automation tasks.

## Quick Start

### Using a Running Server

```python
from openclaw_env import OpenClawEnv

# Connect to a running OpenClaw environment
with OpenClawEnv(base_url="http://localhost:8000") as env:
    env.reset()

    # List available tools
    tools = env.list_tools()
    print([t.name for t in tools])
    # ['exec', 'read', 'write', 'edit', 'web_search', 'web_fetch', 'memory_search', 'memory_get']

    # Execute a shell command
    result = env.call_tool("exec", command="echo 'Hello from OpenClaw!'")
    print(result)  # {"stdout": "Hello from OpenClaw!\n", "exit_code": 0, ...}

    # Create and read a file
    env.call_tool("write", path="hello.txt", content="Hello, World!")
    result = env.call_tool("read", path="hello.txt")
    print(result["content"])  # "Hello, World!"
```

### Using Docker

```python
from openclaw_env import OpenClawEnv

# Start container automatically
env = OpenClawEnv.from_docker_image("openclaw-env:latest")
try:
    env.reset()
    result = env.call_tool("exec", command="pwd")
    print(result)
finally:
    env.close()
```

### Using HuggingFace Space

```python
from openclaw_env import OpenClawEnv

env = OpenClawEnv.from_env("openenv/openclaw-env")
try:
    env.reset()
    tools = env.list_tools()
finally:
    env.close()
```

## Available Tools

### File System Tools

#### `read`
Read contents of a file with optional line range.

```python
result = env.call_tool("read", path="config.py", offset=1, limit=100)
# Returns: {"content": "...", "lines_read": 100, "truncated": False, "path": "/workspace/config.py"}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | required | Path to file (relative or absolute) |
| `offset` | int | 1 | Starting line number (1-indexed) |
| `limit` | int | 2000 | Maximum lines to read |

#### `write`
Write content to a file, creating directories as needed.

```python
result = env.call_tool("write", path="src/main.py", content="print('hello')")
# Returns: {"success": True, "path": "/workspace/src/main.py", "bytes_written": 14}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | required | Path to file |
| `content` | str | required | Content to write |

#### `edit`
Make precise edits by replacing exact text.

```python
result = env.call_tool(
    "edit", path="config.py", old_string="DEBUG = False", new_string="DEBUG = True"
)
# Returns: {"success": True, "replacements": 1, "path": "/workspace/config.py"}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | required | Path to file |
| `old_string` | str | required | Exact text to find |
| `new_string` | str | required | Replacement text |

### Shell Execution

#### `exec`
Execute shell commands in the workspace.

```python
result = env.call_tool("exec", command="ls -la", timeout=30)
# Returns: {"stdout": "...", "stderr": "", "exit_code": 0, "command": "ls -la"}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | str | required | Shell command to execute |
| `workdir` | str | workspace | Working directory |
| `timeout` | int | 30 | Timeout in seconds |

### Web Tools

#### `web_search`
Search the web (simulated in sandbox mode).

```python
result = env.call_tool("web_search", query="python asyncio tutorial", count=5)
# Returns: {"query": "...", "results": [{"title": "...", "url": "...", "snippet": "..."}]}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query |
| `count` | int | 5 | Number of results (1-10) |

#### `web_fetch`
Fetch content from a URL (simulated in sandbox mode).

```python
result = env.call_tool("web_fetch", url="https://example.com", extract_mode="markdown")
# Returns: {"url": "...", "content": "...", "extract_mode": "markdown"}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | required | URL to fetch |
| `extract_mode` | str | "markdown" | "markdown" or "text" |
| `max_chars` | int | 10000 | Maximum characters |

### Memory Tools

#### `memory_search`
Search memory files for relevant context.

```python
result = env.call_tool("memory_search", query="API endpoints", max_results=5)
# Returns: {"query": "...", "results": [{"path": "...", "snippet": "..."}]}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query |
| `max_results` | int | 5 | Maximum results |

#### `memory_get`
Get a snippet from a memory file.

```python
result = env.call_tool("memory_get", path="memory/notes.md", from_line=10, lines=20)
# Returns: {"content": "...", "lines_read": 20, "path": "..."}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | required | Path to memory file |
| `from_line` | int | 1 | Starting line (1-indexed) |
| `lines` | int | 50 | Number of lines |

## Training Examples

### With TRL (GRPO)

```python
from trl import GRPOTrainer, GRPOConfig
from openclaw_env import OpenClawEnv


# Environment factory
def env_factory():
    return OpenClawEnv(base_url="http://localhost:8000")


config = GRPOConfig(
    # ... training config
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    env_factory=env_factory,
)

trainer.train()
```

### With torchforge

A complete torchforge training example for OpenClaw is not yet included in this repository.

## Building Docker Image

```bash
# Build base image first
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

# Build OpenClaw environment
docker build -t openclaw-env:latest -f envs/openclaw_env/server/Dockerfile .

# Run the container
docker run -p 8000:8000 openclaw-env:latest
```

## Development

### Local Testing

```bash
# Install in development mode
cd envs/openclaw_env
pip install -e ".[dev]"

# Run tests
PYTHONPATH=../../src:.. pytest tests/ -v

# Run server locally
uv run --project . server
```

### Running Tests

```bash
# From repository root
PYTHONPATH=src:envs pytest tests/envs/test_openclaw_environment.py -v
```

## Security Notes

- **Sandbox mode**: By default, web tools return simulated results
- **Workspace isolation**: Each episode gets a fresh workspace directory
- **Command restrictions**: Commands run with limited environment variables
- **File access**: Files are constrained to the workspace directory

The workspace and environment-variable allowlist are not a shell sandbox. `exec`
retains the server user's filesystem and network permissions, including access to
other episode directories and any reachable orchestration endpoints. Path checks
also do not prevent a concurrent process from replacing a symlink after validation.

Run only in a disposable, externally isolated container with no credentials,
privileged mode, Docker socket, or host filesystem mounts. Restrict network access
explicitly and isolate the agent process from the orchestration API before using
untrusted commands. These deployment boundaries are not enforced by this environment;
container isolation and RFC 005 alignment still require review.

## License

BSD 3-Clause License (see [LICENSE](../../LICENSE))

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.
